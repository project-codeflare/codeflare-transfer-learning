#!/usr/bin/env python

# Copyright 2021 IBM Corp.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import datetime
import tempfile
import boto3
import tarfile
import subprocess
import ray
import json
import argparse
from glob import glob
import logging
import socket
import re


# ------------ validate S3 -----------
# Hard to diagnose without these checks

def Validate_S3(bucket,model,gluedata):
  param = os.environ.get('AWS_ACCESS_KEY_ID')
  if param == None:
    status = ['ERROR','AWS_ACCESS_KEY_ID is missing from environment']
    return False,status
  param = os.environ.get('AWS_SECRET_ACCESS_KEY')
  if param == None:
    status = ['ERROR','AWS_SECRET_ACCESS_KEY is missing from environment']
    return False,status
  param = os.environ.get('ENDPOINT_URL')
  if param == None:
    status = ['ERROR','ENDPOINT_URL is missing from environment']
    return False,status

  client = boto3.client(
    's3',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
    endpoint_url = os.environ.get('ENDPOINT_URL')
  )

  try:
    check = client.head_bucket(Bucket=bucket)
  except Exception as e:
    errmsg = f"bucket={bucket} not found"
    return False,errmsg
  
  try:
    check = client.head_object(Bucket=bucket, Key=model)
  except Exception as e:
    errmsg = f"object={model} not found in bucket={bucket}"
    return False,errmsg

  try:
    check = client.head_object(Bucket=bucket, Key=gluedata)
  except Exception as e:
    errmsg = f"object={gluedata} not found in bucket={bucket}"
    return False,errmsg

  return True,"all good"
  
# ------------ detached ray actor: DataRefs -----------
# pulls data from S3 and caches in Plasma for local scaleout
# S3 credentials must be defined in the env

@ray.remote
class DataRefs:
  def __init__(self,bucket,model,gluedata):
    self.glue_state = 'unavailable'
    self.model_state = 'unavailable'
    self.client = boto3.client(
      's3',
      aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
      aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
      endpoint_url = os.environ.get('ENDPOINT_URL')
    )
    self.bucket = bucket
    self.model_key = model
    self.glue_key = gluedata

  # get glue_data from s3 and put it in plasma
  def cache_glue(self):
    if not self.glue_state == 'unavailable':
      return True
    print("try to get glue from s3")
    try:
      dataobject = self.client.get_object(Bucket=self.bucket, Key=self.glue_key)
      glue_data = dataobject['Body'].read()
      self.glueref = ray.put(glue_data)
      self.glue_state = gluedata
      return True
    except Exception as e:
      print("Unable to retrieve/put object contents: {0}\n\n".format(e))
      ray.actor.exit_actor()
      return False

  # get model_data from s3 and put it in plasma
  def cache_model(self):
    if not self.model_state == 'unavailable':
      return True
    print("try to get model from s3")
    try:
      dataobject = self.client.get_object(Bucket=self.bucket, Key=self.model_key)
      model_data = dataobject['Body'].read()
      self.modelref = ray.put(model_data)
      self.model_state = model
      return True
    except Exception as e:
      print("Unable to retrieve/put object contents: {0}\n\n".format(e))
      ray.actor.exit_actor()
      return False

  def get_glue_dataref(self):
    if self.cache_glue():
      return self.glueref
    else:
      return None

  def get_model_dataref(self):
    if self.cache_model():
      return self.modelref
    else:
      return None

  def get_state(self):
    return([f"glue={self.glue_state}", f"model={self.model_state}"])


# -------------------- process_task -----------------  
# process_task first checks if the glue datasets and the model to test are present
#   if not, it calls the DataRefs actor to provide references to the data
# Two log streams are created: a debug level stream to stdout and an info level to file
# The results are packed into a python hashmap and returned

@ray.remote(num_gpus=1)
def process_task(dataRefs,bucket,model,gluedata,task,seed,LR,savemodel):
  # check if S3 credentials are set and objects look accessible
  rc,msg = Validate_S3(bucket,model,gluedata)
  if not rc:
    taskres = {}
    taskres['ERROR'] = msg
    return taskres

  # clean and recreate result directory
  resultdir = OutDir(model,task,seed,LR)
  subprocess.run(['rm', '-rf', resultdir])
  subprocess.run(['mkdir', '-p', resultdir])

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setLevel(logging.DEBUG)
  logger.addHandler(consoleHandler)
  fileHandler = logging.FileHandler(f"{resultdir}/log.log")
  fileHandler.setLevel(logging.INFO)
  logger.addHandler(fileHandler)

  # Check if glue data directory exists locally
  if not os.path.isdir('./'+gluedata):
    try:
      logger.info("Get glue dataset tarball from data actor")
      time_start = time.time()
      ref = ray.get(dataRefs.get_glue_dataref.remote())
      if ref == None:
        logger.warning(f"Could not get {gluedata} tarball from data actor")
        return None

      glue_dataset = ray.get(ref)
      time_done = time.time()
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      logger.info(f"{st} getting tarball length={len(glue_dataset)} took {time_done-time_start:.2f}s")
      tmpdata = f"/tmp/{gluedata}.tgz"
      f = open(tmpdata, "wb")
      f.write(glue_dataset)
      f.close
      time_start = time.time()
      file = tarfile.open(tmpdata)
      file.extractall('./')
      file.close()
      time_done = time.time()
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      logger.info(f"{st} unpacking {gluedata}.tgz took {time_done-time_start:.2f}s")

    except Exception as e:
      logger.warning("Unable to retrieve/unpack glue dataset: {0}".format(e))
      return None

  else:
    logger.info("Reusing previous existing glue-dataset")

  # Check if model directory exists
  if not os.path.isdir('./'+model):
    try:
      logger.info("Get model tarball from data actor")
      time_start = time.time()
      ref = ray.get(dataRefs.get_model_dataref.remote())
      if ref == None:
        logger.warning("Could not get {model} tarball from data actor")
        return None

      model_data = ray.get(ref)
      time_done = time.time()
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      logger.info(f"{st} getting tarball length={len(model_data)} took {time_done-time_start:.2f}s")
      tmpdata = f"/tmp/{model}.tgz"
      f = open(tmpdata, "wb")
      f.write(model_data)
      f.close
      time_start = time.time()
      file = tarfile.open(tmpdata)
      file.extractall('./')
      file.close()
      time_done = time.time()
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      logger.info(f"{st} unpacking {model}.tgz took {time_done-time_start:.2f}s")

    except Exception as e:
      logger.warning("Unable to retrieve/unpack model data: {0}".format(e))
      return None

  else:
    logger.info(f"Reusing {model} directory")

  logger.info(f"Processing task {task} seed {seed} with model {model}")

  # Pull run_glue.py into local pod
  # This code version must match the transformer version being used
  if not os.path.isfile('./run_glue.py'):
    subprocess.run(['wget', 'https://raw.githubusercontent.com/huggingface/transformers/b0892fa0e8df02d683e05e625b3903209bff362d/examples/text-classification/run_glue.py'])

  # change location of transformer cache to a writable directory
  os.environ['TRANSFORMERS_CACHE'] = '/tmp/cache/'

  runargs = ["python","./run_glue.py"]
  runargs.extend(["--model_name_or_path",model])
  runargs.extend(["--task_name",task])
  runargs.extend(["--do_train","--do_eval"])
  runargs.extend(["--data_dir",f"{gluedata}/{task}"])
  runargs.extend(["--max_seq_length","128"])
  runargs.extend(["--per_device_train_batch_size","32"])
  runargs.extend(["--learning_rate",LR])
  runargs.extend(["--num_train_epochs","3.0"])
  runargs.extend(["--save_steps","50000"])
  runargs.extend(["--save_total_limit","0"])
  runargs.extend(["--seed",seed])
  runargs.extend(["--overwrite_output_dir","--output_dir",resultdir])

  # use this regex to exclude debug content from logfile
  p = re.compile(r".*(Epoch|Iteration|Evaluation): .*(s/it|it/s)].*")

  # finally, do the work
  time_start = time.time()
  proc = subprocess.Popen(runargs,stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
  for line in proc.stdout:
    if re.match(p,line) is None:
      if not line == "\n":
        logger.info(line.rstrip())
    else:
      logger.debug(line.rstrip())
  proc.wait()
  time_proc = time.time()-time_start

  # flush logfile
  logger.removeHandler(consoleHandler)
  logger.removeHandler(fileHandler)
  del logger, consoleHandler, fileHandler

  results = PackResults(model,task,seed,LR,time_proc,savemodel)

  # clean up local result directory
  subprocess.run(['rm', '-rf', resultdir])

  return results


# ------------------ Return subtask output directory name
def OutDir(model,task,seed,LR):
  taskl = task.lower()
  return f"result/{model}/{task}/lr-{LR}/{taskl}_seed-{seed}_lr-{LR}_TBATCH-32"


# ------------------ PackResults
# Puts selected info, files, and optionally a reference to the generated subtask model, into a python hashmap

def PackResults(model,task,seed,LR,time,savemodel):
  dir = OutDir(model,task,seed,LR)
  files = glob(os.path.join(dir, f"eval_results_*.txt"))
  files.append(os.path.join(dir, "log.log"))
  taskres = {}
  taskres["subtask"] = f"{task}_seed-{seed}_lr-{LR}"
  taskres["time"] = time
  taskres["hostname"] = socket.gethostname()
  for f in files:
    with open(f, "rb") as afile:
      data = afile.read()
    taskres[os.path.basename(f)] = data

  # put the model in plasma and reference in hashmap
  if savemodel:
    f = os.path.join(dir, "pytorch_model.bin")
    if os.path.isfile(f):
      with open(f, "rb") as afile:
        data = afile.read()
      taskres["pytorch_model.bin"] = ray.put(data)

  return taskres
  
# -------------------- MAIN ------------------

parser = argparse.ArgumentParser(description='Driver for run_glue')
parser.add_argument('-m',"--model", required=True,
                    help="S3 Key and local directory name of base model, e.g. roberta-base")
parser.add_argument('-g',"--gluedata", default="glue_data",
                    help="S3 key and local directory name of glue dataset (Default=glue_data)")
parser.add_argument('-b',"--bucket", required=True, help="S3 bucket name")
parser.add_argument('-t','--tasks', nargs='+',
                    # required MRPC data missing from public download
                    # help="tasks to run, e.g. -t WNLI CoLA (Default=WNLI STS-B CoLA RTE MRPC SST-2 MNLI QNLI QQP)",
                    # default=['WNLI','STS-B','CoLA','RTE','MRPC','SST-2','MNLI','QNLI','QQP'], action='store')
                    help="tasks to run, e.g. -t WNLI CoLA (Default=WNLI STS-B CoLA RTE SST-2 MNLI QNLI QQP)",
                    default=['WNLI','STS-B','CoLA','RTE','SST-2','MNLI','QNLI','QQP'], action='store')
parser.add_argument('-s','--seeds', nargs='+', default=list(range(38,48)), action='store',
                    help="seeds to run, e.g. -s 38 39  (Default=38 39 40 41 42 43 44 45 46 47)")
parser.add_argument('-l',"--learning_rate", default="2e-5",help="Learning Rate (Default=2e-5)")
parser.add_argument('-M',"--savemodel", default=False,help="Save model for each task (Default=False)")
parser.add_argument('-r',"--ray", default="glue-cluster-ray-head:10001",help="ray_service:port")
parser.add_argument('-v',"--verbose", default=False,help="show remote consoles (Default=False)")
args = parser.parse_args()

model=args.model
gluedata=args.gluedata
bucket=args.bucket
tasks=args.tasks
seeds=[str(x) for x in args.seeds]
LR=args.learning_rate
savemodel=args.savemodel
ray_service=args.ray
verbose=args.verbose

# create logger for driver stdout and logfile
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
logger.addHandler(consoleHandler)
fileHandler = logging.FileHandler("/tmp/gluejob.console")
fileHandler.setLevel(logging.INFO)
logger.addHandler(fileHandler)

st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
logger.info(f"\n{st} Starting Glue benchmark ---------------")
logger.info(f"model: {model}")
logger.info(f"gluedata: {gluedata}")
logger.info(f"bucket: {bucket}")
logger.info(f"tasks: {' '.join(tasks)}")
logger.info(f"seeds: {' '.join(seeds)}")
logger.info(f"learning_rate: {float(LR)}")
logger.info(f"savemodel: {savemodel}")
logger.info(f"ray_service: {ray_service}")

# connect to ray cluster
# when running outside of OCP, need to have a local file defining S3 credentials
# when running in OCP credentials need to be populated in the environment from k8s secrets
if os.path.isfile('./s3_env.json'):
  envdata = open('./s3_env.json',)
  s3_env = json.load(envdata)
  ray.init("ray://"+ray_service,runtime_env=s3_env,log_to_driver=verbose,namespace="ibm-glue")
else:
  ray.init("ray://"+ray_service,log_to_driver=verbose,namespace="ibm-glue")

data_actor_name = 'DataRefsActor'
# check if data actor exists and create if not
# a namespace is required to find a previously persisted actor instance
try:
  dataRefs = ray.get_actor(data_actor_name)
  state = ray.get(dataRefs.get_state.remote())
  logger.info(f" Found actor={data_actor_name} with state {state}")
except Exception as e:
  logger.info(f"  actor={data_actor_name} not found ... deploy it")
  dataRefs = DataRefs.options(name=data_actor_name,lifetime="detached").remote(bucket,model,gluedata)
  state = ray.get(dataRefs.get_state.remote())
  logger.info(f"  actor deployed with state {state}")

# submit all subtasks at the same time
tasks = [process_task.remote(dataRefs,bucket,model,gluedata,task,str(seed),LR,savemodel) for task in tasks for seed in seeds]
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
logger.info(f"{st} Submitted {len(tasks)} subtasks")

# wait for all to be done, one at a time
# TODO handle remote processing exceptions
incomplete = tasks
complete = []
while len(complete) < len(tasks):
  onedone, incomplete = ray.wait(incomplete, num_returns=1, timeout=None)
  results = ray.get(onedone)
  complete.append(onedone)
  taskres = results[0]

  st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
  if taskres == None:
    logger.info(f"{st} received None result")
    continue
  if "ERROR" in taskres:
    logger.info(f"{st} Fatal error: {taskres['ERROR']}")
    sys.exit()
  logger.info(f"{st} {taskres['subtask']} took {taskres['time']:.1f}s on {taskres['hostname']} ... {len(complete)} of {len(tasks)} subtasks done")

  # copy results to a known place for access from outside pod
  outfolder = f"/tmp/summary/{taskres['subtask']}"
  subprocess.run(['mkdir', '-p', outfolder])

  for key in taskres.keys():
    if key == 'subtask' or key == 'time' or key == 'hostname':
      continue
    f = open(outfolder+'/'+key, "wb")
    if not key == 'pytorch_model.bin':
      f.write(taskres[key])
    else:
      # get model from plasma and store locally
      time_start = time.time()
      plasobj = taskres[key]
      modelbin = ray.get(plasobj)
      del (plasobj)
      time_pull = time.time()-time_start
      st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
      logger.info(f"{st}  took {time_pull:.1f}s to pull model from plasma with length={len(modelbin)}")
      f.write(modelbin)
      f.close
