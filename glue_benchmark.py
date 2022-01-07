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

def Validate_S3(logger,bucket,model,gluedata):
  param = os.environ.get('AWS_ACCESS_KEY_ID')
  if param == None:
    logger.warning("AWS_ACCESS_KEY_ID is missing from environment")
    return False
  param = os.environ.get('AWS_SECRET_ACCESS_KEY')
  if param == None:
    logger.warning("AWS_SECRET_ACCESS_KEY is missing from environment")
    return False
  param = os.environ.get('ENDPOINT_URL')
  if param == None:
    logger.warning("ENDPOINT_URL is missing from environment")
    return False

  client = boto3.client(
    's3',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
    endpoint_url = os.environ.get('ENDPOINT_URL')
  )

  try:
    check = client.head_bucket(Bucket=bucket)
  except Exception as e:
    logger.warning(f"bucket={bucket} not found")
    return False

  try:
    check = client.head_object(Bucket=bucket, Key=model)
  except Exception as e:
    logger.warning(f"key={model} not found in bucket={bucket}")
    return False

  try:
    check = client.head_object(Bucket=bucket, Key=gluedata)
  except Exception as e:
    logger.warning(f"key={gluedata} not found in bucket={bucket}")
    return False

  return True


# ------------ detached ray actor: DataRefs -----------
# pulls data from S3 and caches in Plasma for local scaleout
# returns objref for data previously cached
# S3 credentials must be defined in the env

@ray.remote
class DataRefs:
  def __init__(self,bucket):
    self.state = {}
    self.refs = {}
    self.bucket = bucket
    self.client = boto3.client(
      's3',
      aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
      aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
      endpoint_url = os.environ.get('ENDPOINT_URL')
    )

  # check if data for key is already cached
  # if not, try to get data from s3 and put it in plasma
  def Get_dataref(self,key):
    if key in self.state:
      if self.state[key] == 'Cached':
        return self.refs[key]
    print(f"  try to get {key} from s3")
    try:
      dataobject = self.client.get_object(Bucket=self.bucket, Key=key)
      data = dataobject['Body'].read()
      print(f"  try to put {key} data into plasma")
      self.refs[key] = ray.put(data)
      self.state[key] = 'Cached'
      return self.refs[key]
    except Exception as e:
      print("Unable to retrieve/put object contents: {0}\n\n".format(e))
      self.state[key] = 'Failed'
      return None

  def Get_state(self):
    if 0 == len(self.state):
      return ["empty"]
    retstate = []
    for key in self.state.keys():
      retstate.append(f"{key}={self.state[key]}")
    return(retstate)


# ------------ Fetch data to cache -----------
# Calls pulls data from S3 and caches in Plasma for local scaleout
def Fetch_data_to_cache(logger,dataRefs,key):
  try:
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{st} Get {key} data reference from data actor")
    ref = ray.get(dataRefs.Get_dataref.remote(key))
    if ref == None:
      logger.warning(f"Could not get {key} data reference from data actor")
      return False
    return True

  except Exception as e:
    logger.warning(f"Unable to retrieve {key} dataset: {0}".format(e))
    return False

# ------------ Fetch data to local dir -----------
# pulls data from Plasma and unpack in local directory
def Fetch_data_to_local_dir(logger,dataRefs,key):
  if not Fetch_data_to_cache(logger,key):
    return False
  try:
    time_start = time.time()
    ref = ray.get(dataRefs.Get_dataref.remote(key))
    if ref == None:
      logger.warning(f"Could not get {key} data reference from data actor")
      return False

    dataset = ray.get(ref)
    time_done = time.time()
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{st} getting data length={len(dataset)} took {time_done-time_start:.2f}s")
    tmpdata = f"/tmp/{key}.tgz"
    f = open(tmpdata, "wb")
    f.write(dataset)
    f.close
    time_start = time.time()
    file = tarfile.open(tmpdata)
    file.extractall('./')
    file.close()
    time_done = time.time()
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{st} unpacking {key} tarfile took {time_done-time_start:.2f}s")
    return True

  except Exception as e:
    logger.warning(f"Unable to retrieve/unpack {key} dataset: {0}".format(e))
    return False


# -------------------- Process_task -----------------
# process_task first checks if the glue datasets and the model to test are present
#   if not, it requests the data to be fetched from plasma and unpacked locally
# Two log streams are created: a debug level stream to stdout and an info level to file
# The results are packed into a python hashmap and returned

@ray.remote(num_gpus=1)
def Process_task(dataRefs,bucket,model,gluedata,task,seed,LR,savemodel):
  # clean and recreate result directory
  resultdir = ResultDir(model,task,seed,LR)
  subprocess.run(['rm', '-rf', resultdir])
  subprocess.run(['mkdir', '-p', resultdir])

  # create console handler at DEBUG and logfile hander at INFO
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setLevel(logging.DEBUG)
  logger.addHandler(consoleHandler)
  fileHandler = logging.FileHandler(f"{resultdir}/log.log")
  fileHandler.setLevel(logging.INFO)
  logger.addHandler(fileHandler)

  # Reuse local glue data directory or try to create it
  if not os.path.isdir('./'+gluedata):
    if not Fetch_data_to_local_dir(logger, dataRefs, gluedata):
      return ['ERROR',f"Fetch_data_to_local_dir for {gluedata} failed"]
  else:
    logger.info("Reusing previous existing glue-dataset")

  # Reuse local model directory or try to create it
  if not os.path.isdir('./'+model):
    if not Fetch_data_to_local_dir(logger, dataRefs, model):
      return ['ERROR',f"Fetch_data_to_local_dir for {model} failed"]
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


# ------------------ Return remote result directory name
def ResultDir(model,task,seed,LR):
  taskl = task.lower()
  return f"result/{model}/{task}/lr-{LR}/{taskl}_seed-{seed}_lr-{LR}_TBATCH-32"


# ------------------ PackResults
# Puts selected info, files, and optionally a reference to the generated subtask model, into a python hashmap

def PackResults(model,task,seed,LR,time,savemodel):
  dir = ResultDir(model,task,seed,LR)
  files = glob(os.path.join(dir, f"eval_results_*.txt"))
  files.append(os.path.join(dir, "log.log"))
  taskres = {}
  taskres["model"] = model
  taskres["LR"] = LR
  taskres["task"] = task
  taskres["seed"] = seed
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


# ------------------ Return local result directory name
def SummaryDir(model,LR,task,seed):
  if seed == None:
    return f"/tmp/summary/{taskres['model']}_lr-{taskres['LR']}/{taskres['task']}"
  else:
    return f"/tmp/summary/{taskres['model']}_lr-{taskres['LR']}/{taskres['task']}/seed-{taskres['seed']}"


# ------------------ Best_model ----------------
# checks if this is best model yet for task. If so delete last model and return eval score
def Best_model(model,LR,task,seed):
  # per task metric for evaluating best model (from Masayasu Muraoka)
  eval_metric = {
    "cola": "mcc", "mnli": "mnli/acc", "sst-2": "acc", "sts-b": "corr",
    "qqp": "acc_and_f1", "qnli": "acc",  "rte": "acc", "wnli": "acc",
    "mrpc": "f1"
  }
  subtasks_dir = SummaryDir(model,LR,task,None)
  new_subtask_dir = SummaryDir(model,LR,task,seed)
  metric = eval_metric[task.lower()]
  grppr = "eval_"+metric+" = "
  best_score = 0
  bin_dirs = []
  # scan all subtasks for this task, get new score and best previous score
  for f in os.listdir(subtasks_dir):
    if os.path.exists(f"{subtasks_dir}/{f}/pytorch_model.bin"):
      bin_dirs.append(f"{subtasks_dir}/{f}/pytorch_model.bin")

    with open(f"{subtasks_dir}/{f}/eval_results_{task.lower()}.txt") as fp:
      for line in fp:
        if line.startswith(grppr):
          score = float(line.split(grppr)[1])
    if f"{subtasks_dir}/{f}" == new_subtask_dir:
      new_score = score
    else:
      if score > best_score:
        best_score = score

  if new_score <= best_score:
    return False, 0
  # remove previous best model
  for f in bin_dirs:
    os.remove(f)
  return True, new_score


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
parser.add_argument('-M',"--savemodel", action='store_true',help="Save best scoring model for each task (Default=False)")
parser.add_argument('-r',"--ray", default="glue-cluster-ray-head:10001",help="ray_service:port")
parser.add_argument('-v',"--verbose", action='store_true',help="show remote consoles (Default=False)")
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
ray.init("ray://"+ray_service,log_to_driver=verbose,namespace="ibm-glue")

# check if S3 credentials are set and objects look accessible
if not Validate_S3(logger,bucket,model,gluedata):
  logger.error(f"Fatal error verifying S3 access to specified objects")
  sys.exit()

data_actor_name = 'DataRefsActor'
# create data actor if not yet exists
# namespace is required to find a previously persisted actor instance
try:
  dataRefs = ray.get_actor(data_actor_name)
  state = ray.get(dataRefs.Get_state.remote())
  logger.info(f" Found actor={data_actor_name} with state {state}")
except Exception as e:
  logger.info(f"  actor={data_actor_name} not found ... deploy it")
  dataRefs = DataRefs.options(name=data_actor_name,lifetime="detached").remote(bucket)
  state = ray.get(dataRefs.Get_state.remote())

# make sure required datasets are cached in actor
if not Fetch_data_to_cache(logger,dataRefs,gluedata) or not Fetch_data_to_cache(logger,dataRefs,model):
  logger.error(f"Fatal error caching dataset from S3")
  sys.exit()

# submit all subtasks at the same time
tasks = [Process_task.remote(dataRefs,bucket,model,gluedata,task,str(seed),LR,savemodel) for task in tasks for seed in seeds]
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
  if "ERROR" in taskres:
    logger.error(f"{st} Fatal error: {taskres['ERROR']}")
    sys.exit()

  # check for valid result
  if any(x.startswith('eval_results') for x in taskres):
    logger.info(f"{st} {taskres['model']} lr-{taskres['LR']} {taskres['task']} seed-{taskres['seed']}"+
                f" took {taskres['time']:.1f}s on {taskres['hostname']} ... {len(complete)} of {len(tasks)} subtasks done")
  else:
    logger.error(f"{st} {taskres['model']} lr-{taskres['LR']} {taskres['task']} seed-{taskres['seed']}"+
                f" returned ERROR ... {len(complete)} of {len(tasks)} subtasks done")

  # copy results to a known place for access from outside pod; Remove any leftover files
  outfolder = SummaryDir(taskres['model'],taskres['LR'],taskres['task'],taskres['seed'])
  subprocess.run(['mkdir', '-p', outfolder])
  subprocess.run(['rm', '-rf', outfolder+"/*"])

  for key in taskres.keys():
    if key == 'model' or key == 'LR' or key == 'task' or key == 'seed' or key == 'time' or key == 'hostname':
      continue
    if not key == 'pytorch_model.bin':
      f = open(outfolder+'/'+key, "wb")
      f.write(taskres[key])
      f.close
    else:
      # check if this subtask model should be saved
      save,score = Best_model(taskres['model'],taskres['LR'],taskres['task'],taskres['seed'])
      if save:
        # get model from plasma and store locally
        time_start = time.time()
        plasobj = taskres[key]
        modelbin = ray.get(plasobj)
        del (plasobj)
        time_pull = time.time()-time_start
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"{st}   eval={score}, model pull took {time_pull:.1f}s for length={len(modelbin)}")
        f = open(outfolder+'/'+key, "wb")
        f.write(modelbin)
        f.close
