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
import boto3
import argparse

parser = argparse.ArgumentParser(description='S3 object checker')
parser.add_argument('-b',"--bucket", required=True, help="S3 bucket name")
parser.add_argument('-k',"--key", help="S3 Key of object in bucket")
args = parser.parse_args()

bucket=args.bucket
objkey=args.key

param = os.environ.get('AWS_ACCESS_KEY_ID')
if param == None:
  print("AWS_ACCESS_KEY_ID is missing from environment")
  sys.exit()
param = os.environ.get('AWS_SECRET_ACCESS_KEY')
if param == None:
  print("AWS_SECRET_ACCESS_KEY is missing from environment")
  sys.exit()

# if "ENDPOINT_URL" in os.environ:
#     param = os.environ.get('ENDPOINT_URL')
#     print("found endpoint in env with value=|"+param+"|")
#     print(param == "")
# sys.exit()

param = os.environ.get('ENDPOINT_URL')
if param == "":
  print("ENDPOINT_URL is empty, assuming AWS object store")
  client = boto3.client(
    's3',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
  )
else:
  client = boto3.client(
    's3',
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY'),
    endpoint_url = os.environ.get('ENDPOINT_URL')
  )

try:
  check = client.head_bucket(Bucket=bucket)
  print(f"found bucket={bucket}")
except Exception as e:
  print("bucket="+bucket+" not found: {0}\n".format(e))
  sys.exit()

if objkey == None:
  sys.exit()

try:
  check = client.head_object(Bucket=bucket, Key=objkey)
  print(f"found key={objkey} with length={check['ContentLength']}")
except Exception as e:
  print("key="+objkey+" not found in bucket="+bucket+": {0}\n".format(e))
  sys.exit()
