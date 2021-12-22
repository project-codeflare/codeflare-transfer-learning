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
import ray
import argparse

parser = argparse.ArgumentParser(description='ray actor killer')
parser.add_argument('-r',"--ray", default="glue-cluster-ray-head:10001",help="ray_service:port")
parser.add_argument('-n',"--namespace", default="ibm-glue",help="Default=ibm-glue")
parser.add_argument('-a','--actor_name', default="DataRefsActor",help="Default=DataRefsActor")
args = parser.parse_args()

print("trying to kill actor",args.actor_name,"in namespace",args.namespace)
# connect to ray cluster
ray.init("ray://"+args.ray,namespace=args.namespace)

namespace=ray.get_runtime_context().namespace
if not namespace == args.namespace:
  print("namespace",args.namespace,"not found")
  sys.exit(0)
try:
  actor = ray.get_actor(args.actor_name)
#  if actor == None:
#    print("actor",args.actor_name,"not found")
#    sys.exit(0)
except Exception as e:
  print(f"Actor '{args.actor_name}' not found in namespace '{args.namespace}'")
  sys.exit(0)

print("killing actor",actor)
ray.kill(actor)
