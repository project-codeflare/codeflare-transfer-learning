# codeflare-transfer-learn

# Running glue_benchmark on OpenShift (OCP)
Assumes:
* A copy of this repository is installed and you have a command line in this directory
* You have the OpenShift CLI installed (instructions are available in the IBM Cloud and OpenShift web consoles if not)
* You have the S3 credentials needed to access the glue datasets and model to evaluate

1. Log into OCP using the oc login command from the OCP web console 
   (Go to the menu under IAM#<your username/email>, then "Copy Login Command").  

2. Use `oc project` to confirm your namespace is as desired. If not:
```
$ oc project {your-namespace}
```

3. Starting from template-s3-creds.yaml, create a personal yaml secrets file with your namespace and S3 credentials. Then register the secrets:
```
$ oc create -f {your-handle}-s3-creds.yaml
```
4. [Required only once] Check if Ray CRD is installed. Install if not.
```
$ oc get crd | grep ray
```
If not there:
```
$ oc apply -f cluster_crd.yaml  
```

5. Create a ray operator in your namespace:
```
$ oc apply -f glue-operator.yaml
```

6. Create a ray cluster in your namespace. Change the min and max number of workers as needed (around line 100)
```
 $ oc apply -f glue-cluster.yaml 
```

7. When the ray cluster head and worker pods are in ready state, copy the application driver to the head node:
```
$ oc get po --watch
$ oc cp glue_benchmark.py glue-cluster-head-XXXXX:/home/ray/glue
```

8. exec into the head node and run the application. For example:
```
$ oc exec -it glue-cluster-head-cjgzk -- /bin/bash
(base) 1000650000@glue-cluster-head-cjgzk:~/glue$ nohup ./glue_benchmark -b {bucket-name} -m roberta-base -t WNLI 2>&1 &
```
This will evaluate the roberta-base model against the WNLI task with 10 different seeds

9. monitor the progress using nohup.out. The evaluation results will be in /tmp/summary.

10. When finished, clean up the active resources in your project:
```
$ oc delete -f glue-cluster.yaml
$ oc delete -f glue-operator.yaml
```
