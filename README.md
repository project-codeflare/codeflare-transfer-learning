# Scaling transfer learning tasks using CodeFlare on OpenShift Container Platform (OCP)

Foundation models (e.g., BERT, GPT-3, RoBERTa) are trained on a large corpus of data and enable a wide variety of downstream tasks such as sentiment analysis, Q&A, and classification. This repository demonstrates how an enterprise can take a foundation model and run downstream tasks in a parallel manner on a Hybrid Cloud platform.

We use RoBERTa as our base model and run the [GLUE benchmark](https://gluebenchmark.com) that consists of 10 downstream tasks, each with 10 seeds. Each of these tasks is transformed to a [`ray` task](https://docs.ray.io/en/latest/walkthrough.html) by using the `@ray.remote` annotation with a single GPU allocated for each task.

## Setting up an OpenShift cluster

We assume that the user of this repoistory has an [OpenShift](https://www.redhat.com/en/technologies/cloud-computing/openshift) cluster setup with the [GPU operator](https://docs.nvidia.com/datacenter/cloud-native/). We also assume that the end user has [OpenShift CLI](https://docs.openshift.com/container-platform/4.2/cli_reference/openshift_cli/getting-started-cli.html#cli-installing-cli_cli-developer-commands) installed and have their data in an S3 compatible object storage. Python scripts for downloading all GLUE data are avaible [here](https://github.com/nyu-mll/GLUE-baselines#downloading-glue).

## Creating the S3 object for roberta-base and glue_data 

Create the RoBERTa base model S3 object with key="roberta-base" and contents=roberta-base.tgz
```
- git clone https://huggingface.co/roberta-base
- tar -czf roberta-base.tgz roberta-base
```

Create an object for the GLUE datasets with key=glue_data and contents=glue_data.tgz  
```
- python download_glue_data.py --data_dir glue_data --tasks all
- tar -czf glue_data.tgz glue_data
```


## Running glue_benchmark

1. Log into OCP using the `oc login` command (On IBM Cloud, one can go to the menu under IAM#<your username/email>, then "Copy Login Command").  

2. Use `oc project` to confirm your namespace is as desired. You can switch to your desired namespace by:
```
$ oc project {your-namespace}
```

3. Use provided `template-s3-creds.yaml` and create a personal `yaml` secrets file with your namespace and S3 credentials. Then register the secrets:
```
$ oc create -f {your-handle}-s3-creds.yaml
```

4. [Required only once] Check if Ray CRD is installed.
```
$ oc get crd | grep ray
```
You can install the Ray CRD using:
```
$ oc apply -f cluster_crd.yaml  
```

5. Create a `ray` operator in your namespace:
```
$ oc apply -f glue-operator.yaml
```

6. Create a `ray` cluster in your namespace. Change the `min` and `max` number of workers as needed (around line 100)
```
 $ oc apply -f glue-cluster.yaml 
```

7. When the `ray` cluster head and worker pods are in ready state, copy the application driver to the head node:
```
$ oc get po --watch
$ oc cp glue_benchmark.py glue-cluster-head-XXXXX:/home/ray/glue
```

8. Exec into the head node and run the application. For example:
```
$ oc exec -it glue-cluster-head-cjgzk -- /bin/bash
(base) 1000650000@glue-cluster-head-cjgzk:~/glue$ nohup ./glue_benchmark -b {bucket-name} -m roberta-base -t WNLI 2>&1 &
```
This will run the GLUE benchmark, a set of downstream tasks on RoBERTa base model against the WNLI task with 10 different seeds

9. Monitor the progress using `nohup.out`. The evaluation results will be in `/tmp/summary`.

10. When finished, clean up the active resources in your project:
```
$ oc delete -f glue-cluster.yaml
$ oc delete -f glue-operator.yaml
```

## Conclusion

This demonstrates how we can run downstream fine tuning tasks in parallel on a GPU enabled OpenShift cluster. Users can take arbitrary fine tuning tasks written by data scientists and following the pattern in this repository scale it out on their Hybrid Cloud environment. The data will never leave the user's environment and all the GPUs can be leveraged during the transfer learning process. In our experiments, we observed that all the 8 GPUs on four nodes were leveraged for training the various downstream tasks.
