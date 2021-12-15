# docker build -f Dockerfile -t glue-runtime:ray-1.9.0 ./

FROM rayproject/ray:1.9.0-gpu

# make and cmake is required for installing
RUN sudo apt-get update && sudo apt-get install -y \
    build-essential \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN mkdir /home/ray/glue
WORKDIR /home/ray/glue

# install requirements.txt
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

#install torch
RUN pip install --no-cache-dir torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# install boto3
RUN pip install --no-cache-dir boto3

# change group permissions for running in OCP
RUN sudo chgrp 0 /home/ray/glue
RUN sudo chmod g+w /home/ray/glue
