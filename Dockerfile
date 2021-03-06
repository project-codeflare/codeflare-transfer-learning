# docker build -f Dockerfile -t glue-runtime:ray-1.12.0 ./

FROM rayproject/ray:1.12.0-gpu

RUN sudo apt-get update && sudo apt-get install -y \
    build-essential iperf \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN mkdir /home/ray/glue
WORKDIR /home/ray/glue

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
RUN pip install --no-cache-dir torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# change group permissions for running in OCP
RUN sudo chgrp 0 /home/ray/glue
RUN chmod g+w /home/ray/glue
