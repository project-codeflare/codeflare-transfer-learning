# docker build -f Dockerfile -t glue-runtime:ray-1.9.0 ./

FROM rayproject/ray:1.9.0-gpu

RUN mkdir /home/ray/glue
WORKDIR /home/ray/glue

RUN pip install --no-cache-dir torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir transformers

# change group permissions for running in OCP
RUN sudo chgrp 0 /home/ray/glue
RUN chmod g+w /home/ray/glue
