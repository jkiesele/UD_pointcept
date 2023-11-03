FROM kdlong/pointcept:v4

SHELL ["/bin/bash", "-c"]

USER root

RUN python3 -m pip install wandb

RUN git clone https://github.com/cms-pepr/pytorch_cmspepr.git &&\
    cd pytorch_cmspepr &&\
    pip install -e .

COPY select_knn_cuda.so /pytorch_cmspepr/
