# hash:sha256:1e0d907ed5234de13428daac9f46f1877daf01cda446dec70c363a01ff074ec6
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

COPY environment.yml /tmp

RUN conda config --add channels conda-forge \
    && conda config --set channel_priority strict \
    && conda update conda -n base \
    && conda init bash \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda activate \
    && conda install mamba \
    && mamba update --all \
    && mamba env update -n base -f /tmp/environment.yml \
    && R -e 'install.packages("USAboundariesData", repos = "http://packages.ropensci.org", type = "source")' \
    && mamba clean -ya

# set locations for code
ENV MOSAIKS_HOME=/
