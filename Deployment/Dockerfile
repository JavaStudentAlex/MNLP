FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV APT_FILE=Deployment/apt.txt 
COPY ${APT_FILE} /tmp/apt.txt
RUN apt -qq update \
    && apt -qy upgrade \
    && apt -qy install --no-install-recommends $(cat /tmp/apt.txt) \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/*


RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8


ENV SHELL=/bin/bash \
    HOST_UID=1317 \
    USER_NAME=jovyan \
    USER_HOME=/home/jovyan
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    --home ${USER_HOME} \
    ${USER_NAME}
USER ${USER_NAME}
WORKDIR ${USER_HOME}

ENV CONDA_SOURCE=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh \
    CONDA_DIR=${USER_HOME}/conda
RUN wget -nv -O miniconda.sh ${CONDA_SOURCE} \
    && bash miniconda.sh -b -p ${CONDA_DIR} \
    && . ${CONDA_DIR}/etc/profile.d/conda.sh \
    && conda clean -y -a \
    && rm -rf miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

ENV START_FILE=Deployment/start.sh
COPY ${START_FILE} start.sh

RUN mkdir workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    C_FORCE_ROOT=true \
    REQUIREMENTS_FILE_PATH=requirements.yaml
COPY ${REQUIREMENTS_FILE_PATH} /tmp/requirements.yaml
RUN conda env update --file /tmp/requirements.yaml --name base && conda clean -y -a