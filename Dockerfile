FROM --platform=linux/amd64 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /app

RUN mkdir -p /app/.cache /.local && chmod -R 777 /app /app/.cache /.local /usr /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libsnappy-dev \
    libboost-all-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    python3-pip \
    tar \
    pkg-config \
    libhdf5-dev \
    findutils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip
RUN /usr/bin/python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . .

ENV CUDA_HOME=/usr/local/cuda
ENV CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
ENV CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:usr/include:$LD_LIBRARY_PATH

ENV MPLCONFIGDIR=/app/.cache
ENV MKL_THREADING_LAYER=GNU
ENV HF_HOME=/app/.cache
ENV XDG_CACHE_HOME=/app/.cache
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

EXPOSE 7860

RUN chmod +x /app/tryon/docker-entrypoint.sh
ENTRYPOINT ["/app/tryon/docker-entrypoint.sh"]