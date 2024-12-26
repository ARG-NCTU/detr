FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

##################################### System Dependencies ######################################

RUN apt-get update -qq && \
    apt-get install -y \
    git \
    vim \
    libgtk2.0-dev \
    python3.8 \
    python3.8-distutils \
    python3-pip && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

##################################### Python Dependencies ######################################

# Install Python dependencies explicitly using python3.8
RUN python3.8 -m pip install --no-cache-dir \
    cython \
    pycocotools \
    submitit \
    scipy \
    onnx \
    onnxruntime \
    pathlib

# Clone and install panopticapi manually using python3.8
RUN git clone --depth 1 https://github.com/cocodataset/panopticapi.git /tmp/panopticapi && \
    python3.8 -m pip install /tmp/panopticapi && \
    rm -rf /tmp/panopticapi

##################################### Additional Setup #########################################

# Set ownership and permissions for the user
RUN chown -R ${USER}:${USER} ${HOME}/ && \
    echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]
