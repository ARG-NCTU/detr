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

###################################### ROS2 ####################################

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-dev-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

##################################### Additional Setup #########################################

# Set ownership and permissions for the user
RUN chown -R ${USER}:${USER} ${HOME}/ && \
    echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]
