##################################################
#### SETUP ROCM-6.1.2 and CLBlas and rocBLAS #####
##################################################
FROM rocm/dev-ubuntu-22.04:6.1.2 AS rocm

# Login as root user.    
USER root

# Install dependencies and rocm-5.7
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    sudo wget git cmake rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev make build-essential \
    ocl-icd-opencl-dev opencl-headers clinfo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup rocm-user, add password-less sudo and set workdir.
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
RUN useradd --create-home -G sudo,video --shell /bin/bash rocm-user
USER rocm-user
WORKDIR /home/rocm-user
ENV PATH "${PATH}:/opt/rocm/bin"

########################################
######### PRE-INSTALL CLBLAST ##########
########################################
FROM rocm AS rocm-clblas

# Login as root user.    
USER root

# Set env for building and installing clblas
ENV ROCM_PATH=/opt/rocm
ENV CLBlast_DIR=/usr/lib/cmake/CLBlast

# Set env for building and installing clblas
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
RUN git clone https://github.com/CNugteren/CLBlast.git \
    && cd CLBlast \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install

#######################################
########## INSTALL CLBLAST ############
####################################### 
FROM rocm-clblas AS clblas-installer

# Set env for building and installing clblas
ENV ROCM_PATH=/opt/rocm
ENV CLBlast_DIR=/usr/lib/cmake/CLBlast

# Login as rocm-user.    
USER rocm-user

# Install required packages
RUN sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    git gcc \
    build-essential \
    python3 python3-dev python3-pip \
    libclblast-dev libopenblas-dev libaio-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libamdrocopencl.so" | sudo tee /etc/OpenCL/vendors/amd.icd \
    # && sudo ln -s /usr/bin/python3 /usr/bin/python \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

######################################
########## PRE-INSTALL IFW ###########
######################################
FROM clblas-installer AS insanely-fast-whisper-pre-install
# # Login as rocm-user.    
USER rocm-user
ENV PATH="/home/rocm-user/.local/bin:${PATH}"

# Install specific packages using pip
RUN sudo apt-get update -y && sudo apt-get upgrade -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    ffmpeg nano pipx \
    migraphx half \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the workdir of the application    
WORKDIR /app

# Copy files to workdir and change permissions.
COPY requirements.txt /app

# Install specific packages using pip
RUN pip install -U pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pipx install insanely-fast-whisper

# Install torch and onnxruntime
# RUN yes | pip uninstall onnxruntime-rocm numpy

##################################
########## INSTALL IFW ###########
##################################
FROM insanely-fast-whisper-pre-install AS insanely-fast-whisper

# # Login as rocm-user.    
USER rocm-user

# Set the workdir of the application
WORKDIR /app

# Inject packages into insanely-fast-whisper pipx environment  
# RUN pipx runpip insanely-fast-whisper install --no-cache-dir https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl numpy==1.26.4 \
    # && pipx runpip insanely-fast-whisper install --no-cache-dir --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
RUN pipx runpip insanely-fast-whisper install --no-cache-dir -r requirements-onnxruntime-rocm.txt \
    && pipx runpip insanely-fast-whisper install --no-cache-dir --force-reinstall -r requirements-torch-rocm.txt

# Copy all files to workdir and change permissions.
COPY --chown=rocm-user:rocm-user . /app

# Port to expose
# EXPOSE 7860

# Set up Gradio host
# ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set AMDGPU envs
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Run the application
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]