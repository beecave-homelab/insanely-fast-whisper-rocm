##################################################
#### SETUP ROCM-6.1.2 and CLBlas and rocBLAS #####
##################################################
FROM rocm/dev-ubuntu-22.04:6.1.2 AS rocm

# Login as root user.    
USER root

# Install dependencies and rocm-6.1.2
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    sudo wget git cmake rocsparse-dev hipsparse-dev rocthrust-dev rocblas-dev hipblas-dev make build-essential \
    ocl-icd-opencl-dev opencl-headers clinfo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup rocm-user, add password-less sudo, and set workdir.
COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd
RUN useradd --create-home -G sudo,video --shell /bin/bash rocm-user
USER rocm-user
WORKDIR /home/rocm-user
ENV PATH="${PATH}:/opt/rocm/bin"

########################################
######### PRE-INSTALL CLBLAST ##########
########################################
FROM rocm AS rocm-clblas

# Login as root user.
USER root

# Set environment variables
ENV ROCM_PATH=/opt/rocm \
    CLBlast_DIR=/usr/lib/cmake/CLBlast \
    HSA_OVERRIDE_GFX_VERSION=10.3.0

# Build and install CLBlast
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

# Login as rocm-user.
USER rocm-user

# Install required packages
RUN sudo apt-get update -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    nano pipx git gcc build-essential \
    python3 python3-dev python3-pip \
    libclblast-dev libopenblas-dev libaio-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libamdrocopencl.so" | sudo tee /etc/OpenCL/vendors/amd.icd \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

######################################
########## PRE-INSTALL IFW ###########
######################################
FROM clblas-installer AS insanely-fast-whisper-pre-install

# Add pipx to PATH
ENV PATH="/home/rocm-user/.local/bin:${PATH}"

# Install specific packages using pip
RUN sudo apt-get update -y && sudo apt-get upgrade -y && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \ 
    ffmpeg nano pipx \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the workdir of the application    
WORKDIR /app

# Copy requirement files to workdir and change permissions.
COPY --chown=rocm-user:rocm-user requirements.txt .

# Install dependencies
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt
RUN pipx install insanely-fast-whisper


##################################
########## INSTALL IFW ###########
##################################
FROM insanely-fast-whisper-pre-install AS insanely-fast-whisper-install

# Login as rocm-user.
USER rocm-user

# Set the workdir of the application
WORKDIR /app

# Copy requirement files to workdir and change permissions.
COPY --chown=rocm-user:rocm-user requirements-onnxruntime-rocm.txt .
COPY --chown=rocm-user:rocm-user requirements-torch-rocm.txt .

# Install additional dependencies for insanely-fast-whisper
RUN pipx runpip insanely-fast-whisper install --no-cache-dir -r requirements-onnxruntime-rocm.txt \
    && pipx runpip insanely-fast-whisper install --no-cache-dir --force-reinstall -r requirements-torch-rocm.txt


##############################
########## RUN IFW ###########
##############################
FROM insanely-fast-whisper-install AS insanely-fast-whisper

# Login as rocm-user.
USER rocm-user

# Set the workdir of the application
WORKDIR /app

# Copy the application to the workdir
COPY --chown=rocm-user:rocm-user . .

# Expose port if needed
EXPOSE 7860

# Set up Gradio host if needed
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set AMDGPU envs
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0

# Run the application
# ENTRYPOINT ["/bin/bash", "-c", "./entrypoint.sh -u uploads -t transcripts -l logs -b 10 -v"]
ENTRYPOINT ["python3", "app.py"]