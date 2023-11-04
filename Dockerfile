FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install compiler and libraries
RUN apt update && apt install gcc mpich libx11-dev libomp-dev xorg check -y

# Install cmake
RUN apt update && apt install -y software-properties-common lsb-release wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && apt install kitware-archive-keyring && rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt update && apt install cmake -y

# Install develop tools
RUN apt update && apt install gdb git clang sudo -y

RUN useradd -ms /bin/bash csc5001

USER csc5001
WORKDIR /home/csc5001
