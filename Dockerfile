FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install compiler and libraries
RUN apt update && apt install gcc mpich libx11-dev xorg check -y

# Install develop tools
RUN apt update && apt install cmake gdb git -y

RUN useradd -ms /bin/bash csc5001

USER csc5001
WORKDIR /home/csc5001
