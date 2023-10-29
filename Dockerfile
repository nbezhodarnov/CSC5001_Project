FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt update && apt install gcc mpich libx11-dev gdb xorg check -y

RUN useradd -ms /bin/bash csc5001

USER csc5001
WORKDIR /home/csc5001
