#!/bin/sh/

docker build -t donghe/ubuntu_mmdet_${1} .

docker run -it \
        --gpus all \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /data:/data \
        -v /home/dhe/workspace/MTDA:/workspace/ \
        --privileged \
        --network=host \
        --ipc=host \
        donghe/ubuntu_mmdet_${1}