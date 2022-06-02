FROM nvcr.io/nvidia/pytorch:22.04-py3
EXPOSE 8888

ADD "$pwd" /workspace

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -p 8888:8888 -v "$(pwd)":/workspace storygen/pororo:1.0