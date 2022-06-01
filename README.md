# StoryGeneration


## Dockerfile 

[Base Image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

### RENAME <None> -> name/version1.0 
```docker tag {image ID} {name/version1.0}```

### RUN Dockerfile
```docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -p 8888:8888 -v "$pwd":/workspace storygen/pororo:1.0```