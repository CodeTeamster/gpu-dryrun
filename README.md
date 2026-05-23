# gpu-dryrun

`gpu-dryrun` is a standalone CUDA C++ GPU load tool. It runs as a native binary without Python, PyTorch, pip packages, or `run.sh`.

## Requirements

- NVIDIA driver
- CUDA Toolkit with `nvcc`
- GNU Make

NVML is loaded at runtime from the NVIDIA driver library, so `nvml.h` is not required.

## Build

```sh
make -f Makefile.cpp CXX=/path/to/nvcc
```

Output binary:

```sh
./gpu-dryrun
```

Clean:

```sh
make -f Makefile.cpp clean
```

## Usage

Foreground run:

```sh
./gpu-dryrun --gpus 0,1
```

Background daemon:

```sh
./gpu-dryrun up 0,1
./gpu-dryrun status
./gpu-dryrun down
```

If `--gpus` is omitted in foreground mode, the program scans all GPUs and uses suitable ones.

## Process Management (No PID/Log Files)

The tool no longer writes persistent PID or log files.

`status` and `down` locate daemon processes by scanning `/proc` and matching:

- same executable path as the current binary
- internal daemon marker argument

This allows `down` to stop the correct local daemon process without depending on state files.

## GPU Selection and Load Strategy

GPU IDs use the same physical index style as `nvidia-smi`.

A GPU is considered suitable when:

- memory usage < 70%
- GPU utilization < 30%

For each suitable GPU, the program allocates about 60% of free memory and continuously launches CUDA kernels to keep compute load.

## Help

```sh
./gpu-dryrun --help
```
