# CuPADMAN
Determine the in-plane orientation of photon-sparse images and merge to obtain transmission function of unknown mask.

This project includes a simple simulated data generator as well as a reconstruction program. Both programs use the CUDA-based `cupy` replacement to `numpy` to perform the reconstructions. The goal is to keep the code short and easy to comprehend by using as many library functions as possible.

## Installation
This is a pure python package, which has the following dependencies:
 * cupy, which in turn needs CUDA
 * numpy
 * h5py

## Usage
To perform a quick simulation, run the following commands:
```sh
$ mkdir data/
$ ./make_data.py
$ ./emc.py 10
```

A sample `config.ini` file has been provided to specify simulation parameters. One can edit the data generation parameters in the `[make_data]` section and regenerate the frames using the same mask by doing 
```
$ ./make_data.py -d
```

## Features
The reconstruction program implements the EMC algorithm to determine the orientations with a Poisson noise model. It currently runs on a single CUDA-capable GPU. The only unknown is the orientation, and the simulated data has pure Poisson noise.

### Future goals
These are some of the future enhancements we would like to achieve:
 * Include shot-by-shot incident fluence variations, and recover them
 * Include non-uniform background in simulated data and ability to incorporate that information
 * ~~Scale to multiple GPUs, first on same node, but later across nodes~~ [DONE [1078a58](https://github.com/kartikayyer/CuPADMAN/commit/1078a58f9ba1cdc48f816a5606c5f56c5b9ce52a)]
 * Scale to large data sets and fine orientation sampling without running out of memory
 * Use CUDATextureObject API for faster rotations (needs modificatin of cupy)
