# CuPADMAN
Determine the in-plane orientation of photon-sparse images and merge to obtain transmission function of unknown mask.

This project includes a simple simulated data generator as well as a reconstruction program. Both programs use the CUDA-based `cupy` replacement to `numpy` to perform the reconstructions. The goal is to keep the code short and easy to comprehend by using as many library functions as possible.

**Note:** Turns out the code's much faster with custom kernels, so some simplicity has been sacrificed for speed.

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
The reconstruction program implements the EMC algorithm [1] to determine the orientations with a Poisson noise model. It currently runs on a single CUDA-capable GPU. The only unknown is the orientation, and the simulated data has pure Poisson noise.

This corresponds to a real experiment [2] to demonstrate the noise-tolerance of both the EMC algorithm and the X-ray detector used to collect the data. The data from that experiment is available on the CXIDB [ID 18](http://cxidb.org/id-18.html).

The data format convention used in this project are inspired by the [Dragonfly](https://github.com/duaneloh/Dragonfly) repository.

### Future goals
These are some of the future enhancements we would like to achieve:
 * ~~Include shot-by-shot incident fluence variations, and recover them~~ \[DONE (trivially) [6e642fe](https://github.com/kartikayyer/CuPADMAN/commit/6e642fe1854e1186882b61244b325a839b3b3b38)\]
 * Include non-uniform background in simulated data and ability to incorporate that information
 * ~~Scale to multiple GPUs, first on same node, but later across nodes~~ \[DONE [1078a58](https://github.com/kartikayyer/CuPADMAN/commit/1078a58f9ba1cdc48f816a5606c5f56c5b9ce52a)\]
 * ~~Scale to large data sets and fine orientation sampling without running out of memory~~ \[DONE [7728cb9](https://github.com/kartikayyer/CuPADMAN/commit/7728cb9a9f7fb377a4ecff4bb9eb4c8a49c861f9)\]
 * Use CUDATextureObject API for faster rotations (needs modification of cupy)

## References
 1. Loh, Ne-Te Duane, and Veit Elser. "Reconstruction algorithm for single-particle diffraction imaging experiments." *Physical Review E* 80, no. 2 (2009): 026705.
 2. Philipp, Hugh T., Kartik Ayyer, Mark W. Tate, Veit Elser, and Sol M. Gruner. "Solving structure with sparse, randomly-oriented x-ray data." *Optics express* 20, no. 12 (2012): 13129-13137.
 3. Ayyer, Kartik, T-Y. Lan, Veit Elser, and N. Duane Loh. "Dragonfly: an implementation of the expand–maximize–compress algorithm for single-particle imaging." *Journal of applied crystallography* 49, no. 4 (2016): 1320-1335.
