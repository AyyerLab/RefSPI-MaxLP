# MaxLP algorithm for reference-enhanced SPI
Maximum Likelihood Phaser (MaxLP) for holographic, or reference-enhanced single particle imaging (SPI) experiments. For the description of the experiment and reconstruction algorithm, see the following papers:

 * K Ayyer, "Reference-enhanced x-ray single-particle imaging." [Optica 7.6 (2020): 593-601](https://doi.org/10.1364/OPTICA.391373)
 * A Mall & K Ayyer, "Holographic single particle imaging for weakly scattering, heterogeneous nanoscale objects." [arXiv:2210.10611](https://arxiv.org/abs/2210.10611)

## Installation
This is a pure python 3 package, which has the following dependencies:
 * cupy, which in turn needs CUDA
 * numpy
 * h5py
 * mpi4py (for multiple GPUs)

## Usage
To perform a quick simulation, run the following commands:
```sh
$ mkdir -p data/test/
$ ./make_data.py -c config_sim.ini
$ ./recon.py -c config_sim.ini 1
```
The output can be found in `data/test/output_001.h5`


If you have access to multiple GPUs, you can parallelize the reconstruction using MPI. Before one can run the reconstruction, you have to create a 'devices' file which tells the program which GPU number each rank must run on. This is a simple text file with one number per line, referring to the GPU ID as seen by `nvidia-smi`.
```
$ mpirun -np 4 python recon.py -d devices.txt 10
```
In order to see benefits from MPI, you should increase the number of frames and/or the number of rotational samples.
