# Experimental data reconstruction

The data we need is available in the CXIDB as [ID-18](http://cxidb.org/id-18.html). Download the following files from there into your data directory:
 * `detector.dat`
 * `025photons.dat`
 * `115photons.dat`

The first file contains information about the detector geometry. The other two files are two different datasets in a custom sparse format, where locations of photons are indicated by their pixel umber. The `detector.dat` geometry file tells us where those pixels are located in the 2D array.

A conversion script has been provided for you in this folder, which will convert these data frames into the h5 format we are using.

## Conversion

Go to the main directory and run:
```
$ ./convert/cxidb.py <path_to_photons_file> <path_to_detector.dat>
```
This should take a minute or so for each file.

## Reconstruction

Modify the configuration file (or create a new one) and set the following parameters:
```ini
size = 185
in_photons_file = data/115photons.h5
```
in the appropriate sections (`[parameters]` and `[emc]` respectively).

Run the reconstruction code as usual (with or without MPI):
```
$ ./emc.py 40
or
$ mpirun -np 2 ./emc.py -d devices.txt 40
```

You can examine the output in `data.model_040.npy`. For the other dataset with lower flux, you will have to do many more iterations (~400) in order to converge.
