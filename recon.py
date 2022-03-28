import sys
import os.path as op
import argparse
import time
import socket

import cupy as cp
from mpi4py import MPI

from emc import EMC

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='emc_config.ini',
                        help='Path to configuration file (default: emc_config.ini)')
    parser.add_argument('-d', '--devices', default = 'device.txt',
                        help='Path to devices file')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_proc = comm.size
    if args.devices is None:
        if num_proc == 1:
            print('Running on default device 0')
        else:
            print('Require a "devices" file if using multiple processes (one number per line)')
            sys.exit(1)
    else:
        with open(args.devices) as f:
            dev = int(f.readlines()[rank].strip())
            print('Rank %d: %s (Device %d)' % (rank, socket.gethostname(), dev))
            sys.stdout.flush()
            cp.cuda.Device(dev).use()

    recon = EMC(args.config_file, num_streams=args.streams)
    logf = open(op.join(recon.output_folder, 'EMC.log'), 'w')
    if rank == 0:
        logf.write('Iter  time(s)  change\n')
        logf.flush()
        avgtime = 0.
        numavg = 0
    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        sys.stderr.write('\r%d/%d (%f s)\n'% (i+1, args.num_iter, etime-stime))
        if rank == 0:
            norm = float(cp.linalg.norm(cp.array(recon.model) - m0))
            logf.write('%-6d%-.2e %e\n' % (i+1, etime-stime, norm))
            print('Change from last iteration: ', norm)
            logf.flush()
            if i > 0:
                avgtime += etime-stime
                numavg += 1
    if rank == 0 and numavg > 0:
        print('\n%.4e s/iteration on average' % (avgtime / numavg))
    logf.close()

if __name__ == '__main__':
    main()
