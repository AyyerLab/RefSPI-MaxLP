#!/usr/bin/env python

import sys
import os.path as op
import argparse
import time
import socket

import cupy as cp

from emc import EMC

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='emc_config.ini',
                        help='Path to configuration file (default: emc_config.ini)')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='Device index (default=0)')
    args = parser.parse_args()

    print('Running on device', args.device)
    cp.cuda.Device(args.device).use()

    recon = EMC(args.config_file, num_streams=args.streams)
    logf = open(op.join(recon.output_folder, 'EMC.log'), 'w')
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
        norm = float(cp.linalg.norm(cp.array(recon.model.ravel()) - m0.ravel()))
        logf.write('%-6d%-.2e %e\n' % (i+1, etime-stime, norm))
        print('Change from last iteration: ', norm)
        logf.flush()
        if i > 0:
            avgtime += etime-stime
            numavg += 1
    if numavg > 0:
        print('\n%.4e s/iteration on average' % (avgtime / numavg))
    logf.close()

if __name__ == '__main__':
    main()
