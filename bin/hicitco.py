#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np
import pandas as pd

import hicitco


def main():
    parser = argparse.ArgumentParser(description='HiCitco: Hi-C Iterative Correction (ICE)')
    parser.add_argument('fragments', metavar='in-frags.bed', help='fragments file')
    parser.add_argument('contacts', metavar='in-contacts.txt', help='contacts file')
    parser.add_argument('biases', metavar='out-biases.txt', help='output biases file')

    parser.add_argument('--diag', type=int, default=0, help='number of diagonals to remove (default: 0)')
    parser.add_argument('--eps', type=float, default=1e-4, help='stopping condition (when dBias is negligible)')
    parser.add_argument('-i', '--iter', type=int, default=50, help='maximum number of iterations (default: 50)')

    parser.add_argument('--low', metavar='FLOAT', type=float, default=0,
                        help='fraction of bins with the lowest coverage to remove (default: 0)')
    parser.add_argument('--high', metavar='FLOAT', type=float, default=0,
                        help='fraction of bins with the highest coverage to remove (default: 0)')
    parser.add_argument('--auto', action='store_true', help='find the best cutoffs for bin-level filtering')
    parser.add_argument('--last', action='store_true', help='use the last local minima instead of the first one')

    parser.add_argument('-o', dest='output', help='output corrected matrix file')
    parser.add_argument('--chunksize', type=int, default=0, help='load contacts by chunks of INT rows')
    parser.add_argument('--verbose', action='store_true', help='display progress')
    argv = parser.parse_args()

    if not os.path.isfile(argv.fragments):
        sys.stderr.write('{}: no such file or directory\n'.format(argv.fragments))
        exit(1)
    elif not os.path.isfile(argv.contacts):
        sys.stderr.write('{}: no such file or directory\n'.format(argv.counts))
        exit(1)

    if argv.verbose:
        sys.stderr.write('Loading fragments\n')
    fragments = hicitco.load_fragments(argv.fragments)

    if argv.verbose:
        sys.stderr.write('Loading matrix\n')
    mat = hicitco.load_contacts(argv.contacts, fragments, diag=argv.diag, chunksize=argv.chunksize)

    if argv.auto:
        argv.low, argv.high = hicitco.filter_auto(mat, argv.biases + '.png', use_last=argv.last)

    if argv.low:
        if argv.verbose:
            sys.stderr.write('Removing regions with a low coverage\n')
        hicitco.filter_low_bins(mat, p=argv.low)

    if argv.high:
        if argv.verbose:
            sys.stderr.write('Removing regions with a high coverage\n')
        hicitco.filter_high_bins(mat, p=argv.high)

    if argv.verbose:
        sys.stderr.write('Performing ICE normalization\n')
    biases = hicitco.ice_norm(mat, max_iter=argv.iter, eps=argv.eps, verbose=argv.verbose)

    # Saving output
    if argv.verbose:
        sys.stderr.write('Saving biases\n')
    np.savetxt(argv.biases, biases, fmt='%.18e')

    if argv.output:
        if argv.verbose:
            sys.stderr.write('Saving corrected matrix\n')
        compression = 'gzip' if argv.output.lower().endswith('.gz') else None

        # We need to load all contacts (with the diagonals and the regions with a low/high coverage)
        mat = hicitco.load_contacts(argv.contacts, fragments, chunksize=argv.chunksize)

        # Apply biases
        hicitco.update_matrix(mat, np.array(biases, dtype=np.float64).flatten())

        row, col = hicitco.triu(mat).nonzero()
        df = pd.DataFrame({'i': row, 'j': col, 'v': np.array(mat[row, col]).flatten()})
        df.to_csv(argv.output, sep='\t', header=False, index=False, float_format='%.6f', compression=compression)


if __name__ == '__main__':
    main()