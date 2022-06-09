#!/usr/bin/env python3

import argparse
import subprocess
from tempfile import NamedTemporaryFile


class TestError(Exception):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test runner for Genome Index MPI assignment')

    parser.add_argument('--nproc', type=int, help='Number of processes', default=4)
    parser.add_argument('-n', type=int, help='Number of genomes', default=3)
    parser.add_argument('-m', type=int, help='Number of queries', default=2)
    parser.add_argument('--genome_in', type=str, help='Prefix of genome file names', default='tests/example/genome')
    parser.add_argument('--queries_in', type=str, help='Queries file name', default='tests/example/queries')
    parser.add_argument('--expected_out', type=str, help='Expected results file name', default='tests/example/results')
    args = parser.parse_args()

    with NamedTemporaryFile() as results_file:
        call_args = ['mpiexec', '-n', str(args.nproc), './genome_index',
                     str(args.n), str(args.m),
                     args.genome_in, args.queries_in, results_file.name]

        call = subprocess.run(call_args)

        if call.returncode != 0:
            raise TestError(f'Test call {" ".join(call_args)} failed with error {call.returncode}')

        with open(args.expected_out, 'r') as expected_results:
            expected = expected_results.read().strip()
            got = results_file.read().decode('UTF-8').strip()

            if expected != got:
                raise TestError(f'\nExpected \n{expected}\nGot\n{got}')

    print("TEST PASSED")
