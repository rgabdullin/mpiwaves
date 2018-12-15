#!/bin/bash
set -e

module load SpectrumMPI/10.1.0

mpirun -n $(($2*$3*$4)) ./bin/mpiwaves $1 $2 $3 $4 1 100