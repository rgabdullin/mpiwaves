#!/bin/bash
set -e
module load SpectrumMPI/10.1.0

echo "empty" > logs/test.out
echo "empty" > logs/test.err

bsub -n $(($2*$3*$4)) -o "logs/test.out" -e "logs/test.err" ./wrapper.sh $1 $2 $3 $4
