#!/bin/bash
set -e

module load SpectrumMPI/10.1.0

MPICXX=mpixlC

SRC_PATH="./src"
OBJ_PATH="./obj"
BIN_PATH="./bin"
LOG_PATH="./logs"

echo "Clean"
rm -rf ${OBJ_PATH}/*.o
rm -rf ${BIN_PATH}/*

echo "Compile"
${MPICXX} -O3 -o ${BIN_PATH}/mpiwaves ${SRC_PATH}/mpiwaves.cpp
# ${MPICXX} -O3 -qsmp=omp -o ${BIN_PATH}/mpiwaves_omp ${SRC_PATH}/mpiwaves.cpp