export OPENBLAS_PARAM_M=$1
export OPENBLAS_PARAM_N=$3
export OPENBLAS_PARAM_K=$2

OMP_NUM_THREADS=1 ../..//zgemm.goto 1 1  
OMP_NUM_THREADS=1 ./zgemm.goto 1 1  
