export OPENBLAS_PARAM_M=${1:-1}
export OPENBLAS_PARAM_N=${2:-1}
export OPENBLAS_PARAM_K=${3:-1}
export OPENBLAS_ALPHA=${4:-1}
export OPENBLAS_BETA=${5:-1}

export OMP_NUM_THREADS=1
export OPENBLAS_INIT_TYPE=3
export OPENBLAS_DEBUG=0

./benchmark/dgemm.opt 1 1 

./benchmark/dgemm.opt 1 1 |& tail -n +5 > opt.txt
./benchmark/dgemm.base 1 1 |& tail -n +5 > base.txt

#tail -n +5 opt.txt > opt.txt
#tail -n +5 base.txt > base.txt

diff opt.txt base.txt
