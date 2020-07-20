/***************************************************************************
Copyright (c) 2014, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#ifdef __CYGWIN32__
#include <sys/time.h>
#endif
#include "common.h"


#undef GEMM

#ifndef COMPLEX

#ifdef DOUBLE
#define GEMM   BLASFUNC(dgemm)
#elif defined(HALF)
#define GEMM   BLASFUNC(shgemm)
#else
#define GEMM   BLASFUNC(sgemm)
#endif

#else

#ifdef DOUBLE
#define GEMM   BLASFUNC(zgemm)
#else
#define GEMM   BLASFUNC(cgemm)
#endif

#endif

#if defined(__WIN32__) || defined(__WIN64__)

#ifndef DELTA_EPOCH_IN_MICROSECS
#define DELTA_EPOCH_IN_MICROSECS 11644473600000000ULL
#endif

int gettimeofday(struct timeval *tv, void *tz){

  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;

  if (NULL != tv)
    {
      GetSystemTimeAsFileTime(&ft);

      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;

      /*converting file time to unix epoch*/
      tmpres /= 10;  /*convert into microseconds*/
      tmpres -= DELTA_EPOCH_IN_MICROSECS;
      tv->tv_sec = (long)(tmpres / 1000000UL);
      tv->tv_usec = (long)(tmpres % 1000000UL);
    }

  return 0;
}

#endif

#if !defined(__WIN32__) && !defined(__WIN64__) && !defined(__CYGWIN32__) && 0

static void *huge_malloc(BLASLONG size){
  int shmid;
  void *address;

#ifndef SHM_HUGETLB
#define SHM_HUGETLB 04000
#endif

  if ((shmid =shmget(IPC_PRIVATE,
		     (size + HUGE_PAGESIZE) & ~(HUGE_PAGESIZE - 1),
		     SHM_HUGETLB | IPC_CREAT |0600)) < 0) {
    printf( "Memory allocation failed(shmget).\n");
    exit(1);
  }

  address = shmat(shmid, NULL, SHM_RND);

  if ((BLASLONG)address == -1){
    printf( "Memory allocation failed(shmat).\n");
    exit(1);
  }

  shmctl(shmid, IPC_RMID, 0);

  return address;
}

#define malloc huge_malloc

#endif

int main(int argc, char *argv[]){

  IFLOAT *a, *b;
  FLOAT *c;
  FLOAT alpha[] = {1.5, 0.5};
  FLOAT beta [] = {3.5, 2.5};
  char transa = 'N';
  char transb = 'N';
  blasint m, n, k, i, j, lda, ldb, ldc;
  int loops = 1;
  int init_type = 0;
  int result_debug = 0;
  int has_param_m = 0;
  int has_param_n = 0;
  int has_param_k = 0;
  char *p;

  int from =   1;
  int to   = 200;
  int step =   1;

  struct timeval start, stop;
  double time1, timeg;
  struct drand48_data drand_buf;

  argc--;argv++;

  if (argc > 0) { from = atol(*argv);            argc--; argv++; }
  if (argc > 0) { to   = MAX(atol(*argv), from); argc--; argv++; }
  if (argc > 0) { step = atol(*argv);            argc--; argv++; }

  if ((p = getenv("OPENBLAS_TRANS"))) {
    transa=*p;
    transb=*p;
  }
  if ((p = getenv("OPENBLAS_TRANSA"))) {
    transa=*p;
  }
  if ((p = getenv("OPENBLAS_TRANSB"))) {
    transb=*p;
  }
  TOUPPER(transa);
  TOUPPER(transb);
  if ((p = getenv("OPENBLAS_INIT_TYPE"))) {
    init_type = atoi(p);
  }
  if ((p = getenv("OPENBLAS_DEBUG"))) {
    result_debug = atoi(p);
  }
  if ((p = getenv("OPENBLAS_ALPHA"))) {
    alpha[0] = atoi(p);
    alpha[1] = alpha[0];
  }
  if ((p = getenv("OPENBLAS_BETA"))) {
    beta[0] = atoi(p);
    beta[1] = beta[0];
  }

  fprintf(stderr, "From : %3d  To : %3d Step=%d : Transa=%c : Transb=%c\n", from, to, step, transa, transb);

  p = getenv("OPENBLAS_LOOPS");
  if ( p != NULL ) {
    loops = atoi(p);
  }

  if ((p = getenv("OPENBLAS_PARAM_M"))) {
    m = atoi(p);
    has_param_m=1;
  } else {
    m = to;
  }
  if ((p = getenv("OPENBLAS_PARAM_N"))) {
    n = atoi(p);
    has_param_n=1;
  } else {
    n = to;
  }
  if ((p = getenv("OPENBLAS_PARAM_K"))) {
    k = atoi(p);
    has_param_k=1;
  } else {
    k = to;
  }

  BLASLONG maxdim = (m > k) ? ((m > n) ? m : n) : ((k > n) ? k : n);
  if (( a = (IFLOAT *)malloc(sizeof(IFLOAT) * maxdim * maxdim * COMPSIZE)) == NULL) {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }
  if (( b = (IFLOAT *)malloc(sizeof(IFLOAT) * maxdim * maxdim * COMPSIZE)) == NULL) {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }
  if (( c = (FLOAT *)malloc(sizeof(FLOAT) * maxdim * maxdim * COMPSIZE)) == NULL) {
    fprintf(stderr,"Out of Memory!!\n");exit(1);
  }

  if (init_type == 0) {
    /* Initialize with random value serially */
#ifdef linux
    srandom(getpid());
#endif

    for (i = 0; i < m * k * COMPSIZE; i++)
      a[i] = ((IFLOAT) rand() / (IFLOAT) RAND_MAX) - 0.5;
    for (i = 0; i < m * k * COMPSIZE; i++)
      b[i] = ((IFLOAT) rand() / (IFLOAT) RAND_MAX) - 0.5;
    for (i = 0; i < m * k * COMPSIZE; i++)
      c[i] = ((IFLOAT) rand() / (IFLOAT) RAND_MAX) - 0.5;

  } else if (init_type == 1) {
#pragma omp parallel private(drand_buf)
    {
      /* Initialize with random value in parallel */
      srand48_r(time(NULL) + omp_get_thread_num(), &drand_buf);
#pragma omp for
      for (i = 0; i < m * k * COMPSIZE; i++) {
        long int random_int;
        lrand48_r(&drand_buf, &random_int);
        a[i] = ((IFLOAT) random_int / (IFLOAT) RAND_MAX) - 0.5;
      }
#pragma omp for
      for (i = 0; i < k * n * COMPSIZE; i++) {
        long int random_int;
        lrand48_r(&drand_buf, &random_int);
        b[i] = ((IFLOAT) random_int / (IFLOAT) RAND_MAX) - 0.5;
      }
#pragma omp for
      for (i = 0; i < m * n * COMPSIZE; i++) {
        long int random_int;
        lrand48_r(&drand_buf, &random_int);
        c[i] = ((IFLOAT) random_int / (IFLOAT) RAND_MAX) - 0.5;
      }
    }
  } else if (init_type == 2) {
    /* Initialize with constant value in parallel */
#pragma omp parallel
    {
#pragma omp for
      for (i = 0; i < m * k * COMPSIZE; i++) {
        a[i] = ((IFLOAT) 1.0 / (IFLOAT) RAND_MAX) - 0.5;
      }
#pragma omp for
      for (i = 0; i < k * n * COMPSIZE; i++) {
        b[i] = ((IFLOAT) 1.0 / (IFLOAT) RAND_MAX) - 0.5;
      }
#pragma omp for
      for (i = 0; i < m * n * COMPSIZE; i++) {
        c[i] = ((IFLOAT) 1.0 / (IFLOAT) RAND_MAX) - 0.5;
      }
    }
  } else if (init_type == 3) {
    /* Initialize with constant value in parallel */
#pragma omp parallel
    {
#pragma omp for
      for (i = 0; i < m * k * COMPSIZE; i++) {
        a[i] = i % 1024 + 1;
      }
#pragma omp for
      for (i = 0; i < k * n * COMPSIZE; i++) {
        b[i] = i % 1024 + 1;
      }
#pragma omp for
      for (i = 0; i < m * n * COMPSIZE; i++) {
        c[i] = i % 1024 + 1;
      }
    }
  }

  fprintf(stderr, "          SIZE                   Flops             Time\n");

  for (i = from; i <= to; i += step) {
    
    timeg=0;

    if (!has_param_m) { m = i; }
    if (!has_param_n) { n = i; }
    if (!has_param_k) { k = i; }

    if (transa == 'N') { lda = m; }
    else { lda = k; }
    if (transb == 'N') { ldb = k; }
    else { ldb = n; }
    ldc = m;

    fprintf(stderr, " M=%4d, N=%4d, K=%4d : \n", (int)m, (int)n, (int)k);
    gettimeofday( &start, (struct timezone *)0);

    for (j=0; j<loops; j++) {
      GEMM (&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
    }

    gettimeofday( &stop, (struct timezone *)0);
    time1 = (double)(stop.tv_sec - start.tv_sec) + (double)((stop.tv_usec - start.tv_usec)) * 1.e-6;

    timeg = time1/loops;
    fprintf(stderr,
	    " %10.2f MFlops %10.6f sec\n",
	    COMPSIZE * COMPSIZE * 2. * (double)k * (double)m * (double)n / timeg * 1.e-6, time1);
    if (result_debug) print_matrix(c, m, n); 
  }

  return 0;
}


void print_matrix(double *A, long int m, long int n)
{
  long int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      fprintf(stderr, "%0lx(%5.2lf) ", *(long int *)(A + i * m + j), *(A + i * m + j));
      //fprintf(stderr, "%lf ", *(A + i * m + j));
    }
    fprintf(stderr, "\n");
  }
}

// void main(int argc, char *argv[]) __attribute__((weak, alias("MAIN__")));
