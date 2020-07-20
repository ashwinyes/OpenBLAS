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
#include <stdint.h>
#include <cblas.h>

#define DOUBLE
#define GEMM   BLASFUNC(dgemm)

int main(int argc, char *argv[]){

  double *a, *b, *c;
  double alpha = 1.0;
  double beta  = 0.0;
  char transa = CblasNoTrans;
  char transb = CblasNoTrans;
  int order = CblasRowMajor;
  int i;
  int m = 2560, n = 2400, k = 2560;
  //int m = 640, n = 600, k = 768;
  int lda = m, ldb = k, ldc = m;

  argc--;argv++;

  if (argc > 0) { m = atol(*argv);            argc--; argv++; }
  if (argc > 0) { n = atol(*argv);            argc--; argv++; }
  if (argc > 0) { k = atol(*argv);            argc--; argv++; }

  printf("M=%d N=%d K=%d\n", m, n, k);

  a = (double *)malloc(sizeof(double) * m * k);
  b = (double *)malloc(sizeof(double) * k * n);
  c = (double *)malloc(sizeof(double) * m * n);

  for (i = 0; i < m * k * COMPSIZE; i++) {
    a[i] = i + 1;
  }
  for (i = 0; i < k * n * COMPSIZE; i++) {
    b[i] = i + 1;
  }
  for (i = 0; i < m * n * COMPSIZE; i++) {
    c[i] = i + 1;
  }

  cblas_dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  return 0;
}

