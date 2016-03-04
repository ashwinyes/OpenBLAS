/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include "common.h"

#define PREFETCHSIZE 4

int CNAME(BLASLONG n, BLASLONG k1, BLASLONG k2, FLOAT *a, BLASLONG lda, blasint *ipiv, FLOAT *buffer){

  BLASLONG i, j, ip;
  blasint *piv;
  FLOAT *dx1, *dy1;
  FLOAT *dx2, *dy2;
  FLOAT *dx3, *dy3;
  FLOAT *dx4, *dy4;
  FLOAT *dx5, *dy5;
  FLOAT *dx6, *dy6;
  FLOAT *dx7, *dy7;
  FLOAT *dx8, *dy8;
  FLOAT *dx9, *dy9;
  FLOAT *dx10, *dy10;
  FLOAT *dx11, *dy11;
  FLOAT *dx12, *dy12;
  FLOAT *dx13, *dy13;
  FLOAT *dx14, *dy14;
  FLOAT *dx15, *dy15;
  FLOAT *dx16, *dy16;
  FLOAT atemp1, btemp1;
  FLOAT atemp2, btemp2;
  FLOAT atemp3, btemp3;
  FLOAT atemp4, btemp4;
  FLOAT atemp5, btemp5;
  FLOAT atemp6, btemp6;
  FLOAT atemp7, btemp7;
  FLOAT atemp8, btemp8;
  FLOAT atemp9, btemp9;
  FLOAT atemp10, btemp10 ;
  FLOAT atemp11, btemp11;
  FLOAT atemp12, btemp12;
  FLOAT atemp13, btemp13;
  FLOAT atemp14, btemp14;
  FLOAT atemp15, btemp15;
  FLOAT atemp16, btemp16;

  a--;
  ipiv += k1 - 1;

  if (n  <= 0) return 0;
  if (k1 > k2) return 0;

  j = (n >> 4);
  if (j > 0) {
    do {
      piv = ipiv;
      i = k1;

    do {
      ip = *piv;
      piv ++;

      dx1 = a + i;
      dy1 = a + ip;
      dx2 = a + i  + lda * 1;
      dy2 = a + ip + lda * 1;
      dx3 = a + i  + lda * 2;
      dy3 = a + ip + lda * 2;
      dx4 = a + i  + lda * 3;
      dy4 = a + ip + lda * 3;
      dx5 = a + i  + lda * 4;
      dy5 = a + ip + lda * 4;
      dx6 = a + i  + lda * 5;
      dy6 = a + ip + lda * 5;
      dx7 = a + i  + lda * 6;
      dy7 = a + ip + lda * 6;
      dx8 = a + i  + lda * 7;
      dy8 = a + ip + lda * 7;
      dx9 = a + i  + lda * 8;
      dy9 = a + ip + lda * 8;
      dx10 = a + i  + lda * 9;
      dy10 = a + ip + lda * 9;
      dx11 = a + i  + lda * 10;
      dy11 = a + ip + lda * 10;
      dx12 = a + i  + lda * 11;
      dy12 = a + ip + lda * 11;
      dx13 = a + i  + lda * 12;
      dy13 = a + ip + lda * 12;
      dx14 = a + i  + lda * 13;
      dy14 = a + ip + lda * 13;
      dx15 = a + i  + lda * 14;
      dy15 = a + ip + lda * 14;
      dx16 = a + i  + lda * 15;
      dy16 = a + ip + lda * 15;

#ifdef __GNUC__
      __builtin_prefetch(dx1 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx2 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx3 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx4 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx5 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx6 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx7 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx8 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx9 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx10 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx11 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx12 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx13 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx14 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx15 + PREFETCHSIZE, 0, 1);
      __builtin_prefetch(dx16 + PREFETCHSIZE, 0, 1);
#endif

      atemp1 = *dx1;
      btemp1 = *dy1;
      atemp2 = *dx2;
      btemp2 = *dy2;
      atemp3 = *dx3;
      btemp3 = *dy3;
      atemp4 = *dx4;
      btemp4 = *dy4;

      atemp5 = *dx5;
      btemp5 = *dy5;
      atemp6 = *dx6;
      btemp6 = *dy6;
      atemp7 = *dx7;
      btemp7 = *dy7;
      atemp8 = *dx8;
      btemp8 = *dy8;

      atemp9 = *dx9;
      btemp9 = *dy9;
      atemp10 = *dx10;
      btemp10 = *dy10;
      atemp11 = *dx11;
      btemp11 = *dy11;
      atemp12 = *dx12;
      btemp12 = *dy12;

      atemp13 = *dx13;
      btemp13 = *dy13;
      atemp14 = *dx14;
      btemp14 = *dy14;
      atemp15 = *dx15;
      btemp15 = *dy15;
      atemp16 = *dx16;
      btemp16 = *dy16;

      if (ip != i) {
	*dy1 = atemp1;
	*dy2 = atemp2;
	*dy3 = atemp3;
	*dy4 = atemp4;
	*dy5 = atemp5;
	*dy6 = atemp6;
	*dy7 = atemp7;
	*dy8 = atemp8;
	*dy9 = atemp9;
	*dy10 = atemp10;
	*dy11 = atemp11;
	*dy12 = atemp12;
	*dy13 = atemp13;
	*dy14 = atemp14;
	*dy15 = atemp15;
	*dy16 = atemp16;
	*(buffer + 0) = btemp1;
	*(buffer + 1) = btemp2;
	*(buffer + 2) = btemp3;
	*(buffer + 3) = btemp4;
	*(buffer + 4) = btemp5;
	*(buffer + 5) = btemp6;
	*(buffer + 6) = btemp7;
	*(buffer + 7) = btemp8;
	*(buffer + 8) = btemp9;
	*(buffer + 9) = btemp10;
	*(buffer + 10) = btemp11;
	*(buffer + 11) = btemp12;
	*(buffer + 12) = btemp13;
	*(buffer + 13) = btemp14;
	*(buffer + 14) = btemp15;
	*(buffer + 15) = btemp16;
      } else {
	*(buffer + 0) = atemp1;
	*(buffer + 1) = atemp2;
	*(buffer + 2) = atemp3;
	*(buffer + 3) = atemp4;
	*(buffer + 4) = atemp5;
	*(buffer + 5) = atemp6;
	*(buffer + 6) = atemp7;
	*(buffer + 7) = atemp8;
	*(buffer + 8) = atemp9;
	*(buffer + 9) = atemp10;
	*(buffer + 10) = atemp11;
	*(buffer + 11) = atemp12;
	*(buffer + 12) = atemp13;
	*(buffer + 13) = atemp14;
	*(buffer + 14) = atemp15;
	*(buffer + 15) = atemp16;
      }

      buffer += 16;

      i++;
    } while (i <= k2);

      a += 16 * lda;
      j --;
    } while (j > 0);
  }

  if (n & 8) {
    piv = ipiv;

      ip = *piv;
      piv ++;

      dx1 = a + k1;
      dy1 = a + ip;
      dx2 = a + k1  + lda * 1;
      dy2 = a + ip  + lda * 1;
      dx3 = a + k1  + lda * 2;
      dy3 = a + ip  + lda * 2;
      dx4 = a + k1  + lda * 3;
      dy4 = a + ip  + lda * 3;
      dx5 = a + k1  + lda * 4;
      dy5 = a + ip  + lda * 4;
      dx6 = a + k1  + lda * 5;
      dy6 = a + ip  + lda * 5;
      dx7 = a + k1  + lda * 6;
      dy7 = a + ip  + lda * 6;
      dx8 = a + k1  + lda * 7;
      dy8 = a + ip  + lda * 7;

    i = k1;

    do {
#ifdef __GNUC__
    __builtin_prefetch(dx1 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx2 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx3 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx4 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx5 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx6 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx7 + PREFETCHSIZE, 0, 1);
    __builtin_prefetch(dx8 + PREFETCHSIZE, 0, 1);
#endif

      atemp1 = *dx1;
      btemp1 = *dy1;
      atemp2 = *dx2;
      btemp2 = *dy2;
      atemp3 = *dx3;
      btemp3 = *dy3;
      atemp4 = *dx4;
      btemp4 = *dy4;

      atemp5 = *dx5;
      btemp5 = *dy5;
      atemp6 = *dx6;
      btemp6 = *dy6;
      atemp7 = *dx7;
      btemp7 = *dy7;
      atemp8 = *dx8;
      btemp8 = *dy8;

      if (ip != i) {
	*dy1 = atemp1;
	*dy2 = atemp2;
	*dy3 = atemp3;
	*dy4 = atemp4;
	*dy5 = atemp5;
	*dy6 = atemp6;
	*dy7 = atemp7;
	*dy8 = atemp8;
	*(buffer + 0) = btemp1;
	*(buffer + 1) = btemp2;
	*(buffer + 2) = btemp3;
	*(buffer + 3) = btemp4;
	*(buffer + 4) = btemp5;
	*(buffer + 5) = btemp6;
	*(buffer + 6) = btemp7;
	*(buffer + 7) = btemp8;
      } else {
	*(buffer + 0) = atemp1;
	*(buffer + 1) = atemp2;
	*(buffer + 2) = atemp3;
	*(buffer + 3) = atemp4;
	*(buffer + 4) = atemp5;
	*(buffer + 5) = atemp6;
	*(buffer + 6) = atemp7;
	*(buffer + 7) = atemp8;
      }
      ip = *piv;
      piv ++;

      i++;
      dx1 = a + i;
      dy1 = a + ip;
      dx2 = a + i   + lda * 1;
      dy2 = a + ip  + lda * 1;
      dx3 = a + i   + lda * 2;
      dy3 = a + ip  + lda * 2;
      dx4 = a + i   + lda * 3;
      dy4 = a + ip  + lda * 3;
      dx5 = a + i   + lda * 4;
      dy5 = a + ip  + lda * 4;
      dx6 = a + i   + lda * 5;
      dy6 = a + ip  + lda * 5;
      dx7 = a + i   + lda * 6;
      dy7 = a + ip  + lda * 6;
      dx8 = a + i   + lda * 7;
      dy8 = a + ip  + lda * 7;

      buffer += 8;

    } while (i <= k2);

      a += 8 * lda;
  }

  if (n & 4) {
    piv = ipiv;

      ip = *piv;
      piv ++;

      dx1 = a + k1;
      dy1 = a + ip;
      dx2 = a + k1 + lda * 1;
      dy2 = a + ip + lda * 1;
      dx3 = a + k1 + lda * 2;
      dy3 = a + ip + lda * 2;
      dx4 = a + k1 + lda * 3;
      dy4 = a + ip + lda * 3;

    i = k1;

    do {
      atemp1 = *dx1;
      atemp2 = *dx2;
      atemp3 = *dx3;
      atemp4 = *dx4;

      btemp1 = *dy1;
      btemp2 = *dy2;
      btemp3 = *dy3;
      btemp4 = *dy4;

      if (ip != i) {
	*dy1 = atemp1;
	*dy2 = atemp2;
	*dy3 = atemp3;
	*dy4 = atemp4;
	*(buffer + 0) = btemp1;
	*(buffer + 1) = btemp2;
	*(buffer + 2) = btemp3;
	*(buffer + 3) = btemp4;
      } else {
	*(buffer + 0) = atemp1;
	*(buffer + 1) = atemp2;
	*(buffer + 2) = atemp3;
	*(buffer + 3) = atemp4;
      }

      ip = *piv;
      piv ++;

      i++;
      dx1 = a + i;
      dy1 = a + ip;
      dx2 = a + i  + lda * 1;
      dy2 = a + ip + lda * 1;
      dx3 = a + i  + lda * 2;
      dy3 = a + ip + lda * 2;
      dx4 = a + i  + lda * 3;
      dy4 = a + ip + lda * 3;

      buffer += 4;

    } while (i <= k2);

      a += 4 * lda;
  }

  if (n & 2) {
    piv = ipiv;

    i = k1;
    do {
      ip = *piv;
      piv ++;

      dx1 = a + i;
      dy1 = a + ip;
      dx2 = a + i  + lda;
      dy2 = a + ip + lda;

      atemp1 = *dx1;
      btemp1 = *dy1;
      atemp2 = *dx2;
      btemp2 = *dy2;

      if (ip != i) {
	*dy1 = atemp1;
	*dy2 = atemp2;
	*(buffer + 0) = btemp1;
	*(buffer + 1) = btemp2;
      } else {
	*(buffer + 0) = atemp1;
	*(buffer + 1) = atemp2;
      }

      buffer += 2;

      i++;
    } while (i <= k2);

    a += 2 * lda;
  }


  if (n & 1) {
    piv = ipiv;

    i = k1;
    do {
      ip = *piv;
      piv ++;

      dx1 = a + i;
      dy1 = a + ip;
      atemp1 = *dx1;
      btemp1 = *dy1;

      if (ip != i) {
	*dy1 = atemp1;
	*buffer = btemp1;
      } else {
	*buffer = atemp1;
      }

      buffer ++;

      i++;
    } while (i <= k2);

    a += lda;
  }

  return 0;
}

