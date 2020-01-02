/*******************************************************************************
Copyright (c) 2019, The OpenBLAS Project
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
*******************************************************************************/

#define ASSEMBLER
#include "common.h"

/*                   X0          X1          X2          s0         X3        x4       x5           x6  */
/*int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha0,FLOAT* ba,FLOAT* bb,FLOAT* C,BLASLONG ldc )*/

#define origM           x0
#define origN           x1
#define origK           x2
#define origPA          x3
#define origPB          x4
#define origPC          x5
#define LDC             x6
#define tmp             x7
#define tmpK            x8
#define I               x9
#define J               x10
#define pA              x11
#define pB              x12
#define pC              x13
#define alpha           x14
#define pCRow           x15
#define pCRow0          x16
#define pCRow1          x17
#define pCRow2          x18
#define pCRow3          x19
#define pCRow4          x20
#define pCRow5          x21
#define pCRow6          x22
#define pCRow7          x23

#define alphaV          z15.d

#define pMask8          p0
#define pMask4          p1
#define pMask2          p2
#define pMask1          p3

/*******************************************************************************
* Macro definitions
*******************************************************************************/

/** N8xM16 ********************************************************************/
.macro INIT_N8xM16
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
    dup     z24.d, #0
    dup     z25.d, #0
    dup     z26.d, #0
    dup     z27.d, #0
    dup     z28.d, #0
    dup     z29.d, #0
    dup     z30.d, #0
    dup     z31.d, #0
.endm

.macro MULTIPLY_N8xM16
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1d    z1.d, pMask8/z, [pA]
    add     pA, pA, #64
    /* TODO: Try using ld1d + dup */
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    ld1rqd  z10.d, pMask8/z, [pB, #32]
    ld1rqd  z11.d, pMask8/z, [pB, #48]
    add     pB, pB, #64

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z0.d, z10.d[0]
    fmla    z21.d, z0.d, z10.d[1]
    fmla    z22.d, z0.d, z11.d[0]
    fmla    z23.d, z0.d, z11.d[1]
    fmla    z24.d, z1.d, z8.d[0]
    fmla    z25.d, z1.d, z8.d[1]
    fmla    z26.d, z1.d, z9.d[0]
    fmla    z27.d, z1.d, z9.d[1]
    fmla    z28.d, z1.d, z10.d[0]
    fmla    z29.d, z1.d, z10.d[1]
    fmla    z30.d, z1.d, z11.d[0]
    fmla    z31.d, z1.d, z11.d[1]
.endm

.macro SAVE_N8xM16
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow4, pCRow3, LDC
    add     pCRow5, pCRow4, LDC
    add     pCRow6, pCRow5, LDC
    add     pCRow7, pCRow6, LDC
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]
    ld1d    z2.d, pMask8/z, [pCRow2]
    ld1d    z3.d, pMask8/z, [pCRow3]
    ld1d    z4.d, pMask8/z, [pCRow4]
    ld1d    z5.d, pMask8/z, [pCRow5]
    ld1d    z6.d, pMask8/z, [pCRow6]
    ld1d    z7.d, pMask8/z, [pCRow7]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV
    fmla    z2.d, pMask8/m, z18.d, alphaV
    fmla    z3.d, pMask8/m, z19.d, alphaV
    fmla    z4.d, pMask8/m, z20.d, alphaV
    fmla    z5.d, pMask8/m, z21.d, alphaV
    fmla    z6.d, pMask8/m, z22.d, alphaV
    fmla    z7.d, pMask8/m, z23.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]
    st1d    {z2.d}, pMask8, [pCRow2]
    st1d    {z3.d}, pMask8, [pCRow3]
    st1d    {z4.d}, pMask8, [pCRow4]
    st1d    {z5.d}, pMask8, [pCRow5]
    st1d    {z6.d}, pMask8, [pCRow6]
    st1d    {z7.d}, pMask8, [pCRow7]

    add     pCRow0, pCRow0, #64
    add     pCRow1, pCRow1, #64
    add     pCRow2, pCRow2, #64
    add     pCRow3, pCRow3, #64
    add     pCRow4, pCRow4, #64
    add     pCRow5, pCRow5, #64
    add     pCRow6, pCRow6, #64
    add     pCRow7, pCRow7, #64

    ld1d    z8.d, pMask8/z, [pCRow0]
    ld1d    z9.d, pMask8/z, [pCRow1]
    ld1d    z10.d, pMask8/z, [pCRow2]
    ld1d    z11.d, pMask8/z, [pCRow3]
    ld1d    z12.d, pMask8/z, [pCRow4]
    ld1d    z13.d, pMask8/z, [pCRow5]
    ld1d    z14.d, pMask8/z, [pCRow6]
	/* Since Alpha is using z15 */
    ld1d    z0.d, pMask8/z, [pCRow7]

    fmla    z8.d, pMask8/m, z24.d, alphaV
    fmla    z9.d, pMask8/m, z25.d, alphaV
    fmla    z10.d, pMask8/m, z26.d, alphaV
    fmla    z11.d, pMask8/m, z27.d, alphaV
    fmla    z12.d, pMask8/m, z28.d, alphaV
    fmla    z13.d, pMask8/m, z29.d, alphaV
    fmla    z14.d, pMask8/m, z30.d, alphaV
    fmla    z0.d, pMask8/m, z31.d, alphaV

    st1d    {z8.d}, pMask8, [pCRow0]
    st1d    {z9.d}, pMask8, [pCRow1]
    st1d    {z10.d}, pMask8, [pCRow2]
    st1d    {z11.d}, pMask8, [pCRow3]
    st1d    {z12.d}, pMask8, [pCRow4]
    st1d    {z13.d}, pMask8, [pCRow5]
    st1d    {z14.d}, pMask8, [pCRow6]
    st1d    {z0.d}, pMask8, [pCRow7]
.endm
/** N8xM8 *********************************************************************/
.macro INIT_N8xM8
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
.endm

.macro MULTIPLY_N8xM8
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    /* TODO: Try using ld1d + dup */
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    ld1rqd  z10.d, pMask8/z, [pB, #32]
    ld1rqd  z11.d, pMask8/z, [pB, #48]
    add     pB, pB, #64

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z0.d, z10.d[0]
    fmla    z21.d, z0.d, z10.d[1]
    fmla    z22.d, z0.d, z11.d[0]
    fmla    z23.d, z0.d, z11.d[1]
.endm

.macro SAVE_N8xM8
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow4, pCRow3, LDC
    add     pCRow5, pCRow4, LDC
    add     pCRow6, pCRow5, LDC
    add     pCRow7, pCRow6, LDC
    add     pCRow, pCRow, #64

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]
    ld1d    z2.d, pMask8/z, [pCRow2]
    ld1d    z3.d, pMask8/z, [pCRow3]
    ld1d    z4.d, pMask8/z, [pCRow4]
    ld1d    z5.d, pMask8/z, [pCRow5]
    ld1d    z6.d, pMask8/z, [pCRow6]
    ld1d    z7.d, pMask8/z, [pCRow7]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV
    fmla    z2.d, pMask8/m, z18.d, alphaV
    fmla    z3.d, pMask8/m, z19.d, alphaV
    fmla    z4.d, pMask8/m, z20.d, alphaV
    fmla    z5.d, pMask8/m, z21.d, alphaV
    fmla    z6.d, pMask8/m, z22.d, alphaV
    fmla    z7.d, pMask8/m, z23.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]
    st1d    {z2.d}, pMask8, [pCRow2]
    st1d    {z3.d}, pMask8, [pCRow3]
    st1d    {z4.d}, pMask8, [pCRow4]
    st1d    {z5.d}, pMask8, [pCRow5]
    st1d    {z6.d}, pMask8, [pCRow6]
    st1d    {z7.d}, pMask8, [pCRow7]
.endm
/** N8xM4 *********************************************************************/
.macro INIT_N8xM4
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
.endm

.macro MULTIPLY_N8xM4
    ld1d    z0.d, pMask4/z, [pA]
    add     pA, pA, #32
    /* TODO: Try using ld1d + dup */
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    ld1rqd  z10.d, pMask8/z, [pB, #32]
    ld1rqd  z11.d, pMask8/z, [pB, #48]
    add     pB, pB, #64

    /* TODO: Try to avoid multiplying all 128-bit vector lanes */
    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z0.d, z10.d[0]
    fmla    z21.d, z0.d, z10.d[1]
    fmla    z22.d, z0.d, z11.d[0]
    fmla    z23.d, z0.d, z11.d[1]
.endm

.macro SAVE_N8xM4
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow4, pCRow3, LDC
    add     pCRow5, pCRow4, LDC
    add     pCRow6, pCRow5, LDC
    add     pCRow7, pCRow6, LDC
    add     pCRow, pCRow, #32

    ld1d    z0.d, pMask4/z, [pCRow0]
    ld1d    z1.d, pMask4/z, [pCRow1]
    ld1d    z2.d, pMask4/z, [pCRow2]
    ld1d    z3.d, pMask4/z, [pCRow3]
    ld1d    z4.d, pMask4/z, [pCRow4]
    ld1d    z5.d, pMask4/z, [pCRow5]
    ld1d    z6.d, pMask4/z, [pCRow6]
    ld1d    z7.d, pMask4/z, [pCRow7]

    fmla    z0.d, pMask4/m, z16.d, alphaV
    fmla    z1.d, pMask4/m, z17.d, alphaV
    fmla    z2.d, pMask4/m, z18.d, alphaV
    fmla    z3.d, pMask4/m, z19.d, alphaV
    fmla    z4.d, pMask4/m, z20.d, alphaV
    fmla    z5.d, pMask4/m, z21.d, alphaV
    fmla    z6.d, pMask4/m, z22.d, alphaV
    fmla    z7.d, pMask4/m, z23.d, alphaV

    st1d    {z0.d}, pMask4, [pCRow0]
    st1d    {z1.d}, pMask4, [pCRow1]
    st1d    {z2.d}, pMask4, [pCRow2]
    st1d    {z3.d}, pMask4, [pCRow3]
    st1d    {z4.d}, pMask4, [pCRow4]
    st1d    {z5.d}, pMask4, [pCRow5]
    st1d    {z6.d}, pMask4, [pCRow6]
    st1d    {z7.d}, pMask4, [pCRow7]
.endm
/** N8xM2 *********************************************************************/
.macro INIT_N8xM2
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
.endm

.macro MULTIPLY_N8xM2
    ld1d    z0.d, pMask2/z, [pA]
    add     pA, pA, #16
    /* TODO: Try using ld1d + dup */
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    ld1rqd  z10.d, pMask8/z, [pB, #32]
    ld1rqd  z11.d, pMask8/z, [pB, #48]
    add     pB, pB, #64

    /* TODO: Try to avoid multiplying all 128-bit vector lanes */
    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z0.d, z10.d[0]
    fmla    z21.d, z0.d, z10.d[1]
    fmla    z22.d, z0.d, z11.d[0]
    fmla    z23.d, z0.d, z11.d[1]
.endm

.macro SAVE_N8xM2
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow4, pCRow3, LDC
    add     pCRow5, pCRow4, LDC
    add     pCRow6, pCRow5, LDC
    add     pCRow7, pCRow6, LDC
    add     pCRow, pCRow, #16

    ld1d    z0.d, pMask2/z, [pCRow0]
    ld1d    z1.d, pMask2/z, [pCRow1]
    ld1d    z2.d, pMask2/z, [pCRow2]
    ld1d    z3.d, pMask2/z, [pCRow3]
    ld1d    z4.d, pMask2/z, [pCRow4]
    ld1d    z5.d, pMask2/z, [pCRow5]
    ld1d    z6.d, pMask2/z, [pCRow6]
    ld1d    z7.d, pMask2/z, [pCRow7]

    fmla    z0.d, pMask2/m, z16.d, alphaV
    fmla    z1.d, pMask2/m, z17.d, alphaV
    fmla    z2.d, pMask2/m, z18.d, alphaV
    fmla    z3.d, pMask2/m, z19.d, alphaV
    fmla    z4.d, pMask2/m, z20.d, alphaV
    fmla    z5.d, pMask2/m, z21.d, alphaV
    fmla    z6.d, pMask2/m, z22.d, alphaV
    fmla    z7.d, pMask2/m, z23.d, alphaV

    st1d    {z0.d}, pMask2, [pCRow0]
    st1d    {z1.d}, pMask2, [pCRow1]
    st1d    {z2.d}, pMask2, [pCRow2]
    st1d    {z3.d}, pMask2, [pCRow3]
    st1d    {z4.d}, pMask2, [pCRow4]
    st1d    {z5.d}, pMask2, [pCRow5]
    st1d    {z6.d}, pMask2, [pCRow6]
    st1d    {z7.d}, pMask2, [pCRow7]
.endm
/** N8xM1 *********************************************************************/
.macro INIT_N8xM1
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
.endm

.macro MULTIPLY_N8xM1
    ld1d    z0.d, pMask1/z, [pA]
    add     pA, pA, #8
    /* TODO: Try using ld1d + dup */
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    ld1rqd  z10.d, pMask8/z, [pB, #32]
    ld1rqd  z11.d, pMask8/z, [pB, #48]
    add     pB, pB, #64

    /* TODO: Try to avoid multiplying all 128-bit vector lanes */
    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z0.d, z10.d[0]
    fmla    z21.d, z0.d, z10.d[1]
    fmla    z22.d, z0.d, z11.d[0]
    fmla    z23.d, z0.d, z11.d[1]
.endm

.macro SAVE_N8xM1
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow4, pCRow3, LDC
    add     pCRow5, pCRow4, LDC
    add     pCRow6, pCRow5, LDC
    add     pCRow7, pCRow6, LDC
    add     pCRow, pCRow, #8

    ld1d    z0.d, pMask1/z, [pCRow0]
    ld1d    z1.d, pMask1/z, [pCRow1]
    ld1d    z2.d, pMask1/z, [pCRow2]
    ld1d    z3.d, pMask1/z, [pCRow3]
    ld1d    z4.d, pMask1/z, [pCRow4]
    ld1d    z5.d, pMask1/z, [pCRow5]
    ld1d    z6.d, pMask1/z, [pCRow6]
    ld1d    z7.d, pMask1/z, [pCRow7]

    fmla    z0.d, pMask1/m, z16.d, alphaV
    fmla    z1.d, pMask1/m, z17.d, alphaV
    fmla    z2.d, pMask1/m, z18.d, alphaV
    fmla    z3.d, pMask1/m, z19.d, alphaV
    fmla    z4.d, pMask1/m, z20.d, alphaV
    fmla    z5.d, pMask1/m, z21.d, alphaV
    fmla    z6.d, pMask1/m, z22.d, alphaV
    fmla    z7.d, pMask1/m, z23.d, alphaV

    st1d    {z0.d}, pMask1, [pCRow0]
    st1d    {z1.d}, pMask1, [pCRow1]
    st1d    {z2.d}, pMask1, [pCRow2]
    st1d    {z3.d}, pMask1, [pCRow3]
    st1d    {z4.d}, pMask1, [pCRow4]
    st1d    {z5.d}, pMask1, [pCRow5]
    st1d    {z6.d}, pMask1, [pCRow6]
    st1d    {z7.d}, pMask1, [pCRow7]
.endm
/** N4xM16 ********************************************************************/
.macro INIT_N4xM16
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
    dup     z20.d, #0
    dup     z21.d, #0
    dup     z22.d, #0
    dup     z23.d, #0
.endm

.macro MULTIPLY_N4xM16
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1d    z1.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
    fmla    z20.d, z1.d, z8.d[0]
    fmla    z21.d, z1.d, z8.d[1]
    fmla    z22.d, z1.d, z9.d[0]
    fmla    z23.d, z1.d, z9.d[1]
.endm

.macro SAVE_N4xM16
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]
    ld1d    z2.d, pMask8/z, [pCRow2]
    ld1d    z3.d, pMask8/z, [pCRow3]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV
    fmla    z2.d, pMask8/m, z18.d, alphaV
    fmla    z3.d, pMask8/m, z19.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]
    st1d    {z2.d}, pMask8, [pCRow2]
    st1d    {z3.d}, pMask8, [pCRow3]

    add     pCRow0, pCRow0, #64
    add     pCRow1, pCRow1, #64
    add     pCRow2, pCRow2, #64
    add     pCRow3, pCRow3, #64

    ld1d    z4.d, pMask8/z, [pCRow0]
    ld1d    z5.d, pMask8/z, [pCRow1]
    ld1d    z6.d, pMask8/z, [pCRow2]
    ld1d    z7.d, pMask8/z, [pCRow3]

    fmla    z4.d, pMask8/m, z20.d, alphaV
    fmla    z5.d, pMask8/m, z21.d, alphaV
    fmla    z6.d, pMask8/m, z22.d, alphaV
    fmla    z7.d, pMask8/m, z23.d, alphaV

    st1d    {z4.d}, pMask8, [pCRow0]
    st1d    {z5.d}, pMask8, [pCRow1]
    st1d    {z6.d}, pMask8, [pCRow2]
    st1d    {z7.d}, pMask8, [pCRow3]
.endm
/** N4xM8 *********************************************************************/
.macro INIT_N4xM8
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N4xM8
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
.endm

.macro SAVE_N4xM8
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #64

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]
    ld1d    z2.d, pMask8/z, [pCRow2]
    ld1d    z3.d, pMask8/z, [pCRow3]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV
    fmla    z2.d, pMask8/m, z18.d, alphaV
    fmla    z3.d, pMask8/m, z19.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]
    st1d    {z2.d}, pMask8, [pCRow2]
    st1d    {z3.d}, pMask8, [pCRow3]
.endm
/** N4xM4 *********************************************************************/
.macro INIT_N4xM4
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N4xM4
    ld1d    z0.d, pMask4/z, [pA]
    add     pA, pA, #32
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
.endm

.macro SAVE_N4xM4
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #32

    ld1d    z0.d, pMask4/z, [pCRow0]
    ld1d    z1.d, pMask4/z, [pCRow1]
    ld1d    z2.d, pMask4/z, [pCRow2]
    ld1d    z3.d, pMask4/z, [pCRow3]

    fmla    z0.d, pMask4/m, z16.d, alphaV
    fmla    z1.d, pMask4/m, z17.d, alphaV
    fmla    z2.d, pMask4/m, z18.d, alphaV
    fmla    z3.d, pMask4/m, z19.d, alphaV

    st1d    {z0.d}, pMask4, [pCRow0]
    st1d    {z1.d}, pMask4, [pCRow1]
    st1d    {z2.d}, pMask4, [pCRow2]
    st1d    {z3.d}, pMask4, [pCRow3]
.endm
/** N4xM2 *********************************************************************/
.macro INIT_N4xM2
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N4xM2
    ld1d    z0.d, pMask2/z, [pA]
    add     pA, pA, #16
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
.endm

.macro SAVE_N4xM2
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #16

    ld1d    z0.d, pMask2/z, [pCRow0]
    ld1d    z1.d, pMask2/z, [pCRow1]
    ld1d    z2.d, pMask2/z, [pCRow2]
    ld1d    z3.d, pMask2/z, [pCRow3]

    fmla    z0.d, pMask2/m, z16.d, alphaV
    fmla    z1.d, pMask2/m, z17.d, alphaV
    fmla    z2.d, pMask2/m, z18.d, alphaV
    fmla    z3.d, pMask2/m, z19.d, alphaV

    st1d    {z0.d}, pMask2, [pCRow0]
    st1d    {z1.d}, pMask2, [pCRow1]
    st1d    {z2.d}, pMask2, [pCRow2]
    st1d    {z3.d}, pMask2, [pCRow3]
.endm
/** N4xM1 *********************************************************************/
.macro INIT_N4xM1
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N4xM1
    ld1d    z0.d, pMask1/z, [pA]
    add     pA, pA, #8
    ld1rqd  z8.d, pMask8/z, [pB]
    ld1rqd  z9.d, pMask8/z, [pB, #16]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z9.d[0]
    fmla    z19.d, z0.d, z9.d[1]
.endm

.macro SAVE_N4xM1
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #8

    ld1d    z0.d, pMask1/z, [pCRow0]
    ld1d    z1.d, pMask1/z, [pCRow1]
    ld1d    z2.d, pMask1/z, [pCRow2]
    ld1d    z3.d, pMask1/z, [pCRow3]

    fmla    z0.d, pMask1/m, z16.d, alphaV
    fmla    z1.d, pMask1/m, z17.d, alphaV
    fmla    z2.d, pMask1/m, z18.d, alphaV
    fmla    z3.d, pMask1/m, z19.d, alphaV

    st1d    {z0.d}, pMask1, [pCRow0]
    st1d    {z1.d}, pMask1, [pCRow1]
    st1d    {z2.d}, pMask1, [pCRow2]
    st1d    {z3.d}, pMask1, [pCRow3]
.endm
/** N2xM16 *********************************************************************/
.macro INIT_N2xM16
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N2xM16
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1d    z1.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rqd  z8.d, pMask8/z, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z1.d, z8.d[0]
    fmla    z19.d, z1.d, z8.d[1]
.endm

.macro SAVE_N2xM16
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]

    add     pCRow0, pCRow0, #64
    add     pCRow1, pCRow1, #64

    ld1d    z2.d, pMask8/z, [pCRow0]
    ld1d    z3.d, pMask8/z, [pCRow1]

    fmla    z2.d, pMask8/m, z18.d, alphaV
    fmla    z3.d, pMask8/m, z19.d, alphaV

    st1d    {z2.d}, pMask8, [pCRow0]
    st1d    {z3.d}, pMask8, [pCRow1]
.endm
/** N2xM8 *********************************************************************/
.macro INIT_N2xM8
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N2xM8
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rqd  z8.d, pMask8/z, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
.endm

.macro SAVE_N2xM8
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #64

    ld1d    z0.d, pMask8/z, [pCRow0]
    ld1d    z1.d, pMask8/z, [pCRow1]

    fmla    z0.d, pMask8/m, z16.d, alphaV
    fmla    z1.d, pMask8/m, z17.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
    st1d    {z1.d}, pMask8, [pCRow1]
.endm
/** N2xM4 *********************************************************************/
.macro INIT_N2xM4
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N2xM4
    ld1d    z0.d, pMask4/z, [pA]
    add     pA, pA, #32
    ld1rqd  z8.d, pMask8/z, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
.endm

.macro SAVE_N2xM4
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #32

    ld1d    z0.d, pMask4/z, [pCRow0]
    ld1d    z1.d, pMask4/z, [pCRow1]

    fmla    z0.d, pMask4/m, z16.d, alphaV
    fmla    z1.d, pMask4/m, z17.d, alphaV

    st1d    {z0.d}, pMask4, [pCRow0]
    st1d    {z1.d}, pMask4, [pCRow1]
.endm
/** N2xM2 *********************************************************************/
.macro INIT_N2xM2
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N2xM2
    ld1d    z0.d, pMask2/z, [pA]
    add     pA, pA, #16
    ld1rqd  z8.d, pMask8/z, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
.endm

.macro SAVE_N2xM2
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #16

    ld1d    z0.d, pMask2/z, [pCRow0]
    ld1d    z1.d, pMask2/z, [pCRow1]

    fmla    z0.d, pMask2/m, z16.d, alphaV
    fmla    z1.d, pMask2/m, z17.d, alphaV

    st1d    {z0.d}, pMask2, [pCRow0]
    st1d    {z1.d}, pMask2, [pCRow1]
.endm
/** N2xM1 *********************************************************************/
.macro INIT_N2xM1
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N2xM1
    ld1d    z0.d, pMask1/z, [pA]
    add     pA, pA, #8
    ld1rqd  z8.d, pMask8/z, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
.endm

.macro SAVE_N2xM1
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #8

    ld1d    z0.d, pMask1/z, [pCRow0]
    ld1d    z1.d, pMask1/z, [pCRow1]

    fmla    z0.d, pMask1/m, z16.d, alphaV
    fmla    z1.d, pMask1/m, z17.d, alphaV

    st1d    {z0.d}, pMask1, [pCRow0]
    st1d    {z1.d}, pMask1, [pCRow1]
.endm
/** N1xM16 ********************************************************************/
.macro INIT_N1xM16
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N1xM16
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1d    z1.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rd   {z8.d}, pMask8/z, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z1.d, z8.d[0]
.endm

.macro SAVE_N1xM16
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8/z, [pCRow0]
    fmla    z0.d, pMask8/m, z16.d, alphaV
    st1d    {z0.d}, pMask8, [pCRow0]

    add     pCRow0, pCRow0, #64

    ld1d    z1.d, pMask8/z, [pCRow0]
    fmla    z1.d, pMask8/m, z17.d, alphaV
    st1d    {z1.d}, pMask8, [pCRow0]
.endm
/** N1xM8 *********************************************************************/
.macro INIT_N1xM8
    dup     z16.d, #0
.endm

.macro MULTIPLY_N1xM8
    ld1d    z0.d, pMask8/z, [pA]
    add     pA, pA, #64
    ld1rd   {z8.d}, pMask8/z, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
.endm

.macro SAVE_N1xM8
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow, pCRow, #64

    ld1d    z0.d, pMask8/z, [pCRow0]

    fmla    z0.d, pMask8/m, z16.d, alphaV

    st1d    {z0.d}, pMask8, [pCRow0]
.endm
/** N1xM4 *********************************************************************/
.macro INIT_N1xM4
    dup     z16.d, #0
.endm

.macro MULTIPLY_N1xM4
    ld1d    z0.d, pMask4/z, [pA]
    add     pA, pA, #32
    ld1rd   {z8.d}, pMask8/z, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
.endm

.macro SAVE_N1xM4
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow, pCRow, #32

    ld1d    z0.d, pMask4/z, [pCRow0]

    fmla    z0.d, pMask4/m, z16.d, alphaV

    st1d    {z0.d}, pMask4, [pCRow0]
.endm
/** N1xM2 *********************************************************************/
.macro INIT_N1xM2
    dup     z16.d, #0
.endm

.macro MULTIPLY_N1xM2
    ld1d    z0.d, pMask2/z, [pA]
    add     pA, pA, #16
    ld1rd   {z8.d}, pMask8/z, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
.endm

.macro SAVE_N1xM2
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow, pCRow, #16

    ld1d    z0.d, pMask2/z, [pCRow0]

    fmla    z0.d, pMask2/m, z16.d, alphaV

    st1d    {z0.d}, pMask2, [pCRow0]
.endm
/** N1xM1 *********************************************************************/
.macro INIT_N1xM1
    dup     z16.d, #0
.endm

.macro MULTIPLY_N1xM1
    ld1d    z0.d, pMask1/z, [pA]
    add     pA, pA, #8
    ld1rd   {z8.d}, pMask8/z, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
.endm

.macro SAVE_N1xM1
    dup     alphaV, alpha
    mov     pCRow0, pCRow
    add     pCRow, pCRow, #32

    ld1d    z0.d, pMask1/z, [pCRow0]

    fmla    z0.d, pMask1/m, z16.d, alphaV

    st1d    {z0.d}, pMask1, [pCRow0]
.endm
/*******************************************************************************
* End of macro definitions
*******************************************************************************/


/*******************************************************************************
* DGEMM Function Start
*******************************************************************************/
    PROLOGUE

    .align  5
    cbz     origM, .END
    cbz     origK, .END
    cbz     origN, .END

.CHECK_VECTOR_LENGTH:
    rdvl    tmp, #1
    cmp     tmp, #64
    bne     .END                                // Evaluate only for VL = 512b

.SAVE_REGS:
    add     sp, sp, #-(11 * 16)                 // SAVE REGISTERS
    stp     d8, d9, [sp, #(0 * 16)]
    stp     d10, d11, [sp, #(1 * 16)]
    stp     d12, d13, [sp, #(2 * 16)]
    stp     d14, d15, [sp, #(3 * 16)]
    stp     d16, d17, [sp, #(4 * 16)]
    stp     x18, x19, [sp, #(5 * 16)]
    stp     x20, x21, [sp, #(6 * 16)]
    stp     x22, x23, [sp, #(7 * 16)]
    stp     x24, x25, [sp, #(8 * 16)]
    stp     x26, x27, [sp, #(9 * 16)]
    str     x28, [sp, #(10 * 16)]

.BEGIN:
    ptrue   p0.d, all                           // 8 lane mask
    mov     tmp, #4
    whilelt p1.d, xzr, tmp                      // 4 lane mask
    mov     tmp, #2
    whilelt p2.d, xzr, tmp                      // 2 lane mask
    mov     tmp, #1
    whilelt p3.d, xzr, tmp                      // 1 lane mask

    lsl     LDC, LDC, #3                        // ldc = ldc * 8
    fmov    alpha, d0
    mov     pC, origPC

/* N8 *************************************************************************/
    mov     J, origN
    asr     J, J, #3                            // N = N / 8
    cmp     J, #0
    ble     .N4_BEGIN

.N8_BEGIN:
    mov     pCRow, pC
    add     pC, pC, LDC, LSL #3                 // pC = pC + LDC * 8
    mov     pA, origPA

/* N8xM16 *********************************************************************/
.N8xM16_BEGIN:
    mov     I, origM
    asr     I, I, #4                            // M / 16
    cmp     I, #0
    ble     .N8xM8_BEGIN

.N8xM16_INIT:
    INIT_N8xM16
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N8xM16_LOOP:
    MULTIPLY_N8xM16
    subs    tmpK, tmpK, #1
    bne     .N8xM16_LOOP

.N8xM16_SAVE:
    SAVE_N8xM16
    subs    I, I, #1                            // CHECK WHETHER MORE 16x CHUNKS
                                                // OF A IS REMAINING
    bne     .N8xM16_INIT

/* N8xM8 **********************************************************************/
.N8xM8_BEGIN:
    tst     origM, #8
    ble     .N8xM4_BEGIN

.N8xM8_INIT:
    INIT_N8xM8
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N8xM8_LOOP:
    MULTIPLY_N8xM8
    subs    tmpK, tmpK, #1
    bne     .N8xM8_LOOP

.N8xM8_SAVE:
    SAVE_N8xM8

/* N8xM4 **********************************************************************/
.N8xM4_BEGIN:
    tst     origM, #4
    ble     .N8xM2_BEGIN

.N8xM4_INIT:
    INIT_N8xM4
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N8xM4_LOOP:
    MULTIPLY_N8xM4
    subs    tmpK, tmpK, #1
    bgt     .N8xM4_LOOP

.N8xM4_100:
    SAVE_N8xM4

/* N8xM2 **********************************************************************/
.N8xM2_BEGIN:
    tst     origM, #2
    ble     .N8xM1_BEGIN

.N8xM2_INIT:
    INIT_N8xM2
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N8xM2_LOOP:
    MULTIPLY_N8xM2
    subs    tmpK, tmpK, #1
    bgt     .N8xM2_LOOP

.N8xM2_100:
    SAVE_N8xM2

/* N8xM1 **********************************************************************/
.N8xM1_BEGIN:
    tst     origM, #1
    ble     .N8_END

.N8xM1_INIT:
    INIT_N8xM1
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N8xM1_LOOP:
    MULTIPLY_N8xM1
    subs    tmpK, tmpK, #1
    bgt     .N8xM1_LOOP

.N8xM1_SAVE:
    SAVE_N8xM1

/* N8 END *********************************************************************/
.N8_END:
    lsl     tmpK, origK, #6                     // tmpk = origK * 8 * sizeof(double)
    add     origPB, origPB, tmpK                // B = B + tmpK
    subs    J, J, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF MATRIX B REMAINS
    bgt     .N8_BEGIN

/* N4 *************************************************************************/
.N4_BEGIN:
    tst     origN, #4
    ble     .N2_BEGIN                           // N HAS NO 4x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, LSL #2                 // pC = pC + LDC * 4
    mov     pA, origPA

/* N4xM16 *********************************************************************/
.N4xM16_BEGIN:
    mov     I, origM
    asr     I, I, #4                            // M / 16
    cmp     I, #0
    ble     .N4xM8_BEGIN

.N4xM16_INIT:
    INIT_N4xM16
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N4xM16_LOOP:
    MULTIPLY_N4xM16
    subs    tmpK, tmpK, #1
    bne     .N4xM16_LOOP

.N4xM16_SAVE:
    SAVE_N4xM16
    subs    I, I, #1                            // CHECK WHETHER MORE 16x CHUNKS
                                                // OF A IS REMAINING
    bne     .N4xM16_INIT

/* N4xM8 **********************************************************************/
.N4xM8_BEGIN:
    tst     origM, #8
    ble     .N4xM4_BEGIN

.N4xM8_INIT:
    INIT_N4xM8
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N4xM8_LOOP:
    MULTIPLY_N4xM8
    subs    tmpK, tmpK, #1
    bne     .N4xM8_LOOP

.N4xM8_SAVE:
    SAVE_N4xM8

/* N4xM4 **********************************************************************/
.N4xM4_BEGIN:
    tst     origM, #4
    ble     .N4xM2_BEGIN

.N4xM4_INIT:
    INIT_N4xM4
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N4xM4_LOOP:
    MULTIPLY_N4xM4
    subs    tmpK, tmpK, #1
    bgt     .N4xM4_LOOP

.N4xM4_100:
    SAVE_N4xM4

/* N4xM2 **********************************************************************/
.N4xM2_BEGIN:
    tst     origM, #2
    ble     .N4xM1_BEGIN

.N4xM2_INIT:
    INIT_N4xM2
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N4xM2_LOOP:
    MULTIPLY_N4xM2
    subs    tmpK, tmpK, #1
    bgt     .N4xM2_LOOP

.N4xM2_100:
    SAVE_N4xM2

/* N4xM1 **********************************************************************/
.N4xM1_BEGIN:
    tst     origM, #1
    ble     .N4_END

.N4xM1_INIT:
    INIT_N4xM1
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N4xM1_LOOP:
    MULTIPLY_N4xM1
    subs    tmpK, tmpK, #1
    bgt     .N4xM1_LOOP

.N4xM1_SAVE:
    SAVE_N4xM1

/* N4 END *********************************************************************/
.N4_END:
    lsl     tmpK, origK, #5                     // tmpk = origK * 4 * sizeof(double)
    add     origPB, origPB, tmpK                // B = B + tmpK

/* N2 *************************************************************************/
.N2_BEGIN:
    tst     origN, #2
    ble     .N1_BEGIN                           // N HAS NO 2x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, LSL #1                 // pC = pC + LDC * 2
    mov     pA, origPA

/* N2xM16 *********************************************************************/
.N2xM16_BEGIN:
    mov     I, origM
    asr     I, I, #4                            // M / 16
    cmp     I, #0
    ble     .N2xM8_BEGIN

.N2xM16_INIT:
    INIT_N2xM16
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N2xM16_LOOP:
    MULTIPLY_N2xM16
    subs    tmpK, tmpK, #1
    bgt     .N2xM16_LOOP

.N2xM16_SAVE:
    SAVE_N2xM16
    subs    I, I, #1                            // CHECK WHETHER MORE 16x CHUNKS
                                                // OF A IS REMAINING
    bgt     .N2xM16_INIT

/* N2xM8 **********************************************************************/
.N2xM8_BEGIN:
    tst     origM, #8
    ble     .N2xM4_BEGIN

.N2xM8_INIT:
    INIT_N2xM8
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N2xM8_LOOP:
    MULTIPLY_N2xM8
    subs    tmpK, tmpK, #1
    bgt     .N2xM8_LOOP

.N2xM8_SAVE:
    SAVE_N2xM8

/* N2xM4 **********************************************************************/
.N2xM4_BEGIN:
    tst     origM, #4
    ble     .N2xM2_BEGIN

.N2xM4_INIT:
    INIT_N2xM4
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N2xM4_LOOP:
    MULTIPLY_N2xM4
    subs    tmpK, tmpK, #1
    bgt     .N2xM4_LOOP

.N2xM4_SAVE:
    SAVE_N2xM4

/* N2xM2 **********************************************************************/
.N2xM2_BEGIN:
    tst     origM, #2
    ble     .N2xM1_BEGIN

.N2xM2_INIT:
    INIT_N2xM2
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N2xM2_LOOP:
    MULTIPLY_N2xM2
    subs    tmpK, tmpK, #1
    bgt     .N2xM2_LOOP

.N2xM2_SAVE:
    SAVE_N2xM2

/* N2xM1 **********************************************************************/
.N2xM1_BEGIN:
    tst     origM, #1
    ble     .N2_END

.N2xM1_INIT:
    INIT_N2xM1
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N2xM1_LOOP:
    MULTIPLY_N2xM1
    subs    tmpK, tmpK, #1
    bgt     .N2xM1_LOOP

.N2xM1_SAVE:
    SAVE_N2xM1

/* N2 END *********************************************************************/
.N2_END:
    add     origPB, origPB, origK, lsl #4       // B = B + origK * 2 * sizeof(double)

/* N1 *************************************************************************/
.N1_BEGIN:
    tst     origN, #1
    ble     .RESTORE_REGS                       // N WAS MULTIPLE OF 2
    mov     pCRow, pC
    add     pC, pC, LDC                         // pC = pC + LDC
    mov     pA, origPA

/* N1xM16 *********************************************************************/
.N1xM16_BEGIN:
    mov     I, origM
    asr     I, I, #4                            // M / 16
    cmp     I, #0
    ble     .N1xM8_BEGIN

    .align  5
.N1xM16_INIT:
    INIT_N1xM16
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N1xM16_LOOP:
    MULTIPLY_N1xM16
    subs    tmpK, tmpK, #1
    bgt     .N1xM16_LOOP

.N1xM16_SAVE:
    SAVE_N1xM16
    subs    I, I, #1                            // CHECK WHETHER MORE 16x CHUNKS
                                                // OF A IS REMAINING
    bgt     .N1xM16_INIT

/* N1xM8 **********************************************************************/
.N1xM8_BEGIN:
    tst     origM, #8
    ble     .N1xM4_BEGIN

    .align  5
.N1xM8_INIT:
    INIT_N1xM8
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N1xM8_LOOP:
    MULTIPLY_N1xM8
    subs    tmpK, tmpK, #1
    bgt     .N1xM8_LOOP

.N1xM8_SAVE:
    SAVE_N1xM8

/* N1xM4 **********************************************************************/
.N1xM4_BEGIN:
    tst     origM, #4
    ble     .N1xM2_BEGIN

.N1xM4_INIT:
    INIT_N1xM4
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N1xM4_LOOP:
    MULTIPLY_N1xM4
    subs    tmpK, tmpK, #1
    bgt     .N1xM4_LOOP

.N1xM4_SAVE:
    SAVE_N1xM4

/* N1xM2 **********************************************************************/
.N1xM2_BEGIN:
    tst     origM, #2
    ble     .N1xM1_BEGIN

.N1xM2_INIT:
    INIT_N1xM2
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N1xM2_LOOP:
    MULTIPLY_N1xM2
    subs    tmpK, tmpK, #1
    bgt     .N1xM2_LOOP

.N1xM2_SAVE:
    SAVE_N1xM2

/* N1xM1 **********************************************************************/
.N1xM1_BEGIN:
    tst     origM, #1
    ble     .N1_END

.N1xM1_INIT:
    INIT_N1xM1
    mov     pB, origPB
    mov     tmpK, origK

    .align  5
.N1xM1_LOOP:
    MULTIPLY_N1xM1
    subs    tmpK, tmpK, #1
    bgt     .N1xM1_LOOP

.N1xM1_SAVE:
    SAVE_N1xM1

.N1_END:

/* THE END ********************************************************************/
.RESTORE_REGS:
    ldp     d8, d9, [sp, #(0 * 16)]             // RESTORE REGISTERS
    ldp     d10, d11, [sp, #(1 * 16)]
    ldp     d12, d13, [sp, #(2 * 16)]
    ldp     d14, d15, [sp, #(3 * 16)]
    ldp     d16, d17, [sp, #(4 * 16)]
    ldp     x18, x19, [sp, #(5 * 16)]
    ldp     x20, x21, [sp, #(6 * 16)]
    ldp     x22, x23, [sp, #(7 * 16)]
    ldp     x24, x25, [sp, #(8 * 16)]
    ldp     x26, x27, [sp, #(9 * 16)]
    ldr     x28, [sp, #(10 * 16)]
    add     sp, sp, #(11*16)
.END:
    mov     x0, #0                              // SET RETURN VALUE
    ret

    EPILOGUE


//#define origM           x24
//    mov     origM, x0
//    ldr     x0, .LPRINT_N8xM8
//    stp     x29, x30, [sp, #-16]!
    //bl      printf
//    ldp     x29, x30, [sp], #16
//.data
//.LPRINT_N8xM8:
//    .asciz  "N8xM8\n"

