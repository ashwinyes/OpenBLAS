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

/*                   X0          X1          X2          s0         X3        x4       x5           x6 */
/*int CNAME(BLASLONG bm,BLASLONG bn,BLASLONG bk,FLOAT alpha0,FLOAT* ba,FLOAT* bb,FLOAT* C,BLASLONG ldc )*/

#define origM           x0
#define origN           x1
#define origK           x2
#define origPA          x3
#define origPB          x4
#define origPC          x5
#define LDC             x6
#define temp            x7
#define tempK           x8
#define I               x9
#define J               x10
#define pA              x11
#define pB              x12
#define pC              x13
#define alpha           x14
#define pCRow0          x15
#define pCRow1          x16
#define pCRow2          x17
#define pCRow3          x18
#define pCRow4          x19
#define pCRow5          x20
#define pCRow6          x21
#define pCRow7          x22

#define alphaV          z15.d

#define pMask8          p0/z
#define pMask4          p1/z
#define pMask2          p2/z
#define pMask1          p3/z

/*******************************************************************************
* Macro definitions
*******************************************************************************/
/** 8x8 ***********************************************************************/
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
    ld1d    z0.d, pMask8, [pA]
    add     pA, pA, #64
    ld1d    z8.d, pMask8, [pB]
    add     pB, pB, #64

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z8.d[2]
    fmla    z19.d, z0.d, z8.d[3]
    fmla    z20.d, z0.d, z8.d[4]
    fmla    z21.d, z0.d, z8.d[5]
    fmla    z22.d, z0.d, z8.d[6]
    fmla    z23.d, z0.d, z8.d[7]
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
    add     pCRow, pCRow, #512

    ld1d    z0.d, pMask8, [pCRow0]
    ld1d    z1.d, pMask8, [pCRow1]
    ld1d    z2.d, pMask8, [pCRow2]
    ld1d    z3.d, pMask8, [pCRow3]
    ld1d    z4.d, pMask8, [pCRow4]
    ld1d    z5.d, pMask8, [pCRow5]
    ld1d    z6.d, pMask8, [pCRow6]
    ld1d    z7.d, pMask8, [pCRow7]

    fmla    z0.d, z16.d, alphaV
    fmla    z1.d, z17.d, alphaV
    fmla    z2.d, z18.d, alphaV
    fmla    z3.d, z19.d, alphaV
    fmla    z4.d, z20.d, alphaV
    fmla    z5.d, z21.d, alphaV
    fmla    z6.d, z22.d, alphaV
    fmla    z7.d, z23.d, alphaV

    st1d    z0.d, pMask8, [pCRow0]
    st1d    z1.d, pMask8, [pCRow1]
    st1d    z2.d, pMask8, [pCRow2]
    st1d    z3.d, pMask8, [pCRow3]
    st1d    z4.d, pMask8, [pCRow4]
    st1d    z5.d, pMask8, [pCRow5]
    st1d    z6.d, pMask8, [pCRow6]
    st1d    z7.d, pMask8, [pCRow7]
.endm
/** 4x8 ***********************************************************************/
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
    ld1d    z0.d, pMask4, [pA]
    add     pA, pA, #32
    ld1d    z8.d, pMask8, [pB]
    add     pB, pB, #64

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z8.d[2]
    fmla    z19.d, z0.d, z8.d[3]
    fmla    z20.d, z0.d, z8.d[4]
    fmla    z21.d, z0.d, z8.d[5]
    fmla    z22.d, z0.d, z8.d[6]
    fmla    z23.d, z0.d, z8.d[7]
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
    add     pCRow, pCRow, #512

    ld1d    z0.d, pMask4, [pCRow0]
    ld1d    z1.d, pMask4, [pCRow1]
    ld1d    z2.d, pMask4, [pCRow2]
    ld1d    z3.d, pMask4, [pCRow3]
    ld1d    z4.d, pMask4, [pCRow4]
    ld1d    z5.d, pMask4, [pCRow5]
    ld1d    z6.d, pMask4, [pCRow6]
    ld1d    z7.d, pMask4, [pCRow7]

    fmla    z0.d, z16.d, alphaV
    fmla    z1.d, z17.d, alphaV
    fmla    z2.d, z18.d, alphaV
    fmla    z3.d, z19.d, alphaV
    fmla    z4.d, z20.d, alphaV
    fmla    z5.d, z21.d, alphaV
    fmla    z6.d, z22.d, alphaV
    fmla    z7.d, z23.d, alphaV

    st1d    z0.d, pMask4, [pCRow0]
    st1d    z1.d, pMask4, [pCRow1]
    st1d    z2.d, pMask4, [pCRow2]
    st1d    z3.d, pMask4, [pCRow3]
    st1d    z4.d, pMask4, [pCRow4]
    st1d    z5.d, pMask4, [pCRow5]
    st1d    z6.d, pMask4, [pCRow6]
    st1d    z7.d, pMask4, [pCRow7]
.endm
/** 2x8 ***********************************************************************/
.macro INIT_N8xM2
.endm

.macro MULTIPLY_N8xM2
.endm

.macro SAVE_N8xM2
.endm
/** 1x8 ***********************************************************************/
.macro INIT_N8xM1
.endm

.macro MULTIPLY_N8xM1
.endm

.macro SAVE_N8xM1
.endm
/** 8x4 ***********************************************************************/
.macro INIT_N4xM8
    dup     z16.d, #0
    dup     z17.d, #0
    dup     z18.d, #0
    dup     z19.d, #0
.endm

.macro MULTIPLY_N4xM8
    ld1d    z0.d, pMask8, [pA]
    add     pA, pA, #64
    ld1d    z8.d, pMask4, [pB]
    add     pB, pB, #32

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
    fmla    z18.d, z0.d, z8.d[2]
    fmla    z19.d, z0.d, z8.d[3]
.endm

.macro SAVE_N4xM8
    dup     alphaV, alpha
	mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow2, pCRow1, LDC
    add     pCRow3, pCRow2, LDC
    add     pCRow, pCRow, #256

    ld1d    z0.d, pMask8, [pCRow0]
    ld1d    z1.d, pMask8, [pCRow1]
    ld1d    z2.d, pMask8, [pCRow2]
    ld1d    z3.d, pMask8, [pCRow3]

    fmla    z0.d, z16.d, alphaV
    fmla    z1.d, z17.d, alphaV
    fmla    z2.d, z18.d, alphaV
    fmla    z3.d, z19.d, alphaV

    st1d    z0.d, pMask8, [pCRow0]
    st1d    z1.d, pMask8, [pCRow1]
    st1d    z2.d, pMask8, [pCRow2]
    st1d    z3.d, pMask8, [pCRow3]
.endm
/** 4x4 ***********************************************************************/
.macro INIT_N4xM4
.endm

.macro MULTIPLY_N4xM4
.endm

.macro SAVE_N4xM4
.endm
/** 2x4 ***********************************************************************/
.macro INIT_N4xM2
.endm

.macro MULTIPLY_N4xM2
.endm

.macro SAVE_N4xM2
.endm
/** 1x4 ***********************************************************************/
.macro INIT_N4xM1
.endm

.macro MULTIPLY_N4xM1
.endm

.macro SAVE_N4xM1
.endm
/** 8x2 ***********************************************************************/
.macro INIT_N2xM8
    dup     z16.d, #0
    dup     z17.d, #0
.endm

.macro MULTIPLY_N2xM8
    ld1d    z0.d, pMask8, [pA]
    add     pA, pA, #64
    ld1d    z8.d, pMask2, [pB]
    add     pB, pB, #16

    fmla    z16.d, z0.d, z8.d[0]
    fmla    z17.d, z0.d, z8.d[1]
.endm

.macro SAVE_N2xM8
	dup     alphaV, alpha
	mov     pCRow0, pCRow
    add     pCRow1, pCRow0, LDC
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8, [pCRow0]
    ld1d    z1.d, pMask8, [pCRow1]

    fmla    z0.d, z16.d, alphaV
    fmla    z1.d, z17.d, alphaV

    st1d    z0.d, pMask8, [pCRow0]
    st1d    z1.d, pMask8, [pCRow1]
.endm
/** 4x2 ***********************************************************************/
.macro INIT_N2xM4
.endm

.macro MULTIPLY_N2xM4
.endm

.macro SAVE_N2xM4
.endm
/** 2x2 ***********************************************************************/
.macro INIT_N2xM2
.endm

.macro MULTIPLY_N2xM2
.endm

.macro SAVE_N2xM2
.endm
/** 1x2 ***********************************************************************/
.macro INIT_N2xM1
.endm

.macro MULTIPLY_N2xM1
.endm

.macro SAVE_N2xM1
.endm
/** 8x1 ***********************************************************************/
.macro INIT_N1xM8
    dup     z16.d, #0
.endm

.macro MULTIPLY_N1xM8
    ld1d    z0.d, pMask8, [pA]
    add     pA, pA, #64
    ld1d    z8.d, pMask1, [pB]
    add     pB, pB, #8

    fmla    z16.d, z0.d, z8.d[0]
.endm

.macro SAVE_N1xM8
	dup     alphaV, alpha
	mov     pCRow0, pCRow
    add     pCRow, pCRow, #128

    ld1d    z0.d, pMask8, [pCRow0]

    fmla    z0.d, z16.d, alphaV

    st1d    z0.d, pMask8, [pCRow0]
.endm
/** 4x1 ***********************************************************************/
.macro INIT_N1xM4
.endm

.macro MULTIPLY_N1xM4
.endm

.macro SAVE_N1xM4
.endm
/** 2x1 ***********************************************************************/
.macro INIT_N1xM2
.endm

.macro MULTIPLY_N1xM2
.endm

.macro SAVE_N1xM2
.endm
/** 1x1 ***********************************************************************/
.macro INIT_N1xM1
.endm

.macro MULTIPLY_N1xM1
.endm

.macro SAVE_N1xM1
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
    rdvl    temp, #1
    cmp     temp, #64
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
    mov     temp, #4
    whilelt p1.d, xzr, temp                     // 4 lane mask
    mov     temp, #2
    whilelt p2.d, xzr, temp                     // 2 lane mask
    mov     temp, #1
    whilelt p3.d, xzr, temp                     // 1 lane mask

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
    add     pC, pC, LDC, #3                     // pC = pC + LDC * 8
    mov     pA, origPA

/* N8xM8 **********************************************************************/
.N8xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .N8xM4_BEGIN

.N8xM8_INIT:
    INIT_N8xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N8xM8_LOOP:
    MULTIPLY_N8xM8
    subs    tempK, tempK, #1
    bne     .N8xM8_LOOP

.N8xM8_SAVE:
    SAVE_N8xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bne     .N8xM8_INIT

/* N8xM4 **********************************************************************/
.N8xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .N8xM2_BEGIN

.N8xM4_INIT:
    INIT_N8xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N8xM4_LOOP:
    MULTIPLY_N8xM4
    subs    tempK, tempK, #1
    bgt     .N8xM4_LOOP

.N8xM4_100:
    SAVE_N8xM4

/* N8xM2 **********************************************************************/
.N8xM2_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 2
    cmp     I, #0
    ble     .N8xM1_BEGIN

.N8xM2_INIT:
    INIT_N8xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N8xM2_LOOP:
    MULTIPLY_N8xM2
    subs    tempK, tempK, #1
    bgt     .N8xM2_LOOP

.N8xM2_100:
    SAVE_N8xM2

/* N8xM1 **********************************************************************/
.N8xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 1
    ble     .N8_END

.N8xM1_INIT:
    INIT_N8xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N8xM1_LOOP:
    MULTIPLY_N8xM1
    subs    tempK, tempK, #1
    bgt     .N8xM1_LOOP

.N8xM1_SAVE:
    SAVE_N8xM1

/* N8 END *********************************************************************/
.N8_END:
    lsl     tempK, origK, #6                    // tempk = K * 8 * sizeof(double)
    add     origPB, origPB, tempK               // B = B + tempK
    subs    J, J, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF MATRIX B REMAINS
    bgt     .N8_BEGIN

/* N4 *************************************************************************/
.N4_BEGIN:
    mov     J , origN
    tst     J , #7
    ble     .RESTORE_REGS                       // N WAS MULTIPLE OF 8
    tst     J , #4
    ble     .N2_BEGIN                           // N HAS NO 4x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, #2                     // pC = pC + LDC * 4
    mov     pA, origPA

/* N4xM8 **********************************************************************/
.N4xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .N4xM4_BEGIN

.N4xM8_INIT:
    INIT_N4xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N4xM8_LOOP:
    MULTIPLY_N4xM8
    subs    tempK, tempK, #1
    bne     .N4xM8_LOOP

.N4xM8_SAVE:
    SAVE_N4xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bne     .N4xM8_INIT

/* N4xM4 **********************************************************************/
.N4xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .N4xM2_BEGIN

.N4xM4_INIT:
    INIT_N4xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N4xM4_LOOP:
    MULTIPLY_N4xM4
    subs    tempK, tempK, #1
    bgt     .N4xM4_LOOP

.N4xM4_100:
    SAVE_N4xM4

/* N4xM2 **********************************************************************/
.N4xM2_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 2
    cmp     I, #0
    ble     .N4xM1_BEGIN

.N4xM2_INIT:
    INIT_N4xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N4xM2_LOOP:
    MULTIPLY_N4xM2
    subs    tempK, tempK, #1
    bgt     .N4xM2_LOOP

.N4xM2_100:
    SAVE_N4xM2

/* N4xM1 **********************************************************************/
.N4xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .N4_END

.N4xM1_INIT:
    INIT_N4xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N4xM1_LOOP:
    MULTIPLY_N4xM1
    subs    tempK, tempK, #1
    bgt     .N4xM1_LOOP

.N4xM1_SAVE:
    SAVE_N4xM1

/* N4 END *********************************************************************/
.N4_END:
    lsl     tempK, origK, #5                    // tempk = K * 4 * sizeof(double)
    add     origPB, origPB, tempK               // B = B + tempK

/* N2 *************************************************************************/
.N2_BEGIN:
    mov     J , origN
    tst     J , #3
    ble     .RESTORE_REGS                       // N WAS MULTIPLE OF 4
    tst     J , #2
    ble     .N1_BEGIN                           // N HAS NO 2x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, #1                     // pC = pC + LDC * 2
    mov     pA, origPA

/* N2xM8 **********************************************************************/
.N2xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .N2xM4_BEGIN

.N2xM8_INIT:
    INIT_N2xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N2xM8_LOOP:
    MULTIPLY_N2xM8
    subs    tempK, tempK, #1
    bgt     .N2xM8_LOOP

.N2xM8_SAVE:
    SAVE_N2xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bgt     .N2xM8_INIT

/* N2xM4 **********************************************************************/
.N2xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .N2xM2_BEGIN

.N2xM4_INIT:
    INIT_N2xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N2xM4_LOOP:
    MULTIPLY_N2xM4
    subs    tempK, tempK, #1
    bgt     .N2xM4_LOOP

.N2xM4_SAVE:
    SAVE_N2xM4

/* N2xM2 **********************************************************************/
.N2xM2_BEGIN:
    mov     I, origM
    asr     I, I, #1                            // M / 2
    cmp     I, #0
    ble     .N2xM1_BEGIN

.N2xM2_INIT:
    INIT_N2xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N2xM2_LOOP:
    MULTIPLY_N2xM4
    subs    tempK, tempK, #1
    bgt     .N2xM2_LOOP

.N2xM2_SAVE:
    SAVE2x2

/* N2xM1 **********************************************************************/
.N2xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .N2_END

.N2xM1_INIT:
    INIT_N2xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N2xM1_LOOP:
    MULTIPLY_N2xM1
    subs    tempK, tempK, #1
    bgt     .N2xM1_LOOP

.N2xM1_SAVE:
    SAVE1x2

/* N2 END *********************************************************************/
.N2_END:
    add     origPB, origPB, origK, lsl #4       // B = B + K * 2 * sizeof(double)

/* N1 *************************************************************************/
.N1_BEGIN:
    mov     J , origN
    tst     J , #1
    ble     .RESTORE_REGS                       // N WAS MULTIPLE OF 2
    mov     pCRow, pC
    add     pC, pC, LDC                         // pC = pC + LDC
    mov     pA, origPA

/* N1xM8 **********************************************************************/
.N1xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .N1xM4_BEGIN

    .align  5
.N1xM8_INIT:
    INIT_N1xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N1xM8_LOOP:
    MULTIPLY_N1xM8
    subs    tempK, tempK, #1
    bgt     .N1xM8_LOOP

.N1xM8_SAVE:
    SAVE_N1xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bgt     .N1xM8_INIT

/* N1xM4 **********************************************************************/
.N1xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .N1xM2_BEGIN

.N1xM4_INIT:
    INIT_N1xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N1xM4_LOOP:
    KERNE_N1xM4
    subs    tempK, tempK, #1
    bgt     .N1xM4_LOOP

.N1xM4_SAVE:
    SAVE_N1xM4

/* N1xM2 **********************************************************************/
.N1xM2_BEGIN:
    mov     I, origM
    asr     I, I, #1                            // M / 2
    cmp     I, #0
    ble     .N1xM1_BEGIN

.N1xM2_INIT:
    INIT2x1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N1xM2_LOOP:
    MULTIPLY_N1xM2
    subs    tempK, tempK, #1
    bgt     .N1xM2_42

.N1xM2_SAVE:
    SAVE_N1xM2

/* N1xM1 **********************************************************************/
.N1xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .N1_END

.N1xM1_INIT:
    INIT1x1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.N1xM1_LOOP:
    KERNEL1x1
    subs    tempK, tempK, #1
    bgt     .N1xM1_LOOP

.N1xM1_SAVE:
    SAVE1x1

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

