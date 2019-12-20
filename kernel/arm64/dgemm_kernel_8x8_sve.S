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
#define pC              x5
#define LDC             x6
#define temp            x7
#define tempK           x8
#define I               x9
#define J               x10
#define pB              x11
#define pCRow0          x12
#define pCRow1          x13
#define pCRow2          x14
#define pCRow3          x15
#define pA              x16
#define alpha           x17

/*******************************************************************************
* Macro definitions
*******************************************************************************/
/** 8x8 ***********************************************************************/
.macro INIT_N8xM8
.endm

.macro KERNEL_N8xM8_MULTIPLY
.endm

.macro SAVE_N8xM8
.endm
/** 4x8 ***********************************************************************/
.macro INIT_N8xM4
.endm

.macro KERNEL_N8xM4_MULTIPLY
.endm

.macro SAVE_N8xM4
.endm
/** 2x8 ***********************************************************************/
.macro INIT_N8xM2
.endm

.macro KERNEL_N8xM2_MULTIPLY
.endm

.macro SAVE_N8xM2
.endm
/** 1x8 ***********************************************************************/
.macro INIT_N8xM1
.endm

.macro KERNEL_N8xM1_MULTIPLY
.endm

.macro SAVE_N8xM1
.endm
/** 8x4 ***********************************************************************/
.macro INIT_N4xM8
.endm

.macro KERNEL_N4xM8_MULTIPLY
.endm

.macro SAVE_N4xM8
.endm
/** 4x4 ***********************************************************************/
.macro INIT_N4xM4
.endm

.macro KERNEL_N4xM4_MULTIPLY
.endm

.macro SAVE_N4xM4
.endm
/** 2x4 ***********************************************************************/
.macro INIT_N4xM2
.endm

.macro KERNEL_N4xM2_MULTIPLY
.endm

.macro SAVE_N4xM2
.endm
/** 1x4 ***********************************************************************/
.macro INIT_N4xM1
.endm

.macro KERNEL_N4xM1_MULTIPLY
.endm

.macro SAVE_N4xM1
.endm
/** 8x2 ***********************************************************************/
.macro INIT_N2xM8
.endm

.macro KERNEL_N2xM8_MULTIPLY
.endm

.macro SAVE_N2xM8
.endm
/** 4x2 ***********************************************************************/
.macro INIT_N2xM4
.endm

.macro KERNEL_N2xM4_MULTIPLY
.endm

.macro SAVE_N2xM4
.endm
/** 2x2 ***********************************************************************/
.macro INIT_N2xM2
.endm

.macro KERNEL_N2xM2_MULTIPLY
.endm

.macro SAVE_N2xM2
.endm
/** 1x2 ***********************************************************************/
.macro INIT_N2xM1
.endm

.macro KERNEL_N2xM1_MULTIPLY
.endm

.macro SAVE_N2xM1
.endm
/** 8x1 ***********************************************************************/
.macro INIT_N1xM8
.endm

.macro KERNEL_N1xM8_MULTIPLY
.endm

.macro SAVE_N1xM8
.endm
/** 4x1 ***********************************************************************/
.macro INIT_N1xM4
.endm

.macro KERNEL_N1xM4_MULTIPLY
.endm

.macro SAVE_N1xM4
.endm
/** 2x1 ***********************************************************************/
.macro INIT_N1xM2
.endm

.macro KERNEL_N1xM2_MULTIPLY
.endm

.macro SAVE_N1xM2
.endm
/** 1x1 ***********************************************************************/
.macro INIT_N1xM1
.endm

.macro KERNEL_N1xM1_MULTIPLY
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

    lsl     LDC, LDC, #3                        // ldc = ldc * 8
    fmov    alpha, d0

    mov     J, origN
    asr     J, J, #3                            // N = N / 8
    cmp     J, #0
    ble     .Ldgemm_kernel_N4_BEGIN


/* N8 *************************************************************************/
.Ldgemm_kernel_N8_BEGIN:
    mov     pCRow, pC
    add     pC, pC, LDC, #3                     // pC = pC + LDC * 8
    mov     pA, origPA

/* N8xM8 **********************************************************************/
.Ldgemm_kernel_N8xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .Ldgemm_kernel_N8xM4_BEGIN

.Ldgemm_kernel_N8xM8_INIT:
    INIT_N8xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N8xM8_LOOP:
    KERNEL_N8xM8_MULTIPLY
    subs    tempK, tempK, #1
    bne     .Ldgemm_kernel_N8xM8_LOOP

.Ldgemm_kernel_N8xM8_SAVE:
    SAVE_N8xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bne     .Ldgemm_kernel_N8xM8_INIT

/* N8xM4 **********************************************************************/
.Ldgemm_kernel_N8xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .Ldgemm_kernel_N8xM2_BEGIN

.Ldgemm_kernel_N8xM4_INIT:
    INIT_N8xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N8xM4_LOOP:
    KERNEL_N8xM4_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N8xM4_LOOP

.Ldgemm_kernel_N8xM4_100:
    SAVE_N8xM4

/* N8xM2 **********************************************************************/
.Ldgemm_kernel_N8xM2_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 2
    cmp     I, #0
    ble     .Ldgemm_kernel_N8xM1_BEGIN

.Ldgemm_kernel_N8xM2_INIT:
    INIT_N8xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N8xM2_LOOP:
    KERNEL_N8xM2_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N8xM2_LOOP

.Ldgemm_kernel_N8xM2_100:
    SAVE_N8xM2

/* N8xM1 **********************************************************************/
.Ldgemm_kernel_N8xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 1
    ble     .Ldgemm_kernel_N8_END

.Ldgemm_kernel_N8xM1_INIT:
    INIT_N8xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N8xM1_LOOP:
    KERNEL_N8xM1_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N8xM1_LOOP

.Ldgemm_kernel_N8xM1_SAVE:
    SAVE_N8xM1

/* N8 END *********************************************************************/
.Ldgemm_kernel_N8_END:
    lsl     tempK, origK, #6                    // tempk = K * 8 * sizeof(double)
    add     origPB, origPB, tempK               // B = B + tempK
    subs    J, J, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF MATRIX B REMAINS
    bgt     .Ldgemm_kernel_N8_BEGIN

/* N4 *************************************************************************/
.Ldgemm_kernel_N4_BEGIN:
    mov     J , origN
    tst     J , #7
    ble     .Ldgemm_kernel_L999                 // N WAS MULTIPLE OF 8
    tst     J , #4
    ble     .Ldgemm_kernel_N2_BEGIN             // N HAS NO 4x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, #2                     // pC = pC + LDC * 4
    mov     pA, origPA

/* N4xM8 **********************************************************************/
.Ldgemm_kernel_N4xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .Ldgemm_kernel_N4xM4_BEGIN

.Ldgemm_kernel_N4xM8_INIT:
    INIT_N8xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N4xM8_LOOP:
    KERNEL_N8xM4_MULTIPLY
    subs    tempK, tempK, #1
    bne     .Ldgemm_kernel_N4xM8_LOOP

.Ldgemm_kernel_N4xM8_SAVE:
    SAVE_N8xM4
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bne     .Ldgemm_kernel_N4xM8_INIT

/* N4xM4 **********************************************************************/
.Ldgemm_kernel_N4xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .Ldgemm_kernel_N4xM2_BEGIN

.Ldgemm_kernel_N4xM4_INIT:
    INIT_N4xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N4xM4_LOOP:
    KERNEL_N4xM4_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N4xM4_LOOP

.Ldgemm_kernel_N4xM4_100:
    SAVE_N4xM4

/* N4xM2 **********************************************************************/
.Ldgemm_kernel_N4xM2_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 2
    cmp     I, #0
    ble     .Ldgemm_kernel_N4xM1_BEGIN

.Ldgemm_kernel_N4xM2_INIT:
    INIT_N4xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N4xM2_LOOP:
    KERNEL_N4xM2_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N4xM2_LOOP

.Ldgemm_kernel_N4xM2_100:
    SAVE_N4xM2

/* N4xM1 **********************************************************************/
.Ldgemm_kernel_N4xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .Ldgemm_kernel_N4_END

.Ldgemm_kernel_N4xM1_INIT:
    INIT_N4xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N4xM1_LOOP:
    KERNEL_N4xM1_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N4xM1_LOOP

.Ldgemm_kernel_N4xM1_SAVE:
    SAVE_N4xM1

/* N4 END *********************************************************************/
.Ldgemm_kernel_N4_END:
    lsl     tempK, origK, #5                    // tempk = K * 4 * sizeof(double)
    add     origPB, origPB, tempK               // B = B + tempK

/* N2 *************************************************************************/
.Ldgemm_kernel_N2_BEGIN:
    mov     J , origN
    tst     J , #3
    ble     .Ldgemm_kernel_L999                 // N WAS MULTIPLE OF 4
    tst     J , #2
    ble     .Ldgemm_kernel_N1_BEGIN             // N HAS NO 2x CHUNKS
    mov     pCRow, pC
    add     pC, pC, LDC, #1                     // pC = pC + LDC * 2
    mov     pA, origPA

/* N2xM8 **********************************************************************/
.Ldgemm_kernel_N2xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .Ldgemm_kernel_N2xM4_BEGIN

.Ldgemm_kernel_N2xM8_INIT:
    INIT_N2xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N2xM8_LOOP:
    KERNEL_N2xM8_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N2xM8_LOOP

.Ldgemm_kernel_N2xM8_SAVE:
    SAVE_N2xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bgt     .Ldgemm_kernel_N2xM8_INIT

/* N2xM4 **********************************************************************/
.Ldgemm_kernel_N2xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .Ldgemm_kernel_N2xM2_BEGIN

.Ldgemm_kernel_N2xM4_INIT:
    INIT_N2xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N2xM4_LOOP:
    KERNEL_N2xM4_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N2xM4_LOOP

.Ldgemm_kernel_N2xM4_SAVE:
    SAVE_N2xM4

/* N2xM2 **********************************************************************/
.Ldgemm_kernel_N2xM2_BEGIN:
    mov     I, origM
    asr     I, I, #1                            // M / 2
    cmp     I, #0
    ble     .Ldgemm_kernel_N2xM1_BEGIN

.Ldgemm_kernel_N2xM2_INIT:
    INIT_N2xM2
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N2xM2_LOOP:
    KERNEL_N2xM4_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N2xM2_LOOP

.Ldgemm_kernel_N2xM2_SAVE:
    SAVE2x2

/* N2xM1 **********************************************************************/
.Ldgemm_kernel_N2xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .Ldgemm_kernel_N2_END

.Ldgemm_kernel_N2xM1_INIT:
    INIT_N2xM1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N2xM1_LOOP:
    KERNEL_N2xM1_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N2xM1_LOOP

.Ldgemm_kernel_N2xM1_SAVE:
    SAVE1x2

/* N2 END *********************************************************************/
.Ldgemm_kernel_N2_END:
    add     origPB, origPB, origK, lsl #4       // B = B + K * 2 * sizeof(double)

/* N1 *************************************************************************/
.Ldgemm_kernel_N1_BEGIN:
    mov     J , origN
    tst     J , #1
    ble     .Ldgemm_kernel_L999                 // N WAS MULTIPLE OF 2
    mov     pCRow, pC
    add     pC, pC, LDC                         // pC = pC + LDC
    mov     pA, origPA

/* N1xM8 **********************************************************************/
.Ldgemm_kernel_N1xM8_BEGIN:
    mov     I, origM
    asr     I, I, #3                            // M / 8
    cmp     I, #0
    ble     .Ldgemm_kernel_N1xM4_BEGIN

    .align  5
.Ldgemm_kernel_N1xM8_INIT:
    INIT_N1xM8
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N1xM8_LOOP:
    KERNEL_N1xM8_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N1xM8_LOOP

.Ldgemm_kernel_N1xM8_SAVE:
    SAVE_N1xM8
    subs    I, I, #1                            // CHECK WHETHER MORE 8x CHUNKS
                                                // OF A IS REMAINING
    bgt     .Ldgemm_kernel_N1xM8_INIT

/* N1xM4 **********************************************************************/
.Ldgemm_kernel_N1xM4_BEGIN:
    mov     I, origM
    asr     I, I, #2                            // M / 4
    cmp     I, #0
    ble     .Ldgemm_kernel_N1xM2_BEGIN

.Ldgemm_kernel_N1xM4_INIT:
    INIT_N1xM4
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N1xM4_LOOP:
    KERNE_N1xM4_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N1xM4_LOOP

.Ldgemm_kernel_N1xM4_SAVE:
    SAVE_N1xM4

/* N1xM2 **********************************************************************/
.Ldgemm_kernel_N1xM2_BEGIN:
    mov     I, origM
    asr     I, I, #1                            // M / 2
    cmp     I, #0
    ble     .Ldgemm_kernel_N1xM1_BEGIN

.Ldgemm_kernel_N1xM2_INIT:
    INIT2x1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N1xM2_LOOP:
    KERNEL_N1xM2_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N1xM2_42

.Ldgemm_kernel_N1xM2_SAVE:
    SAVE_N1xM2

/* N1xM1 **********************************************************************/
.Ldgemm_kernel_N1xM1_BEGIN:
    mov     I, origM
    tst     I, #1                               // M % 2
    ble     .Ldgemm_kernel_N1_END

.Ldgemm_kernel_N1xM1_INIT:
    INIT1x1
    mov     pB, origPB
    mov     tempK, K

    .align  5
.Ldgemm_kernel_N1xM1_LOOP:
    KERNEL1x1_MULTIPLY
    subs    tempK, tempK, #1
    bgt     .Ldgemm_kernel_N1xM1_LOOP

.Ldgemm_kernel_N1xM1_SAVE:
    SAVE1x1

/* THE END ********************************************************************/
.Ldgemm_kernel_L999:
    mov     x0, #0                              // SET RETURN VALUE
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
    ret

    EPILOGUE

