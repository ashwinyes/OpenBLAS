#include "common.h"
#include "arm_neon.h"

#define PREFA 64
#define PREFB 32
#define PREFC 8

#define FMLA_LANE(R, A, B, N) \
  R = vfmaq_laneq_f64(R, A, B, N)

#define FMUL_LANE(R, A, B, N) \
  R = vmulq_laneq_f64(A, B, N)

#define LOAD(R, ADDR) \
  R = vld1q_f64(ADDR)

#define LOAD_LANE(R, ADDR, LANE) \
  R = vld1q_lane_f64(ADDR, R, LANE)

#define FMLA_SAVE(ADDR, MUL1, MUL2) \
  vst1q_f64(ADDR, vfmaq_f64(vld1q_f64(ADDR), MUL1, MUL2))

#define PREFETCH(ADDR) \
  __builtin_prefetch(ADDR)

#define PREFETCHx2(ADDR1, ADDR2) \
  PREFETCH(ADDR1); \
  PREFETCH(ADDR2)

#define PREFETCHx4(ADDR1, ADDR2, ADDR3, ADDR4) \
  PREFETCHx2(ADDR1, ADDR2); \
  PREFETCHx2(ADDR3, ADDR4)

#define PREFETCH_C_x4(OFFSET) \
  PREFETCHx4(&pC0[OFFSET], &pC1[OFFSET], &pC2[OFFSET], &pC3[OFFSET]);

#define PREFETCH_C_x2(OFFSET) \
  PREFETCHx2(&pC0[OFFSET], &pC1[OFFSET]);

#define PREFETCH_C_x1(OFFSET) \
  PREFETCH(&pC0[OFFSET]);

#define EARLY_PREFETCH(NVAL) \
  PREFETCH_C_x##NVAL(0); \
  PREFETCHx4(&pA[0], &pA[8], &pA[16], &pA[24]); \
  PREFETCHx4(&pA[32], &pA[40], &pA[48], &pA[56])

#define COMPUTE_LOOP(NVAL, MVAL) \
  LOAD_N##NVAL##xM##MVAL; \
  CALC_N##NVAL##xM##MVAL(FMUL_LANE); \
  PREFETCHx2(&pA[PREFA], &pB[PREFB]); \
  PREFETCH_C_x##NVAL(PREFC); \
  for (k = 0; k < ((K - 1) / 2); k += 1) { \
    LOAD_N##NVAL##xM##MVAL; \
    CALC_N##NVAL##xM##MVAL(FMLA_LANE); \
    PREFETCH(&pA[PREFA]); \
    LOAD_N##NVAL##xM##MVAL; \
    CALC_N##NVAL##xM##MVAL(FMLA_LANE); \
    PREFETCHx2(&pA[PREFA], &pB[PREFB]); \
  } \
  if ((K - 1) & 1) { \
    LOAD_N##NVAL##xM##MVAL; \
    CALC_N##NVAL##xM##MVAL(FMLA_LANE); \
    PREFETCHx2(&pA[PREFA], &pB[PREFB]); \
  } \
  SAVE_N##NVAL##xM##MVAL


#pragma GCC push_options
#pragma GCC optimize ("-Ofast", "-fprefetch-loop-arrays", "-funroll-loops")
int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, double ALPHA, double * A,
    double * B, double * C,BLASLONG LDC)
{
  BLASLONG i, j, k;
  double *pC0, *pC1, *pC2, *pC3;
  double *pA,*pB;
  double r00, r10, r20, r30;
  double lA0;
  double lB0, lB1, lB2, lB3;
  float64x2_t vr00, vr01, vr02, vr03;
  float64x2_t vr10, vr11, vr12, vr13;
  float64x2_t vr20, vr21, vr22, vr23;
  float64x2_t vr30, vr31, vr32, vr33;
  float64x2_t vlA0, vlA1, vlA2, vlA3;
  float64x2_t vlB0, vlB1;
  float64x2_t valpha = vdupq_n_f64(ALPHA);

  for (j = 0; j < (N >> 2); j += 1) {
    pC0 = C + LDC * 0;
    pC1 = C + LDC * 1;
    pC2 = C + LDC * 2;
    pC3 = C + LDC * 3;
    C += LDC * 4;
    pA = A;

    EARLY_PREFETCH(4);

    for (i = 0; i < (M >> 3); i += 1) {
#define LOAD_N4xM8 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD(vlA2, &pA[4]); \
      LOAD(vlA3, &pA[6]); \
      LOAD(vlB0, &pB[0]); \
      LOAD(vlB1, &pB[2]); \
      pA += 8; \
      pB += 4;

#define CALC_N4xM8(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0); \
      OP(vr02, vlA2, vlB0, 0); \
      OP(vr03, vlA3, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \
      OP(vr11, vlA1, vlB0, 1); \
      OP(vr12, vlA2, vlB0, 1); \
      OP(vr13, vlA3, vlB0, 1); \
      OP(vr20, vlA0, vlB1, 0); \
      OP(vr21, vlA1, vlB1, 0); \
      OP(vr22, vlA2, vlB1, 0); \
      OP(vr23, vlA3, vlB1, 0); \
      OP(vr30, vlA0, vlB1, 1); \
      OP(vr31, vlA1, vlB1, 1); \
      OP(vr32, vlA2, vlB1, 1); \
      OP(vr33, vlA3, vlB1, 1)

#define SAVE_N4xM8 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      FMLA_SAVE(&pC0[4], vr02, valpha); \
      FMLA_SAVE(&pC0[6], vr03, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      FMLA_SAVE(&pC1[2], vr11, valpha); \
      FMLA_SAVE(&pC1[4], vr12, valpha); \
      FMLA_SAVE(&pC1[6], vr13, valpha); \
      FMLA_SAVE(&pC2[0], vr20, valpha); \
      FMLA_SAVE(&pC2[2], vr21, valpha); \
      FMLA_SAVE(&pC2[4], vr22, valpha); \
      FMLA_SAVE(&pC2[6], vr23, valpha); \
      FMLA_SAVE(&pC3[0], vr30, valpha); \
      FMLA_SAVE(&pC3[2], vr31, valpha); \
      FMLA_SAVE(&pC3[4], vr32, valpha); \
      FMLA_SAVE(&pC3[6], vr33, valpha); \
      pC0 += 8; \
      pC1 += 8; \
      pC2 += 8; \
      pC3 += 8;

      pB = B;
      COMPUTE_LOOP(4, 8);
    }

    if (M & 4) {
#define LOAD_N4xM4 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD(vlB0, &pB[0]); \
      LOAD(vlB1, &pB[2]); \
      pA += 4; \
      pB += 4;

#define CALC_N4xM4(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \
      OP(vr11, vlA1, vlB0, 1); \
      OP(vr20, vlA0, vlB1, 0); \
      OP(vr21, vlA1, vlB1, 0); \
      OP(vr30, vlA0, vlB1, 1); \
      OP(vr31, vlA1, vlB1, 1);

#define SAVE_N4xM4 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      FMLA_SAVE(&pC1[2], vr11, valpha); \
      FMLA_SAVE(&pC2[0], vr20, valpha); \
      FMLA_SAVE(&pC2[2], vr21, valpha); \
      FMLA_SAVE(&pC3[0], vr30, valpha); \
      FMLA_SAVE(&pC3[2], vr31, valpha); \
      pC0 += 4; \
      pC1 += 4; \
      pC2 += 4; \
      pC3 += 4;

      pB = B;
      COMPUTE_LOOP(4, 4);
    }

    if (M & 2) {
#define LOAD_N4xM2 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlB0, &pB[0]); \
      LOAD(vlB1, &pB[2]); \
      pA += 2; \
      pB += 4;

#define CALC_N4xM2(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \
      OP(vr20, vlA0, vlB1, 0); \
      OP(vr30, vlA0, vlB1, 1); \

#define SAVE_N4xM2 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      FMLA_SAVE(&pC2[0], vr20, valpha); \
      FMLA_SAVE(&pC3[0], vr30, valpha); \
      pC0 += 2; \
      pC1 += 2; \
      pC2 += 2; \
      pC3 += 2;

      pB = B;
      COMPUTE_LOOP(4, 2);
    }

    if (M & 1) {
#define LOAD_N4xM1 \
      lA0 = pA[0]; \
      lB0 = pB[0]; \
      lB1 = pB[1]; \
      lB2 = pB[2]; \
      lB3 = pB[3]; \
      pA += 1; \
      pB += 4;

#define CALC_N4xM1(OP) CALC_N4xM1_##OP

#define CALC_N4xM1_FMUL_LANE \
      r00 = lA0 * lB0; \
      r10 = lA0 * lB1; \
      r20 = lA0 * lB2; \
      r30 = lA0 * lB3;

#define CALC_N4xM1_FMLA_LANE \
      r00 += lA0 * lB0; \
      r10 += lA0 * lB1; \
      r20 += lA0 * lB2; \
      r30 += lA0 * lB3;

#define SAVE_N4xM1 \
      pC0[0] += r00 * ALPHA; \
      pC1[0] += r10 * ALPHA; \
      pC2[0] += r20 * ALPHA; \
      pC3[0] += r30 * ALPHA; \
      pC0 += 1; \
      pC1 += 1; \
      pC2 += 1; \
      pC3 += 1;

      pB = B;
      COMPUTE_LOOP(4, 1);
    }

    B += K * 4;
  }

  if (N & 2) {
    pC0 = C + LDC * 0;
    pC1 = C + LDC * 1;
    C += LDC * 2;
    pA = A;

    EARLY_PREFETCH(2);

    for (i = 0; i < (M >> 3); i += 1) {
#define LOAD_N2xM8 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD(vlA2, &pA[4]); \
      LOAD(vlA3, &pA[6]); \
      LOAD(vlB0, &pB[0]); \
      pA += 8; \
      pB += 2;

#define CALC_N2xM8(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0); \
      OP(vr02, vlA2, vlB0, 0); \
      OP(vr03, vlA3, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \
      OP(vr11, vlA1, vlB0, 1); \
      OP(vr12, vlA2, vlB0, 1); \
      OP(vr13, vlA3, vlB0, 1);

#define SAVE_N2xM8 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      FMLA_SAVE(&pC0[4], vr02, valpha); \
      FMLA_SAVE(&pC0[6], vr03, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      FMLA_SAVE(&pC1[2], vr11, valpha); \
      FMLA_SAVE(&pC1[4], vr12, valpha); \
      FMLA_SAVE(&pC1[6], vr13, valpha); \
      pC0 += 8; \
      pC1 += 8;

      pB = B;
      COMPUTE_LOOP(2, 8);
    }

    if (M & 4) {
#define LOAD_N2xM4 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD(vlB0, &pB[0]); \
      pA += 4; \
      pB += 2;

#define CALC_N2xM4(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \
      OP(vr11, vlA1, vlB0, 1); \

#define SAVE_N2xM4 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      FMLA_SAVE(&pC1[2], vr11, valpha); \
      pC0 += 4; \
      pC1 += 4;

      pB = B;
      COMPUTE_LOOP(2, 4);
    }

    if (M & 2) {
#define LOAD_N2xM2 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlB0, &pB[0]); \
      pA += 2; \
      pB += 2;

#define CALC_N2xM2(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr10, vlA0, vlB0, 1); \

#define SAVE_N2xM2 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC1[0], vr10, valpha); \
      pC0 += 2; \
      pC1 += 2;

      pB = B;
      COMPUTE_LOOP(2, 2);
    }

    if (M & 1) {
#define LOAD_N2xM1 \
      lA0 = pA[0]; \
      lB0 = pB[0]; \
      lB1 = pB[1]; \
      pA += 1; \
      pB += 2;

#define CALC_N2xM1(OP) CALC_N2xM1_##OP

#define CALC_N2xM1_FMUL_LANE \
      r00 = lA0 * lB0; \
      r10 = lA0 * lB1;

#define CALC_N2xM1_FMLA_LANE \
      r00 += lA0 * lB0; \
      r10 += lA0 * lB1;

#define SAVE_N2xM1 \
      pC0[0] += r00 * ALPHA; \
      pC1[0] += r10 * ALPHA; \
      pC0 += 2; \
      pC1 += 2;

      pB = B;
      COMPUTE_LOOP(2, 1);
    }

    B += K * 2;
  }

  if (N & 1) {
    pC0 = C + LDC * 0;
    C += LDC * 1;
    pA = A;

    EARLY_PREFETCH(1);

    for (i = 0; i < (M >> 3); i += 1) {
#define LOAD_N1xM8 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD(vlA2, &pA[4]); \
      LOAD(vlA3, &pA[6]); \
      LOAD_LANE(vlB0, &pB[0], 0); \
      pA += 8; \
      pB += 1;

#define CALC_N1xM8(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0); \
      OP(vr02, vlA2, vlB0, 0); \
      OP(vr03, vlA3, vlB0, 0);

#define SAVE_N1xM8 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      FMLA_SAVE(&pC0[4], vr02, valpha); \
      FMLA_SAVE(&pC0[6], vr03, valpha); \
      pC0 += 8;

      pB = B;
      COMPUTE_LOOP(1, 8);
    }

    if (M & 4) {
#define LOAD_N1xM4 \
      LOAD(vlA0, &pA[0]); \
      LOAD(vlA1, &pA[2]); \
      LOAD_LANE(vlB0, &pB[0], 0); \
      pA += 4; \
      pB += 1;

#define CALC_N1xM4(OP) \
      OP(vr00, vlA0, vlB0, 0); \
      OP(vr01, vlA1, vlB0, 0);

#define SAVE_N1xM4 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      FMLA_SAVE(&pC0[2], vr01, valpha); \
      pC0 += 4;

      pB = B;
      COMPUTE_LOOP(1, 4);
    }

    if (M & 2) {
#define LOAD_N1xM2 \
      LOAD(vlA0, &pA[0]); \
      LOAD_LANE(vlB0, &pB[0], 0); \
      pA += 2; \
      pB += 1;

#define CALC_N1xM2(OP) \
      OP(vr00, vlA0, vlB0, 0);

#define SAVE_N1xM2 \
      FMLA_SAVE(&pC0[0], vr00, valpha); \
      pC0 += 2;

      pB = B;
      COMPUTE_LOOP(1, 2);
    }

    if (M & 1) {
#define LOAD_N1xM1 \
      lA0 = pA[0]; \
      lB0 = pB[0]; \
      pA += 1; \
      pB += 1;

#define CALC_N1xM1(OP) CALC_N1xM1_##OP

#define CALC_N1xM1_FMUL_LANE \
      r00 = lA0 * lB0;

#define CALC_N1xM1_FMLA_LANE \
      r00 += lA0 * lB0;

#define SAVE_N1xM1 \
      pC0[0] += r00 * ALPHA; \
      pC0 += 1;

      pB = B;
      COMPUTE_LOOP(1, 1);
    }

  }
  return 0;
}
#pragma GCC pop_options
