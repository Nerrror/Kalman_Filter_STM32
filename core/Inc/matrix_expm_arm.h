/*******************************************************************************
 * File: matrix_expm_cmsis_f32.h
 *
 * Brief:
 *   Header for single-precision matrix exponential function using CMSIS-DSP.
 *
 * Description:
 *   Declares the function prototype for computing the matrix exponential 
 *   e^A (size n x n) in single precision (float), leveraging CMSIS-DSP 
 *   for matrix multiplication and inversion. Follows the scaling-and-squaring 
 *   approach described by Moler & Van Loan.
 *
 * Usage:
 *   Include this header and link against the corresponding C implementation 
 *   (containing the matrix_expm1_cmsis_f32() definition). Ensure your project 
 *   is configured to use the single-precision CMSIS-DSP library and to enable 
 *   hardware floating-point support (if available).
 *
 * License:
 *   Adapted from John Burkardt's implementation from:
 *   https://people.sc.fsu.edu/~jburkardt/c_src/matrix_exponential/matrix_exponential.html
 *   This code can be distributed under the MIT license,
 *   provided you adapt it to your project’s requirements.
 ******************************************************************************/

#include <math.h>
#include "arm_math.h"
#include <string.h>
#include <stdlib.h>


#ifndef MATRIX_EXPM_ARM_H
#define MATRIX_EXPM_ARM_H

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Copy an n×n matrix (row-major) into another buffer.
 * @param n Dimension of the square matrix (n×n).
 * @param src Pointer to the source matrix data.
 * @param dst Pointer to the destination buffer (must have room for n×n floats).
 */
static void mat_copy_f32(int n, const float *src, float *dst);

/**
 * @brief Compute the infinity norm (maximum absolute row sum) of an n×n matrix A.
 * @param n Dimension of the square matrix (n×n).
 * @param A Pointer to the matrix data (row-major).
 * @return The infinity norm of A.
 */
static float mat_norm_li_f32(int n, const float *A);

/**
 * @brief Return base-2 logarithm of x. (log2 x = ln(x) / ln(2)).
 */
static float log2_f32(float x);

/**
 * @brief Return the maximum of two integers.
 */
static int i4_max(int a, int b);

/**
 * @brief Scale an n×n matrix in place by the scalar t.
 * @param n Matrix dimension.
 * @param t Scalar.
 * @param A Pointer to the matrix data (in row-major order).
 */
static void mat_scale_f32(int n, float t, float *A);

/**
 * @brief Compute C = alpha*A + beta*B for n×n matrices (row-major).
 * @param n  Matrix dimension.
 * @param alpha Scalar multiplier for A.
 * @param A Pointer to matrix A data.
 * @param beta  Scalar multiplier for B.
 * @param B Pointer to matrix B data.
 * @param C Pointer to output data (may be the same as A or B if needed).
 */
static void mat_add_f32(int n,
                        float alpha, const float *A,
                        float beta,  const float *B,
                        float *C);

/**
 * @brief Set an n×n ones matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param I Pointer to the buffer that will hold the ones (must have space for n×n).
 */

void mat_ones_f32(int n, float32_t *II);
void mat_ones_f32_nxm(int n, int m, float32_t *II);

/**
 * @brief Set an n×n zeros matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param I Pointer to the buffer that will hold the zeros (must have space for n×n).
 */

void mat_zeros_f32(int n, float32_t *II);
void mat_zeros_f32_nxm(int n, int m, float32_t *II);

/**
 * @brief Set an n×n identity matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param I Pointer to the buffer that will hold the identity (must have space for n×n).
 */

void mat_eye_f32(int n, float32_t *II);


/**
 * @brief Compute the matrix exponential e^A for a single-precision n x n matrix A,
 *        using the scaling-and-squaring algorithm and CMSIS-DSP for matrix
 *        multiplication (arm_mat_mult_f32) and matrix inversion (arm_mat_inverse_f32).
 *        The matrix A is modified in-place to e^A.
 *
 * @param[in]  n     The size of the matrix (n x n).
 * @param[in]  a     Pointer to input matrix A, of size n*n.
 *
 * @return int 0 for success, 1 for error.
 */
int matrix_expm_cmsis_f32(const arm_matrix_instance_f32* A);


arm_status mat_vec_mult_f32(const arm_matrix_instance_f32 *pSrcA, const float32_t *pSrcVec, float32_t *pDstVec);

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_EXPM_ARM_H */
