/*
Adapted from John Burkardt's implementation,
https://people.sc.fsu.edu/~jburkardt/c_src/matrix_exponential/matrix_exponential.html 
for use with STM32 and CMSIS-DSP by Alex Luce, 2025
*/ 

#include <math.h>
#include <string.h>
#include "arm_math.h"
#include <stdlib.h>

#include "matrix_expm_arm.h"

#ifndef MATRIX_EXPM_ARM
#define MATRIX_EXPM_ARM

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
/**
 * Set a maximum dimension. This code handles up to n = MAX_N.
 * Adjust if you need larger or smaller maximum.
 */
#define MAX_N 64

// -----------------------------------------------------------------------------
// Helper Functions (Static Memory Versions)
// -----------------------------------------------------------------------------

/**
 * @brief Copy an n×n matrix (row-major) into another buffer.
 * @param n Dimension of the square matrix (n×n).
 * @param src Pointer to the source matrix data.
 * @param dst Pointer to the destination buffer (must have room for n×n floats).
 */
static void mat_copy_f32(int n, const float *src, float *dst)
{
    // We assume n ≤ MAX_N in the calling code
    memcpy(dst, src, (size_t)(n * n) * sizeof(float));
}

/**
 * @brief Compute the infinity norm (maximum absolute row sum) of an n×n matrix A.
 * @param n Dimension of the square matrix (n×n).
 * @param A Pointer to the matrix data (row-major).
 * @return The infinity norm of A.
 */
static float mat_norm_li_f32(int n, const float *A)
{
    float max_sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            row_sum += fabsf(A[i * n + j]);
        }
        if (row_sum > max_sum)
        {
            max_sum = row_sum;
        }
    }
    return max_sum;
}

/**
 * @brief Return base-2 logarithm of x. (log2 x = ln(x) / ln(2)).
 */
static float log2_f32(float x)
{
    return logf(x) / logf(2.0f);
}

/**
 * @brief Return the maximum of two integers.
 */
static int i4_max(int a, int b)
{
    return (a > b) ? a : b;
}

/**
 * @brief Scale an n×n matrix in place by the scalar t.
 * @param n Matrix dimension.
 * @param t Scalar.
 * @param A Pointer to the matrix data (in row-major order).
 */
static void mat_scale_f32(int n, float t, float *A)
{
    arm_matrix_instance_f32 mat;
    arm_mat_init_f32(&mat, n, n, A);
    // CMSIS-DSP supports in-place scaling
    arm_mat_scale_f32(&mat, t, &mat);
}

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
                        float *C)
{
    for (int i = 0; i < n * n; i++)
    {
        C[i] = alpha * A[i] + beta * B[i];
    }
}

/**
 * @brief Set an n×n ones matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param II Pointer to the buffer that will hold the ones (must have space for n×n).
 */
void mat_ones_f32(int n, float32_t *II)
{
    for (int i = 0; i < n * n; i++) {
        II[i] = 1.0f;
    }
}

void mat_ones_f32_nxm(int n, int m, float32_t *II)
{
    for (int i = 0; i < n * m; i++) {
        II[i] = 1.0f;
    }
}


/**
 * @brief Set an n×n zeros matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param II Pointer to the buffer that will hold the zeros (must have space for n×n).
 */
void mat_zeros_f32(int n, float32_t *II)
{
    for (int i = 0; i < n * n; i++) {
        II[i] = 0.0f;
    }
}

void mat_zeros_f32_nxm(int n, int m, float32_t *II)
{
    for (int i = 0; i < n * m; i++) {
        II[i] = 0.0f;
    }
}


/**
 * @brief Set an n×n identity matrix in the given buffer.
 * @param n Dimension of the square matrix.
 * @param II Pointer to the buffer that will hold the identity (must have space for n×n).
 */
void mat_eye_f32(int n, float32_t *II)
{
    // zero fill
    memset(II, 0, (size_t)(n * n) * sizeof(float32_t));
    for (int i = 0; i < n; i++)
    {
        II[i * n + i] = 1.0f;
    }
}


// -----------------------------------------------------------------------------
// Matrix Exponential (Static Buffers)
// -----------------------------------------------------------------------------

/**
 * @brief Compute the matrix exponential exp(A) using scaling and squaring (Padé approximation).
 *
 * @param A Pointer to an arm_matrix_instance_f32 structure (square).
 * @return Pointer to a static arm_matrix_instance_f32 that contains exp(A) in its pData.
 *         Returns NULL on error (e.g., n > MAX_N or matrix inversion fails).
 *
 * @warning Not reentrant (always returns the address of one static buffer).
 * @warning If n > MAX_N, the function returns NULL.
 */
int matrix_expm_cmsis_f32(const arm_matrix_instance_f32* A)
{
    // Make sure n is within our static buffer capability
    const int n = A->numRows;
    if ((n != A->numCols) || (n > MAX_N))
    {
        // Error: not square or dimension too large
        return 0;
    }

    // We fix the maximum degree for the Padé approximation
    const int Q = 6;

    // -- Statically allocate all intermediate buffers on the stack --
    float a2Buf[MAX_N * MAX_N];
    float xBuf [MAX_N * MAX_N];
    float eBuf [MAX_N * MAX_N];
    float dBuf [MAX_N * MAX_N];
    float tmpBuf[MAX_N * MAX_N];
    float dInvBuf[MAX_N * MAX_N];
    float squareBuf[MAX_N * MAX_N];

    // 1) Copy A into a2Buf
    mat_copy_f32(n, A->pData, a2Buf);

    // 2) Infinity norm
    float a_norm = mat_norm_li_f32(n, a2Buf);

    // 3) Compute s = max(0, floor(log2(a_norm)) + 2)
    int ee = (int)(log2_f32(a_norm)) + 1;
    int s  = i4_max(0, ee + 1);

    // 4) Scale by 1 / 2^s
    float t = 1.0f / powf(2.0f, (float)s);
    mat_scale_f32(n, t, a2Buf);

    // 5) Copy scaled a2Buf into xBuf
    mat_copy_f32(n, a2Buf, xBuf);

    // 6) eBuf = I + 0.5*a2Buf
    //    dBuf = I - 0.5*a2Buf
    mat_eye_f32(n, eBuf);
    mat_eye_f32(n, dBuf);
    float c = 0.5f;
    mat_add_f32(n, 1.0f, eBuf,  c,     a2Buf, eBuf);
    mat_add_f32(n, 1.0f, dBuf, -c,     a2Buf, dBuf);

    int p = 1; // used to flip sign in the loop

    // Prepare CMSIS structures
    arm_matrix_instance_f32 matX, matA2, matE, matD, matTmp;
    arm_mat_init_f32(&matX,  n, n, xBuf);
    arm_mat_init_f32(&matA2, n, n, a2Buf);
    arm_mat_init_f32(&matE,  n, n, eBuf);
    arm_mat_init_f32(&matD,  n, n, dBuf);
    arm_mat_init_f32(&matTmp, n, n, tmpBuf);

    // 7) Main loop: for k = 2 to Q
    for (int k = 2; k <= Q; k++)
    {
        // Update c
        c = c * (float)(Q - k + 1) / (float)(k * (2 * Q - k + 1));

        // xBuf = xBuf * a2Buf
        if (arm_mat_mult_f32(&matX, &matA2, &matTmp) != ARM_MATH_SUCCESS)
        {
            return NULL; // multiplication failed (unlikely for small n)
        }
        // Copy tmpBuf -> xBuf
        memcpy(xBuf, tmpBuf, (size_t)(n * n) * sizeof(float));

        // eBuf = eBuf + c * xBuf
        for (int i = 0; i < n * n; i++)
        {
            eBuf[i] += c * xBuf[i];
        }
        // dBuf = dBuf +/- c * xBuf
        float sign = p ? +1.0f : -1.0f;
        for (int i = 0; i < n * n; i++)
        {
            dBuf[i] += sign * c * xBuf[i];
        }
        p = !p;
    }

    // 8) Compute E = inv(D) * E
    // re-init because eBuf, dBuf changed
    arm_mat_init_f32(&matE, n, n, eBuf);
    arm_mat_init_f32(&matD, n, n, dBuf);

    arm_matrix_instance_f32 matDinv;
    arm_mat_init_f32(&matDinv, n, n, dInvBuf);

    // invert D
    if (arm_mat_inverse_f32(&matD, &matDinv) != ARM_MATH_SUCCESS)
    {
        return NULL; // matrix is singular?
    }
    // D = D_inv * E
    if (arm_mat_mult_f32(&matDinv, &matE, &matD) != ARM_MATH_SUCCESS)
    {
        return NULL;
    }
    // copy D -> E
    memcpy(eBuf, dBuf, (size_t)(n * n) * sizeof(float));

    // 9) Undo the scaling by repeated squaring: E^(2^s)
    arm_matrix_instance_f32 matEout;
    arm_mat_init_f32(&matEout, n, n, squareBuf);

    for (int k = 1; k <= s; k++)
    {
        // E = E * E
        arm_mat_init_f32(&matE, n, n, eBuf);
        if (arm_mat_mult_f32(&matE, &matE, &matEout) != ARM_MATH_SUCCESS)
        {
            return NULL;
        }
        memcpy(eBuf, squareBuf, (size_t)(n * n) * sizeof(float));
    }

    // -------------------------------------------------------------------------
    // At this point, eBuf holds the result exp(A).
    // In-place modification of A.pData which will hold exp(A)
    // -------------------------------------------------------------------------

    // Copy the final exponent data into resultData
    memcpy(A->pData, eBuf, (size_t)(n * n) * sizeof(float));

    // Return pointer to the static struct
    return 0;
}



/**
 * @brief Multiplies a matrix by a vector.
 *
 * This function performs the matrix–vector multiplication:
 *
 *      pDstVec[i] = Σ (pSrcA->pData[i * pSrcA->numCols + j] * pSrcVec[j])
 *                    for each row i and j = 0...pSrcA->numCols-1.
 *
 * @param pSrcA Pointer to an arm_matrix_instance_f32 structure containing the matrix.
 * @param pSrcVec Pointer to the input vector. The length must equal pSrcA->numCols.
 * @param pDstVec Pointer to the output vector. Must have space for pSrcA->numRows elements.
 * @return ARM_MATH_SUCCESS if the operation is successful; otherwise, ARM_MATH_ARGUMENT_ERROR if a NULL pointer is provided.
 */
arm_status mat_vec_mult_f32(const arm_matrix_instance_f32 *pSrcA,
                            const float32_t *pSrcVec,
                            float32_t *pDstVec)
{
    if ((pSrcA == NULL) || (pSrcVec == NULL) || (pDstVec == NULL))
    {
        return ARM_MATH_ARGUMENT_ERROR;
    }

    uint16_t rows = pSrcA->numRows;
    uint16_t cols = pSrcA->numCols;

    const float32_t *pMat = pSrcA->pData;

    for (uint16_t i = 0; i < rows; i++)
    {
        const float32_t *pRow = pMat;    // start of current row
        const float32_t *pVec = pSrcVec; // start of vector
        float32_t sum = 0.0f;

        uint16_t j = cols;

        // Unroll loop in chunks of 4
        while (j >= 4)
        {
            sum += pRow[0] * pVec[0];
            sum += pRow[1] * pVec[1];
            sum += pRow[2] * pVec[2];
            sum += pRow[3] * pVec[3];

            pRow += 4;
            pVec += 4;
            j -= 4;
        }

        // Handle remaining elements
        while (j > 0)
        {
            sum += (*pRow++) * (*pVec++);
            j--;
        }

        pDstVec[i] = sum;
        pMat += cols; // move to next row
    }

    return ARM_MATH_SUCCESS;
}


#ifdef __cplusplus
}
#endif

#endif // MATRIX_EXPM_ARM
