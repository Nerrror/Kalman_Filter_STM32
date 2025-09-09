/**
 *	module: kalman_filter.c
 *			This module computes the Kalman-Filter for a given System
 *
 *	creator: Alex Luce
 *
 */
#include "kalman_filter.h"
#include <string.h>
#include <stdlib.h>

#include "arm_math.h"
#include "matrix_expm_arm.h"

#ifndef KALMAN_FILTER
#define KALMAN_FILTER




#define JOSEPHSFORM // more robust but more computational effort for the correction of the covariance matrix P


#ifdef __cplusplus
extern "C" {
#endif

/* ================================ Variables =============================== */

/* ================================ Routine Functions ================================ */

/* ================================ Init Functions ========================================== */




/**
 * @brief Discretize the system matrix via matrix exponentiation
 *       M = exp([[A, B]; [0, 0]] * T)
 * @param A_continuous Continuous-time system matrix (n x n)
 * @param B_continuous Continuous-time input matrix (n x m)
 * @param A_discrete Discrete-time system matrix (n x n)
 * @param B_discrete Discrete-time input matrix (n x m)
 * @param T Sampling time
 * @note It is advised to discretize the system during initialization only due to
 *       the computational effort of the matrix exponential.
 */
void compute_AB_discrete(const arm_matrix_instance_f32 *A_continuous,
                         const arm_matrix_instance_f32 *B_continuous,
                         arm_matrix_instance_f32       *A_discrete,
                         arm_matrix_instance_f32       *B_discrete,
                         float32_t                      T)
{
    /* --- basic dimension checks ------------------------------------ */
    const uint16_t n = A_continuous->numRows;          /* #states   */
    const uint16_t m = B_continuous->numCols;          /* #inputs   */

    /* --- build augmented matrix M = [[A,B];[0,0]] ------------------ */
    const uint16_t N = n + m;                          /* total size        */

    float32_t M_d[N * N]; memset(M_d, 0, sizeof M_d);
    arm_matrix_instance_f32 M;
    arm_mat_init_f32(&M, N, N, M_d);

    /* copy A into upper-left block ---------------------------------- */
    for (uint16_t r = 0; r < n; ++r)
        memcpy(&M_d[r * N],                       /* row start in M     */
               &A_continuous->pData[r * n],       /* row start in A     */
               n * sizeof(float32_t));

    /* copy B into upper-right block --------------------------------- */
    for (uint16_t r = 0; r < n; ++r)
        memcpy(&M_d[n + r * N],                   /* col-offset = n     */
               &B_continuous->pData[r * m],
               m * sizeof(float32_t));

    /* --- scale by sampling time ------------------------------------ */
    arm_mat_scale_f32(&M, T, &M);

    /* --- exponential ------------------------------------------------*/
    int success = matrix_expm_cmsis_f32(&M);
//    ARM_MATH_ASSERT(expMT != NULL);

    /* --- extract A_d (upper-left n×n) ------------------------------ */
    for (uint16_t r = 0; r < n; ++r)
        memcpy(&A_discrete->pData[r * n],
               &M.pData[r * N],
               n * sizeof(float32_t));

    /* --- extract B_d (upper-right n×m) ----------------------------- */
    for (uint16_t r = 0; r < n; ++r)
        memcpy(&B_discrete->pData[r * m],
               &M.pData[r * N + n],
               m * sizeof(float32_t));

    /* if matrix_expm_cmsis_f32() allocates internally, free here */
}



/*======================================================================
  update_state

  Updates the state vector using the system model:

      x_k+1 = A * x_k + B * u_k

  A temporary buffer is allocated on the stack.
======================================================================*/
void predict_state(KalmanFilter* filter) {
    if (!filter || !&filter->A || !&filter->B || !filter->state_corrected || !filter->u)
        return;

//    uint16_t n = filter->A->numRows;
//    if (n > MAX_DIM)
//        return;

    float32_t Ax[NX];  // temporary buffer for new state
    float32_t Bu[NX];

    if (mat_vec_mult_f32(&filter->A, filter->state_corrected, Ax) != ARM_MATH_SUCCESS)
        return;

    if (mat_vec_mult_f32(&filter->B, filter->u, Bu) != ARM_MATH_SUCCESS)
        return;

    arm_add_f32(Ax, Bu, filter->state_predicted, NX);

}


/*======================================================================
  update_covariance

  Propagates the error covariance matrix one step ahead:

      P_predicted = A · P_corrected · A^T + G · Q · G^T

  We assume that no process noise is present.
  Temporary matrices are allocated on the stack.
======================================================================*/
void predict_covariance(KalmanFilter* filter) {
    if (!filter || !&filter->A || !&filter->P_predicted || !&filter->P_corrected || !&filter->Q)
        return;

    float32_t APA_T_data[NX * NX], AP_data[NX * NX], A_T_data[NX * NX];
	arm_matrix_instance_f32 APA_T, AP, A_T;

	arm_mat_init_f32(&APA_T, NX, NX, APA_T_data);
    arm_mat_init_f32(&AP, NX, NX, AP_data);
    arm_mat_init_f32(&A_T, NX, NX, A_T_data);

    if (arm_mat_trans_f32(&filter->A, &A_T) != ARM_MATH_SUCCESS)
        return;
    if (arm_mat_mult_f32(&filter->A, &filter->P_corrected, &AP) != ARM_MATH_SUCCESS)
        return;
    if (arm_mat_mult_f32(&AP, &A_T, &APA_T) != ARM_MATH_SUCCESS)
        return;
    if (arm_mat_add_f32(&APA_T, &filter->Q, &filter->P_predicted) != ARM_MATH_SUCCESS)
            return;

}


/*======================================================================
  corrected_K

  the standard Kalman gain is

      K = Pi · C^T · (H · Pi · C^T + R)^-1

  The result is stored in filter->K (n x m).
======================================================================*/
void corrected_K(KalmanFilter* filter) {
    if (!filter || !&filter->P_predicted || !&filter->C || !&filter->R || !&filter->K)
        return;


    /* Compute C^T (n x m) */
    float32_t Ctr_data[NX * NY];
    arm_matrix_instance_f32 Ctr;
    arm_mat_init_f32(&Ctr, NX, NY, Ctr_data);
    if (arm_mat_trans_f32(&filter->C, &Ctr) != ARM_MATH_SUCCESS)
        return;

    float32_t PCtr_data[NX * NY];
    arm_matrix_instance_f32 PCtr;
    arm_mat_init_f32(&PCtr, NX, NY, PCtr_data);

    if (arm_mat_mult_f32(&filter->P_predicted, &Ctr, &PCtr) != ARM_MATH_SUCCESS)
        return;

    /* S = C^T * Temp1 (m x m) */
    float32_t S_data[NY * NY];
    arm_matrix_instance_f32 S;
    arm_mat_init_f32(&S, NY, NY, S_data);
    if (arm_mat_mult_f32(&filter->C, &PCtr, &S) != ARM_MATH_SUCCESS)
        return;

    /* Add measurement noise covariance: S = S + R */
//    for (uint16_t i = 0; i < m * m; i++) {
//        S_data[i] += filter->R[i];
//    }
    if (arm_mat_add_f32(&S, &filter->R, &S)!= ARM_MATH_SUCCESS)
        return;

    /* Compute the inverse of S: S_inv = inv(S) */
    float32_t S_inv_data[NY * NY];
    arm_matrix_instance_f32 S_inv;
    arm_mat_init_f32(&S_inv, NY, NY, S_inv_data);

    if (arm_mat_inverse_f32(&S, &S_inv) != ARM_MATH_SUCCESS)
        return;

	/* Compute the Kalman gain: K = Temp1 * S_inv (n x m) */
	if (arm_mat_mult_f32(&PCtr, &S_inv, &filter->K) != ARM_MATH_SUCCESS)
		return;

}

/*----------------------------------------------------------------------
  corrected_state

  Corrects the state vector using the measurement y residual:

      state_corrected = state_predicted + K * (y - C * state_predicted)

  Temporary buffers are statically allocated.
----------------------------------------------------------------------*/
void corrected_state(KalmanFilter* filter, float32_t* measurement)
{
    if (!filter || !&filter->C || !&filter->D || !&filter->K || !filter->u || !measurement)
        return;


    /* 1) Predicted measurement: y_predicted = C * x_corrected */
    float32_t Delta_y[NY];
    float32_t temp_NX[NX];
    float32_t temp_NY[NY];

    if (mat_vec_mult_f32(&filter->C, filter->state_predicted, Delta_y) != ARM_MATH_SUCCESS)
        return;

    if (mat_vec_mult_f32(&filter->D, filter->u, temp_NY) != ARM_MATH_SUCCESS)
            return;

    arm_add_f32(Delta_y, temp_NY, Delta_y, NY);

    arm_scale_f32(Delta_y, -1.0f, Delta_y, NY);

    /* 2) Measurement residual: delta_y = measurement - y_predicted */

    arm_add_f32(Delta_y, measurement, Delta_y, NY);

    /* 3) State correction: delta_x = K * delta_y */
    if (mat_vec_mult_f32(&filter->K, Delta_y, temp_NX) != ARM_MATH_SUCCESS)
        return;

    /* 4) x_corrected = x_predicted + delta_x */
    arm_add_f32(filter->state_predicted, temp_NX, filter->state_corrected, NX);

}



/*----------------------------------------------------------------------
  corrected_covariance

  Corrects the error covariance after the measurement update in
  Josephs-Form:

      P_corrected =  (1 - K@C)@P_predicted@(1 - K@C)^T + K@R@K^T

  All temporary arrays are statically allocated.
----------------------------------------------------------------------*/
void corrected_covariance(KalmanFilter* filter) {
    if (!filter || !&filter->P_predicted || !&filter->C || !&filter->K)
        return;


    /* Compute Temp = C^T * P (m x n) */
    float32_t IminusKC_data[NX * NX];
    arm_matrix_instance_f32 IminusKC;
    arm_mat_init_f32(&IminusKC, NX, NX, IminusKC_data);
    if (arm_mat_mult_f32(&filter->K, &filter->C, &IminusKC) != ARM_MATH_SUCCESS)
        return;

    if (arm_mat_scale_f32(&IminusKC, -1.0f, &IminusKC) != ARM_MATH_SUCCESS)
            return;

    float32_t unity_data[NX*NX];
    mat_eye_f32(NX, unity_data);
    arm_matrix_instance_f32 unity;
    arm_mat_init_f32(&unity, NX, NX, unity_data);

    if (arm_mat_add_f32(&unity, &IminusKC, &IminusKC) != ARM_MATH_SUCCESS)
                    return;

#ifdef JOSEPHSFORM

    float32_t IminusKC_tr_data[NX * NX];
	arm_matrix_instance_f32 IminusKC_tr;
	arm_mat_init_f32(&IminusKC_tr, NX, NX, IminusKC_tr_data);
	if (arm_mat_trans_f32(&IminusKC, &IminusKC_tr) != ARM_MATH_SUCCESS)
	        return;

    float32_t Ktr_data[NY * NX];
    arm_matrix_instance_f32 Ktr;
    arm_mat_init_f32(&Ktr, NY, NX, Ktr_data);
    if (arm_mat_trans_f32(&filter->K, &Ktr) != ARM_MATH_SUCCESS)
            return;

    float32_t KR_data[NX * NY];
	arm_matrix_instance_f32 KR;
	arm_mat_init_f32(&KR, NX, NY, KR_data);
	if (arm_mat_mult_f32(&filter->K, &filter->R, &KR) != ARM_MATH_SUCCESS)
			return;

	float32_t KRKtr_data[NX * NX];
	arm_matrix_instance_f32 KRKtr;
	arm_mat_init_f32(&KRKtr, NX, NX, KRKtr_data);
	if (arm_mat_mult_f32(&KR, &Ktr, &KRKtr) != ARM_MATH_SUCCESS)
			return;

    float32_t P_corrected_data[NX * NX];
    arm_matrix_instance_f32 P_corrected;
    arm_mat_init_f32(&P_corrected, NX, NX, P_corrected_data);

    if (arm_mat_mult_f32(&IminusKC, &filter->P_predicted, &P_corrected) != ARM_MATH_SUCCESS)
        return;
//    I'm trying to avoid initializing another matrix here, thats why I save the intermediate
//    result in IminusKC because this matrix is not needed anymore
    if (arm_mat_mult_f32(&P_corrected, &IminusKC_tr, &IminusKC) != ARM_MATH_SUCCESS)
            return;

    if (arm_mat_add_f32(&IminusKC, &KRKtr, &filter->P_corrected) != ARM_MATH_SUCCESS)
        return;

#else

    if (arm_mat_mult_f32(&IminusKC, filter->P_predicted, filter->P_corrected) != ARM_MATH_SUCCESS)
            return;

#endif


}



void prediction_step(KalmanFilter* filter){
	predict_state(filter);
	predict_covariance(filter);

}

void correction_step(KalmanFilter* filter, float32_t* measurement){
	corrected_K(filter);
	corrected_state(filter, measurement);
	corrected_covariance(filter);
}



#ifdef __cplusplus
}
#endif

#endif // KALMAN_FILTER
