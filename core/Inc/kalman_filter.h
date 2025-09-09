#include "arm_math.h"
#include "matrix_expm_arm.h"
#include <string.h>
#include <stdlib.h>

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

// #define TEST_KALMAN_FILTER // for testing purposes only

#ifdef __cplusplus
extern "C" {
#endif

#define NX 4 // number of states
#define NY 4 // number of measurements
#define NU 4 // number of control inputs

#define KF_SAMPLING_PERIOD 0.02f // s

/*----------------------------------------------------------------------
  KalmanFilter structure definition

  Members:
    - A: System matrix (n×n).
    - C: Measurement matrix (m×n).  (This is now the “H” matrix in standard notation.)
    - state: State vector (n×1).
    - P0: Covariance matrix (n×n).
    - Q: Process noise covariance (n×n).
    - K: Kalman gain (n×m).
    - R: Measurement noise covariance (m×m).
----------------------------------------------------------------------*/

/* ================================ Typedefs =============================== */
typedef struct {
    arm_matrix_instance_f32 A;   // System matrix: n x n
    float32_t A_d[NX*NX];
    arm_matrix_instance_f32 B;   // Input matrix n x u
    float32_t B_d[NX*NU];
    arm_matrix_instance_f32 C;   // Measurement matrix: y x n  (i.e. H)
    float32_t C_d[NY*NX];
    arm_matrix_instance_f32 D;	  // Feedthrough matrix y x u
    float32_t D_d[NY*NU];
    float32_t state_predicted[NX];   // State vector prediction x_hat: n x 1
    float32_t state_corrected[NX];	  // State vector correction x_tilde: n x 1
    float32_t u[NU];
    arm_matrix_instance_f32 P_predicted;                // Predicted covariance matrix: n x n
    float32_t P_predicted_d[NX*NX];
    arm_matrix_instance_f32 P_corrected;				  // Corrected covariance matrix: n x n
    float32_t P_corrected_d[NX*NX];
    arm_matrix_instance_f32 Q;                 // Process noise covariance: n x n
    float32_t Q_d[NX*NX];
    arm_matrix_instance_f32 K;                 // Kalman gain: n x y
    float32_t K_d[NX*NY];
    arm_matrix_instance_f32 R;                 // Measurement noise covariance: y x y
    float32_t R_d[NY*NY];
} KalmanFilter;

/* ================================ Variables =============================== */

/* ================================ Functions =============================== */
//void state_estimation_task(void);
//
//void kalman_filter_init(void);



void compute_AB_discrete(
		const arm_matrix_instance_f32 *A_continuous,
        const arm_matrix_instance_f32 *B_continuous,
        arm_matrix_instance_f32       *A_discrete,
        arm_matrix_instance_f32       *B_discrete,
        const float32_t                      T);


/**
 * @brief Updates the state vector using the system model.
 *
 * Computes the state update according to:
 *
 *      xₖ₊₁ = A · xₖ
 *
 * A temporary buffer is allocated on the stack.
 *
 * @param filter Pointer to a KalmanFilter structure.
 */
void predict_state(KalmanFilter* filter);

/**
 * @brief Propagates the error covariance matrix one step ahead.
 *
 * Computes the prediction:
 *
 *      P⁻ = A · P⁺ · Aᵀ + Q
 *
 * Temporary matrices are allocated on the stack.
 *
 * @param filter Pointer to a KalmanFilter structure.
 */
void predict_covariance(KalmanFilter* filter);

/**
 * @brief Computes the Kalman gain without explicitly forming a transposed version of C.
 *
 * Given that C represents the measurement matrix (with dimensions m×n), the Kalman gain is computed as:
 *
 *      K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹
 *
 * The transpose of H is handled internally by swapping indices.
 *
 * @param filter Pointer to a KalmanFilter structure.
 */
void corrected_K(KalmanFilter* filter);

/**
 * @brief Updates the error covariance after the measurement update.
 *
 * Applies the correction:
 *
 *      P⁺ = (I - K · H) · P
 *
 * This function computes K · H and updates the covariance matrix accordingly.
 *
 * @param filter Pointer to a KalmanFilter structure.
 */
void corrected_covariance(KalmanFilter* filter);

/**
 * @brief Corrects the state vector using the measurement residual.
 *
 * Applies the correction:
 *
 *      x⁺ = x⁻ + K · (z - H · x)
 *
 * where H is provided by C.
 *
 * @param filter Pointer to a KalmanFilter structure.
 * @param measurement Pointer to the measurement vector.
 */
void corrected_state(KalmanFilter* filter, float32_t* measurement);

/**
 * @brief Executes the prediction of the state and covariance by calling
 * update state and update covariance
 *
 *
 * @param filter Pointer to a KalmanFilter structure.
 */
void prediction_step(KalmanFilter* filter);


/**
 * @brief Corrects the internal values of the Kalman Filter for a new measurement
 *
 * Calls the corrected_K, corrected_covariance and corrected_state
 * and updates K, P and x according to a new measurement
 *
 * @param filter Pointer to a KalmanFilter structure.
 * @param measurement Pointer to the measurement vector.
 */
void correction_step(KalmanFilter* filter, float32_t* measurement);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_FILTER_H */
