#include <stdio.h>
#include <string.h>
#include "arm_math.h"
#include "kalman_filter.h"

/*
 Minimal host-side example showing how to wire up the KalmanFilter
 with CMSIS-DSP. Replace printf parts with your HW-specific logging
 when running on STM32.
*/

static void kalman_init_example(KalmanFilter* kf)
{
    // Initialize matrix instances to use the preallocated buffers inside KalmanFilter
    arm_mat_init_f32(&kf->A,  NX, NX, kf->A_d);
    arm_mat_init_f32(&kf->B,  NX, NU, kf->B_d);
    arm_mat_init_f32(&kf->C,  NY, NX, kf->C_d);
    arm_mat_init_f32(&kf->D,  NY, NU, kf->D_d);
    arm_mat_init_f32(&kf->P_predicted, NX, NX, kf->P_predicted_d);
    arm_mat_init_f32(&kf->P_corrected, NX, NX, kf->P_corrected_d);
    arm_mat_init_f32(&kf->Q,  NX, NX, kf->Q_d);
    arm_mat_init_f32(&kf->K,  NX, NY, kf->K_d);
    arm_mat_init_f32(&kf->R,  NY, NY, kf->R_d);

    // Clear state and input
    memset(kf->state_predicted, 0, sizeof(kf->state_predicted));
    memset(kf->state_corrected, 0, sizeof(kf->state_corrected));
    memset(kf->u, 0, sizeof(kf->u));

    // Example continuous-time model (toy): x_dot = B*u, y = x
    // A_c = 0, B_c = I, C = I, D = 0
    for (uint32_t i = 0; i < NX * NX; ++i) {
        kf->A_d[i] = 0.0f;   // will be discretized from continuous A_c=0
        kf->Q_d[i] = 0.0f;   // process noise (tune as needed)
        kf->P_predicted_d[i] = 0.0f;
        kf->P_corrected_d[i] = 0.0f;
    }
    for (uint32_t i = 0; i < NX * NU; ++i) {
        kf->B_d[i] = 0.0f;
        kf->D_d[i] = 0.0f;
    }
    for (uint32_t i = 0; i < NY * NX; ++i) {
        kf->C_d[i] = 0.0f;
    }
    for (uint32_t i = 0; i < NX * NY; ++i) {
        kf->K_d[i] = 0.0f;
    }
    for (uint32_t i = 0; i < NY * NY; ++i) {
        kf->R_d[i] = 0.0f;   // measurement noise (tune as needed)
    }

    // Set identities where needed
    // C = I
    for (uint32_t i = 0; i < NX; ++i) {
        kf->C_d[i * NX + i] = 1.0f;
    }
    // B_c = I (store temporarily in B, then discretize)
    for (uint32_t i = 0; i < NX; ++i) {
        kf->B_d[i * NU + i] = 1.0f;
    }
    // Initial covariance P0 = I
    for (uint32_t i = 0; i < NX; ++i) {
        kf->P_corrected_d[i * NX + i] = 1.0f;
    }
    // Measurement noise R = 0.1 * I (example)
    for (uint32_t i = 0; i < NY; ++i) {
        kf->R_d[i * NY + i] = 1e-1f;
    }
    // Process noise Q = 1e-4 * I (example)
    for (uint32_t i = 0; i < NX; ++i) {
        kf->Q_d[i * NX + i] = 1e-4f;
    }

    // Discretize (A_c=0, B_c=I) using matrix exponential helper
    // After discretization: A_d = I, B_d = T * I
    // Use a local copy for continuous matrices to keep intent clear
    float32_t Ac_buf[NX * NX]; memset(Ac_buf, 0, sizeof(Ac_buf));
    float32_t Bc_buf[NX * NU]; memset(Bc_buf, 0, sizeof(Bc_buf));
    for (uint32_t i = 0; i < NX; ++i) {
        Bc_buf[i * NU + i] = 1.0f;
    }
    arm_matrix_instance_f32 Ac, Bc;
    arm_mat_init_f32(&Ac, NX, NX, Ac_buf);
    arm_mat_init_f32(&Bc, NX, NU, Bc_buf);

    compute_AB_discrete(&Ac, &Bc, &kf->A, &kf->B, KF_SAMPLING_PERIOD);
}

int main(void)
{
    KalmanFilter kf;
    kalman_init_example(&kf);

    // Example input and measurement
    float32_t u[NU] = {1.0f, 0.0f, 0.0f, 0.0f};
    float32_t y[NY] = {0};

    // Attach input pointer (optional pattern; here we copy into kf.u)
    memcpy(kf.u, u, sizeof(u));

    // Run a small predict-correct loop
    for (int k = 0; k < 5; ++k) {
        // Predict
        prediction_step(&kf);

        // Fake a measurement equal to the predicted state plus small noise
        for (uint32_t i = 0; i < NX; ++i) {
            y[i] = kf.state_predicted[i] + 0.05f; // pretend sensor reads true state + bias
        }

        // Correct
        correction_step(&kf, y);

        // Print result
        printf("k=%d\tstate=[", k);
        for (uint32_t i = 0; i < NX; ++i) {
            printf("% .4f%s", kf.state_corrected[i], (i + 1 == NX) ? "]\n" : ", ");
        }
    }

    return 0;
}
