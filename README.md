# Kalman Filter (CMSIS-DSP, STM32-ready)

A lightweight Kalman Filter implementation built on top of CMSIS-DSP `arm_math.h`. It provides:

- State prediction and covariance propagation
- Measurement update (state and covariance), with Joseph-form covariance correction
- Continuous-to-discrete conversion for (A, B) via matrix exponentiation

The core code lives under `core/Inc` and `core/Src`.

- `core/Inc/kalman_filter.h` — public API and `KalmanFilter` struct
- `core/Src/kalman_filter.c` — prediction/correction implementation
- `core/Inc/matrix_expm_arm.h`, `core/Src/matrix_expm_arm.c` — matrix exponential used for discretization


## Prerequisites

- CMSIS-DSP library and headers (provides `arm_math.h`)
- A C/C++ toolchain (e.g., arm-none-eabi-gcc for STM32 or host gcc/clang if CMSIS-DSP is available)


## Configuration

Static dimensions are defined in `core/Inc/kalman_filter.h`:

- `#define NX 4` — number of states
- `#define NY 4` — number of measurements
- `#define NU 4` — number of control inputs
- `#define KF_SAMPLING_PERIOD 0.02f` — sample time for discretization (s)

Adjust these to your system before building.


## API overview

Key functions (see `kalman_filter.h` for docs):

- `predict_state(KalmanFilter*)`
- `predict_covariance(KalmanFilter*)`
- `prediction_step(KalmanFilter*)` — convenience wrapper
- `corrected_K(KalmanFilter*)`
- `corrected_state(KalmanFilter*, float32_t* measurement)`
- `corrected_covariance(KalmanFilter*)`
- `correction_step(KalmanFilter*, float32_t* measurement)` — convenience wrapper
- `compute_AB_discrete(Ac, Bc, Ad, Bd, T)` — discretize continuous-time model (matrix exponential)


## Quick start (example)

A minimal usage example is provided in `examples/kalman_example.c`. It:

1) Initializes a `KalmanFilter` instance (matrices and buffers)
2) Builds a trivial continuous-time model (A=0, B=I) and discretizes it
3) Runs a short predict/correct loop with a dummy measurement

You can adapt the same pattern to your STM32 project (e.g., inside your control loop):

- Call `prediction_step(&kf);`
- Read sensors into `y[NY]`
- Call `correction_step(&kf, y);`


## Building

The exact steps depend on your environment:

- STM32CubeIDE/STM32 projects: add the `core/Inc` and `core/Src` files to your project, ensure CMSIS-DSP is included and `ARM_MATH_CMx` is defined for your core (e.g., `ARM_MATH_CM4`). Then add `examples/kalman_example.c` to a test app or use your `main.c`.
- Host build (optional): if you have CMSIS-DSP for your host, compile `core/Src/*.c` and `examples/kalman_example.c` and link with CMSIS-DSP.

Include path needs `core/Inc`. Link path must include CMSIS-DSP and any required startup flags.


## Notes

- The implementation uses Joseph-form for covariance correction for improved numerical stability.
- Matrices are stored in row-major order as expected by CMSIS-DSP.
- Ensure `R` (measurement noise) and `Q` (process noise) are tuned for your system.


## License

This project is provided as-is. Integrate and modify to fit your application.
