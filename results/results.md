## 1. Brushing trajectory to robot trajectory
- 1440 poses at 120 Hz, IK residual: 0.0007 mm, 0.00004 deg
- Time scaling needed to respect joint velocity and acceleration limits: **x3.05** (4 Hz human strokes exceed the arm's acceleration limits)
- Collision check (floor and self) along the trajectory: none
- Peak joint speed after scaling (rad/s): [0.94, 0.63, 0.79, 0.89, 1.67, 0.83] (limits [1.5, 1.5, 1.5, 2.5, 2.5, 2.5])

## 2. Replay accuracy (simulated arm with position servos)
- without gravity + velocity feedforward: position RMS 9.65 mm (max 13.67), orientation RMS 5.689 deg
- with gravity + velocity feedforward: position RMS 0.28 mm (max 0.62), orientation RMS 0.067 deg
- Repeatability over 5 replays with random disturbance torques: mean deviation 0.016 mm, 95th percentile 0.030 mm

## 3. Robot to motion-capture calibration (25 poses, frame pose and marker offset solved jointly)
- Frame rotation error: 0.018 deg, translation error: 0.421 mm
- Marker offset error: 0.093 mm, fit residual 0.212 mm

## 4. IMU calibration on the protocol: estimated vs true sensor effects
Gyro calibrated on the roll/pitch/yaw tests, accelerometer on the six static poses (gravity) and rotations; reference = calibrated motion capture.

| | Brush A true | Brush A estimated | Brush B true | Brush B estimated |
|---|---|---|---|---|
| Axis convention |  | correct |  | correct |
| Mount misalignment (deg) | 2.70 | 2.70 | 3.37 | 3.32 |
| Transport delay (ms) | 35 | 16 | 60 | 53 |
| Low-pass cutoff (Hz) | 20 | 11 | 12 | 12 |
| Gyro scale | [1.03, 0.98, 1.01] | [1.029, 0.978, 1.01] | [0.97, 1.04, 1.0] | [0.97, 1.04, 0.999] |
| Gyro bias (rad/s) | [0.02, -0.015, 0.01] | [0.02, -0.015, 0.01] | [-0.03, 0.02, 0.0] | [-0.03, 0.02, 0.0] |
| Accel scale | [0.99, 1.02, 1.01] | [0.989, 1.019, 1.01] | [0.99, 1.02, 1.01] | [0.989, 1.018, 1.007] |
| Accel bias (m/s2) | [0.15, -0.1, 0.2] | [0.146, -0.103, 0.197] | [0.15, -0.1, 0.2] | [0.155, -0.102, 0.201] |
| Gyro noise (rad/s) | 0.010 | 0.010 | 0.010 | 0.010 |

## 5. Validation on brushing (not used for calibration): two different IMUs, same robot motion
- before calibration: brush A vs B per-axis correlation [-0.49, -0.018, -0.311], RMS difference 0.711 rad/s
- after calibration: brush A vs B per-axis correlation [0.998, 0.998, 0.951], RMS difference 0.034 rad/s; brush A vs mocap reference RMS 0.055 rad/s

## 6. Validation protocol on the arm (static +-X/Y/Z and roll/pitch/yaw at 3 speeds)
| Test | IK residual (mm) | Collision-free | Time scaling | Peak joint speed (rad/s) |
|---|---|---|---|---|
| static +X up | 0.000 | yes | x1.00 | 0.00 |
| static -X up | 0.000 | yes | x1.00 | 0.00 |
| static +Y up | 0.001 | yes | x1.00 | 0.00 |
| static -Y up | 0.000 | yes | x1.00 | 0.00 |
| static +Z up | 0.000 | yes | x1.00 | 0.00 |
| static -Z up | 0.000 | yes | x1.00 | 0.00 |
| roll 0.5 Hz | 0.001 | yes | x2.65 | 0.69 |
| roll 1.0 Hz | 0.001 | yes | x2.08 | 1.76 |
| roll 2.0 Hz | 0.001 | yes | x4.17 | 1.76 |
| pitch 0.5 Hz | 0.001 | yes | x1.87 | 0.73 |
| pitch 1.0 Hz | 0.000 | yes | x1.10 | 2.50 |
| pitch 2.0 Hz | 0.000 | yes | x2.19 | 2.50 |
| yaw 0.5 Hz | 0.001 | yes | x1.94 | 0.71 |
| yaw 1.0 Hz | 0.001 | yes | x1.10 | 2.50 |
| yaw 2.0 Hz | 0.001 | yes | x2.19 | 2.50 |
