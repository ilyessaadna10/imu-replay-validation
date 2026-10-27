#!/usr/bin/env python3
"""Run the whole validation pipeline and write results.md + figures."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bench import *

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
lines = []
log = lambda s="": (print(s), lines.append(s))

arm = Arm()
p_home, R_home = arm.tcp(arm.q_home)
workspace = (p_home, R_home)                                   # brushing: compact posture, least slowdown
protocol_ws = (p_home + np.array([0.0, 0.0, 0.15]), R_home)    # protocol: 15 cm higher so flipped poses clear the floor
cmd_tcp = lambda plan, tq: arm.tcp(np.array([np.interp(tq, plan.t, plan.q[:, j]) for j in range(6)]))

# 1. human brushing -> limit-checked robot trajectory ----------------------------------
t, pos, rot = brushing_motion(12.0)
plan = plan_trajectory(arm, t, pos, rot, workspace)
log("## 1. Brushing trajectory to robot trajectory")
log(f"- {len(t)} poses at 120 Hz, IK residual: {plan.ik_pos_err*1e3:.4f} mm, {np.degrees(plan.ik_rot_err):.5f} deg")
log(f"- Time scaling needed to respect joint velocity and acceleration limits: **x{plan.time_scale:.2f}** "
    f"(4 Hz human strokes exceed the arm's acceleration limits)")
log(f"- Collision check (floor and self) along the trajectory: {'none' if plan.collisions == 0 else plan.collisions}")
log(f"- Peak joint speed after scaling (rad/s): {np.round(plan.peak_qd, 2).tolist()} (limits {QD_MAX.tolist()})")

# 2. replay accuracy and repeatability ----------------------------------------------------
log("\n## 2. Replay accuracy (simulated arm with position servos)")
res = {}
for ff in (False, True):
    tr, P, Rs = replay(arm, plan, feedforward=ff)
    C = [cmd_tcp(plan, x) for x in tr]
    Pc = np.array([c[0] for c in C]); Rc = R.concatenate([c[1] for c in C])
    e_p = np.linalg.norm(P - Pc, axis=1) * 1e3; e_r = np.degrees((Rs * Rc.inv()).magnitude())
    res[ff] = (tr, P, Pc)
    log(f"- {'with' if ff else 'without'} gravity + velocity feedforward: position RMS {np.sqrt(np.mean(e_p**2)):.2f} mm "
        f"(max {e_p.max():.2f}), orientation RMS {np.sqrt(np.mean(e_r**2)):.3f} deg")
runs = [replay(arm, plan, joint_noise=0.3, seed=s)[1] for s in range(5)]
spread = np.linalg.norm(np.array(runs) - np.mean(runs, 0), axis=2) * 1e3
log(f"- Repeatability over 5 replays with random disturbance torques: mean deviation {spread.mean():.3f} mm, "
    f"95th percentile {np.percentile(spread, 95):.3f} mm")

# 3. robot <-> motion capture calibration --------------------------------------------------
mocap = Mocap()
Rwm, pwm, off, rms = calibrate_mocap(arm, mocap)
log("\n## 3. Robot to motion-capture calibration (25 poses, frame pose and marker offset solved jointly)")
log(f"- Frame rotation error: {np.degrees((Rwm * mocap.R_wm.inv()).magnitude()):.3f} deg, "
    f"translation error: {np.linalg.norm(pwm - mocap.p_wm)*1e3:.3f} mm")
log(f"- Marker offset error: {np.linalg.norm(off - mocap.marker_offset)*1e3:.3f} mm, fit residual {rms*1e3:.3f} mm")

# 4. IMU calibration on the protocol, validation on brushing ----------------------------
brushes = {
    "Brush A": IMUModel(),
    "Brush B": IMUModel(mount=R.from_matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) * R.from_euler("xyz", [-2, 1, 2.5], degrees=True),
                        gyro_bias=np.array([-0.03, 0.02, 0.0]), gyro_scale=np.array([0.97, 1.04, 1.0]),
                        lowpass_hz=12.0, delay_s=0.06, fs=50.0),
}

def mocap_reference(tr, P, Rs, seed):
    """What the bench really knows: calibrated mocap poses -> brush-head rates."""
    pm, rm = mocap.measure(P, Rs, seed=seed)
    Rs_mc = Rwm * rm
    P_mc = Rwm.apply(pm) + pwm - Rs_mc.apply(off)
    return body_rates(tr, P_mc, Rs_mc, smooth=13)

protocol = []
for k, (name, tp, pp, rr) in enumerate(protocol_motions()):
    pl = plan_trajectory(arm, tp, pp, rr, protocol_ws)
    tr_, P_, Rs_ = replay(arm, pl)
    w_, f_ = mocap_reference(tr_, P_, Rs_, seed=100 + k)
    protocol.append(dict(name=name, plan=pl, t_ref=tr_, P=P_, Rs=Rs_, w_ref=w_, f_ref=f_,
                         kind="static" if name.startswith("static") else "rotation"))

log("\n## 4. IMU calibration on the protocol: estimated vs true sensor effects")
log("Gyro calibrated on the roll/pitch/yaw tests, accelerometer on the six static poses (gravity) and rotations; "
    "reference = calibrated motion capture.\n")
log("| | Brush A true | Brush A estimated | Brush B true | Brush B estimated |")
log("|---|---|---|---|---|")
cals, rows = {}, {}
for name, imu in brushes.items():
    segs = []
    for k, sgm in enumerate(protocol):
        ti, g, a = imu.measure(sgm["t_ref"], sgm["P"], sgm["Rs"], seed=200 + k)
        segs.append({**sgm, "t_imu": ti, "gyro": g, "acc": a})
    cal = calibrate_imu_protocol(segs); cals[name] = cal
    conv_true = np.round(imu.mount.inv().as_matrix())
    mis_true = np.degrees(R.from_matrix(imu.mount.inv().as_matrix() @ conv_true.T).magnitude())
    f = lambda v: np.round(v, 3).tolist()
    for k, (tv, ev) in {
        "Axis convention": ("", "correct" if np.allclose(cal["convention"], conv_true) else "WRONG"),
        "Mount misalignment (deg)": (f"{mis_true:.2f}", f"{cal['misalign_deg']:.2f}"),
        "Transport delay (ms)": (f"{imu.delay_s*1e3:.0f}", f"{cal['transport_delay']*1e3:.0f}"),
        "Low-pass cutoff (Hz)": (f"{imu.lowpass_hz:.0f}", f"{cal['lowpass_hz']:.0f}"),
        "Gyro scale": (f(imu.gyro_scale), f(cal["gyro_scale"])),
        "Gyro bias (rad/s)": (f(imu.gyro_bias), f(cal["gyro_bias"])),
        "Accel scale": (f(imu.acc_scale), f(cal["acc_scale"])),
        "Accel bias (m/s2)": (f(imu.acc_bias), f(cal["acc_bias"])),
        "Gyro noise (rad/s)": (f"{imu.gyro_noise:.3f}", f"{cal['gyro_noise']:.3f}"),
    }.items():
        rows.setdefault(k, []).extend([tv, ev])
for k, v in rows.items():
    log(f"| {k} | " + " | ".join(str(x) for x in v) + " |")

# 5. validation on brushing (never seen during calibration) ------------------------------
tr, P, Rs = replay(arm, plan)
w_ref, _ = mocap_reference(tr, P, Rs, seed=7)
raw = {name: imu.measure(tr, P, Rs, seed=300 + i)[:2] for i, (name, imu) in enumerate(brushes.items())}
tg = np.arange(1.0, tr[-1] - 1.0, 0.01)
def on_grid(name, calibrated):
    ti, g = raw[name]; c = cals[name]
    sig = apply_calibration(c, g) if calibrated else g
    shift = c["delay"] if calibrated else 0.0
    return np.stack([np.interp(tg + shift, ti, sig[:, i]) for i in range(3)], 1)
ref = np.stack([np.interp(tg, tr, w_ref[:, i]) for i in range(3)], 1)
log("\n## 5. Validation on brushing (not used for calibration): two different IMUs, same robot motion")
for calibrated in (False, True):
    A, B = on_grid("Brush A", calibrated), on_grid("Brush B", calibrated)
    corr = [np.corrcoef(A[:, i], B[:, i])[0, 1] for i in range(3)]
    log(f"- {'after' if calibrated else 'before'} calibration: brush A vs B per-axis correlation {np.round(corr, 3).tolist()}, "
        f"RMS difference {np.sqrt(np.mean((A - B)**2)):.3f} rad/s"
        + (f"; brush A vs mocap reference RMS {np.sqrt(np.mean((A - ref)**2)):.3f} rad/s" if calibrated else ""))

# 6. protocol feasibility ----------------------------------------------------------------
log("\n## 6. Validation protocol on the arm (static +-X/Y/Z and roll/pitch/yaw at 3 speeds)")
log("| Test | IK residual (mm) | Collision-free | Time scaling | Peak joint speed (rad/s) |")
log("|---|---|---|---|---|")
for name, tp, pp, rr in protocol_motions():
    pl = plan_trajectory(arm, tp, pp, rr, protocol_ws)
    log(f"| {name} | {pl.ik_pos_err*1e3:.3f} | {'yes' if pl.collisions == 0 else 'NO'} | x{pl.time_scale:.2f} | {pl.peak_qd.max():.2f} |")

# figures --------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for ff, ls in ((False, "--"), (True, "-")):
    tr_, P_, Pc_ = res[ff]
    ax[0].plot(tr_, (P_ - Pc_)[:, 0] * 1e3, ls, label=f"x error, {'with' if ff else 'no'} feedforward")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("tracking error (mm)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("Replay accuracy")
ti, g = raw["Brush B"]; sel = (ti > 10) & (ti < 14)
ax[1].plot(ti[sel], g[sel, 0], color="0.6", label="raw IMU x (wrong axis, biased, late)")
gc = apply_calibration(cals["Brush B"], g)
ax[1].plot(ti[sel] - cals["Brush B"]["delay"], gc[sel, 0], label="calibrated")
selr = (tr > 10) & (tr < 14)
ax[1].plot(tr[selr], w_ref[selr, 0], "k:", label="reference from mocap")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("angular rate (rad/s)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("Brush B gyro: raw vs calibrated")
fig.tight_layout(); fig.savefig(OUT / "replay_and_imu.png", dpi=110)

(OUT / "results.md").write_text("\n".join(lines) + "\n")
print("\nwritten:", OUT / "results.md", OUT / "replay_and_imu.png")
