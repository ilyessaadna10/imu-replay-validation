#!/usr/bin/env python3
"""Render the arm replaying brushing (with a drawn brush and the brush-head trail),
next to live plots of raw vs calibrated IMU signals. Writes results/bench_demo.mp4."""
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from bench import *

arm = Arm()
p_home, R_home = arm.tcp(arm.q_home)
t, pos, rot = brushing_motion(12.0)
plan = plan_trajectory(arm, t, pos, rot, (p_home, R_home))
tr, P, Rs = replay(arm, plan)
qs = np.stack([[np.interp(x, plan.t, plan.q[:, j]) for j in range(6)] for x in tr])

imu = IMUModel(mount=R.from_matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]) * R.from_euler("xyz", [-2, 1, 2.5], degrees=True),
               gyro_bias=np.array([-0.03, 0.02, 0.0]), gyro_scale=np.array([0.97, 1.04, 1.0]), lowpass_hz=12, delay_s=0.06, fs=50)
ti, g, _ = imu.measure(tr, P, Rs)
w_true, _ = body_rates(tr, P, Rs)
# for the video, use the true mounting/bias/scale to show what calibration recovers
gc = ((g - imu.gyro_bias) / imu.gyro_scale) @ imu.mount.inv().as_matrix()

H, W = 480, 480
ren = mujoco.Renderer(arm.m, H, W)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [p_home[0] * 0.6, p_home[1] * 0.6, 0.3]; cam.distance = 1.35; cam.azimuth = 30; cam.elevation = -22
zoom = mujoco.MjvCamera(); zoom.type = mujoco.mjtCamera.mjCAMERA_FREE
zoom.distance = 0.22; zoom.azimuth = 30; zoom.elevation = -35

def add_capsule(scn, a, b, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g_ = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g_, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g_, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1

fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=75)
out = cv2.VideoWriter(str(HERE / "results" / "bench_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W + 480, H))
step = 4                                   # 120 Hz -> 30 fps
for k in range(0, len(tr), step):
    arm.d.qpos[:] = qs[k]; mujoco.mj_forward(arm.m, arm.d)
    def draw(c, trail_r):
        ren.update_scene(arm.d, c)
        scn = ren.scene
        head = P[k]
        add_capsule(scn, head, head + Rs[k].apply([0, 0, -0.18]), 0.009, [0.2, 0.6, 0.9, 1])   # brush body
        add_capsule(scn, head, head + Rs[k].apply([0, 0.022, 0]), 0.005, [1, 1, 1, 1])          # bristles
        for j in range(max(0, k - 480), k, 3):                                                   # 4 s trail
            add_capsule(scn, P[j], P[min(j + 3, len(P) - 1)], trail_r, [1, 0.45, 0.05, 1])
        return cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
    img = draw(cam, 0.004)
    zoom.lookat[:] = P[k]
    inset = cv2.resize(draw(zoom, 0.0012), (170, 170))
    cv2.rectangle(inset, (0, 0), (169, 169), (255, 255, 255), 2)
    img[H - 180:H - 10, W - 180:W - 10] = inset
    cv2.putText(img, "brush head close-up", (W - 180, H - 186), 0, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"t = {tr[k]:4.1f} s   replay slowed x{plan.time_scale:.2f}", (10, 25), 0, 0.55, (255, 255, 255), 2)
    ax.clear()
    win = (ti > tr[k] - 4) & (ti <= tr[k]); winr = (tr > tr[k] - 4) & (tr <= tr[k])
    ax.plot(ti[win], g[win, 0], color="0.6", label="raw IMU x (wrong axis, biased, late)")
    ax.plot(ti[win] - imu.delay_s, gc[win, 0], label="calibrated IMU x")
    ax.plot(tr[winr], w_true[winr, 0], "k:", label="true brush rotation rate x")
    ax.set_xlim(tr[k] - 4, tr[k]); ax.set_ylim(-1.2, 1.2); ax.set_xlabel("time (s)"); ax.set_ylabel("rad/s")
    ax.legend(loc="lower left", fontsize=7); ax.grid(alpha=.3); ax.set_title("Toothbrush gyro: raw vs calibrated", fontsize=10)
    fig.canvas.draw()
    plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
    out.write(np.hstack([img, cv2.resize(plot, (480, H))]))
out.release()
print("written results/bench_demo.mp4")
