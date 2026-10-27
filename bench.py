"""Robot-assisted IMU validation bench, in simulation.

Pipeline (mirrors a real test bench: 6-DOF arm + motion capture + toothbrush IMU):
  1. human brushing poses at 120 Hz (position + quaternion)
  2. -> continuous, limit-checked joint trajectory for a 6-DOF arm (TCP = brush head)
  3. -> replay on the simulated arm with its position controllers
  4. -> "OptiTrack": noisy marker poses in a separate mocap frame, at 120 Hz
  5. -> "IMU": gyro + accelerometer in the brush, with axis convention, misalignment,
        bias, scale, noise, low-pass filtering, delay and its own sample rate
  6. -> calibration and protocol analysis that estimate and remove those effects
"""
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).parent
MODEL = HERE / "mujoco_menagerie" / "universal_robots_ur5e" / "scene.xml"
FS_MOCAP = 120.0
G = np.array([0.0, 0.0, -9.81])

# Conservative replay limits (well inside the UR5e datasheet values)
QD_MAX = np.array([1.5, 1.5, 1.5, 2.5, 2.5, 2.5])      # rad/s
QDD_MAX = np.array([8, 8, 8, 15, 15, 15.0])            # rad/s^2
BRUSH_HEAD = np.array([0.0, 0.0, 0.18])                 # TCP offset from the flange, m


# ----------------------------------------------------------------------------- motions
def brushing_motion(duration=12.0, fs=FS_MOCAP, seed=0):
    """Synthetic human brushing: 4 Hz back-and-forth strokes while slowly sweeping
    along the dental arch, with surface changes (brush rolls) and hand tremor.
    Returns t, positions (N,3) in a local 'mouth' frame, and orientations (Rotation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    sweep = 0.5 * (1 - np.cos(np.pi * t / duration))           # 0 -> 1 across the arch
    arch = np.pi * (sweep - 0.5) * 0.9                          # -81 .. +81 deg
    stroke = 0.008 * np.sin(2 * np.pi * 4.0 * t)                # +-8 mm at 4 Hz
    pos = np.stack([0.03 * np.sin(arch) + stroke * np.cos(arch),
                    0.025 * (1 - np.cos(arch)),
                    0.004 * np.sin(2 * np.pi * 0.7 * t)], 1)
    pos += 0.0004 * rng.standard_normal(pos.shape).cumsum(0) / np.sqrt(len(t))  # slow tremor drift
    roll = np.deg2rad(35) * np.tanh(3 * np.sin(2 * np.pi * t / duration * 2))   # outer <-> inner surface
    yaw = arch
    pitch = np.deg2rad(8) * np.sin(2 * np.pi * 4.0 * t + 0.5)                     # wrist rocking
    rot = R.from_euler("zyx", np.stack([yaw, pitch, roll], 1))
    return t, pos, rot


def protocol_motions(fs=FS_MOCAP):
    """The validation protocol: static +-X/+-Y/+-Z orientations, then roll/pitch/yaw
    sinusoids at three speeds. Returns a list of (name, t, pos, rot)."""
    out, hold = [], 3.0
    static = {"+X up": [0, -90, 0], "-X up": [0, 90, 0], "+Y up": [90, 0, 0],
              "-Y up": [-90, 0, 0], "+Z up": [0, 0, 0], "-Z up": [180, 0, 0]}
    t = np.arange(0, hold, 1 / fs)
    for name, eul in static.items():
        out.append((f"static {name}", t, np.zeros((len(t), 3)),
                    R.from_euler("xyz", np.tile(np.deg2rad(eul), (len(t), 1)))))
    for axis in "xyz":
        for f_hz in (0.5, 1.0, 2.0):
            t = np.arange(0, 4.0, 1 / fs)
            ang = np.deg2rad(25) * np.sin(2 * np.pi * f_hz * t) * np.minimum(1, t / 0.5)
            eul = np.zeros((len(t), 3)); eul[:, "xyz".index(axis)] = ang
            out.append((f"{'roll pitch yaw'.split()['xyz'.index(axis)]} {f_hz} Hz", t,
                        np.zeros((len(t), 3)), R.from_euler("xyz", eul)))
    return out


# ----------------------------------------------------------------------------- robot
class Arm:
    def __init__(self, model_path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(str(model_path))
        self.d = mujoco.MjData(self.m)
        self.flange = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.q_home = self.m.key_qpos[0].copy()
        self.lo, self.hi = self.m.jnt_range[:, 0], self.m.jnt_range[:, 1]

    def tcp(self, q):
        """Brush-head pose (position, Rotation) for joint vector q."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        Rf = self.d.site_xmat[self.flange].reshape(3, 3)
        return self.d.site_xpos[self.flange] + Rf @ BRUSH_HEAD, R.from_matrix(Rf)

    def in_collision(self, q):
        """True if configuration q touches the floor or itself (checked with MuJoCo contacts)."""
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        return self.d.ncon > 0

    def _jac(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        jp, jr = np.zeros((3, 6)), np.zeros((3, 6))
        Rf = self.d.site_xmat[self.flange].reshape(3, 3)
        mujoco.mj_jac(self.m, self.d, jp, jr, self.d.site_xpos[self.flange] + Rf @ BRUSH_HEAD,
                      self.m.site_bodyid[self.flange])
        return np.vstack([jp, jr])

    def ik(self, p, rot, q0, iters=50, tol=1e-6):
        q = q0.copy()
        for _ in range(iters):
            pc, rc = self.tcp(q)
            err = np.concatenate([p - pc, (rot * rc.inv()).as_rotvec()])
            if np.linalg.norm(err) < tol:
                break
            J = self._jac(q)
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
        return q, np.linalg.norm(err[:3]), np.linalg.norm(err[3:])


@dataclass
class Plan:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    time_scale: float
    ik_pos_err: float
    ik_rot_err: float
    peak_qd: np.ndarray = field(default=None)
    peak_qdd: np.ndarray = field(default=None)
    collisions: int = 0


def plan_trajectory(arm, t, pos, rot, workspace):
    """Human poses -> continuous joint trajectory that respects position, velocity and
    acceleration limits. workspace = (p0, R0) places the motion in front of the robot.
    If limits would be exceeded, the whole trajectory is slowed down uniformly
    (time scaling keeps the path identical) and the factor is reported."""
    p0, R0 = workspace
    P = p0 + R0.apply(pos)
    Rot = R0 * rot
    qs, perr, rerr = [], [], []
    rng = np.random.default_rng(0)
    seeds = [arm.q_home + d for d in ([0] * 6, [0, 0, 0, 0, 0, np.pi], [0, 0, 0, np.pi, 0, 0],
                                      [0, 0, 0, 0, np.pi, 0], [0, 0, 0, np.pi / 2, np.pi / 2, 0])]
    seeds += [arm.q_home + rng.uniform(-np.pi, np.pi, 6) for _ in range(80)]
    sols = [arm.ik(P[0], Rot[0], np.asarray(sd, float), iters=300) for sd in seeds]
    # prefer collision-free solutions that stay close to the home posture
    score = lambda r: (r[1] > 1e-4 or r[2] > 1e-3, arm.in_collision(r[0]), np.linalg.norm(r[0] - arm.q_home))
    q = min(sols, key=score)[0]
    for k in range(len(t)):
        q, ep, er = arm.ik(P[k], Rot[k], q)
        qs.append(q.copy()); perr.append(ep); rerr.append(er)
    Q = np.unwrap(np.array(qs), axis=0)
    collisions = int(sum(arm.in_collision(qq) for qq in Q[::6]))
    dt = np.diff(t).mean()
    qd = np.gradient(Q, dt, axis=0); qdd = np.gradient(qd, dt, axis=0)
    scale = max(1.0, np.max(np.abs(qd).max(0) / QD_MAX), np.sqrt(np.max(np.abs(qdd).max(0) / QDD_MAX)))
    return Plan(t * scale, Q, qd / scale, scale, max(perr), max(rerr),
                np.abs(qd).max(0) / scale, np.abs(qdd).max(0) / scale**2, collisions)


def replay(arm, plan, fs_out=FS_MOCAP, joint_noise=0.0, seed=0, feedforward=True):
    """Track the planned joints with the arm's position controllers (physics on).
    feedforward adds the gravity and Coriolis torques of the model (as real arm
    controllers do), plus a velocity reference. Returns time and the true brush-head
    poses at fs_out."""
    rng = np.random.default_rng(seed)
    m, d = arm.m, arm.d
    mujoco.mj_resetData(m, d); d.qpos[:] = plan.q[0]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    t_out = np.arange(0, plan.t[-1], 1 / fs_out)
    P, Rs, tt, k = [], [], 0.0, 0
    while k < len(t_out):
        q_ref = np.array([np.interp(tt, plan.t, plan.q[:, j]) for j in range(6)])
        qd_ref = np.array([np.interp(tt, plan.t, plan.qd[:, j]) for j in range(6)])
        kp, kv = m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 2]
        # position servo: ctrl is a position setpoint; shift it to also track the velocity reference
        d.ctrl[:] = q_ref + (kv / kp) * qd_ref if feedforward else q_ref
        d.qfrc_applied[:] = (d.qfrc_bias if feedforward else 0) + joint_noise * rng.standard_normal(6)
        mujoco.mj_step(m, d); tt += m.opt.timestep
        if tt >= t_out[k]:
            p, r = arm.tcp(d.qpos.copy()); P.append(p); Rs.append(r.as_quat()); k += 1
    return t_out, np.array(P), R.from_quat(Rs)


# ----------------------------------------------------------------------------- sensors
@dataclass
class Mocap:
    """OptiTrack-like system: its own world frame, a rigid marker body on the brush."""
    R_wm: R = field(default_factory=lambda: R.from_euler("xyz", [3, -2, 95], degrees=True))
    p_wm: np.ndarray = field(default_factory=lambda: np.array([1.2, -0.4, 0.05]))
    marker_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.015, -0.12]))
    pos_noise: float = 0.0002       # 0.2 mm
    rot_noise_deg: float = 0.1

    def measure(self, P, Rs, seed=1):
        rng = np.random.default_rng(seed)
        pm = P + Rs.apply(self.marker_offset)                       # marker in robot frame
        p_meas = self.R_wm.inv().apply(pm - self.p_wm) + self.pos_noise * rng.standard_normal(P.shape)
        r_meas = R.from_rotvec(np.deg2rad(self.rot_noise_deg) * rng.standard_normal(P.shape)) * self.R_wm.inv() * Rs
        return p_meas, r_meas


@dataclass
class IMUModel:
    """Toothbrush IMU with the usual error sources."""
    mount: R = field(default_factory=lambda: R.from_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) *
                     R.from_euler("xyz", [1.5, -2.0, 1.0], degrees=True))   # axis swap + misalignment
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.02, -0.015, 0.01]))      # rad/s
    gyro_scale: np.ndarray = field(default_factory=lambda: np.array([1.03, 0.98, 1.01]))
    acc_bias: np.ndarray = field(default_factory=lambda: np.array([0.15, -0.10, 0.20]))        # m/s^2
    acc_scale: np.ndarray = field(default_factory=lambda: np.array([0.99, 1.02, 1.01]))
    gyro_noise: float = 0.01
    acc_noise: float = 0.05
    lowpass_hz: float = 20.0
    delay_s: float = 0.035
    fs: float = 100.0

    def measure(self, t, P, Rs, seed=2):
        """True motion of the brush head -> what the IMU reports, at its own rate."""
        rng = np.random.default_rng(seed)
        dt = np.diff(t).mean()
        w_body = (Rs[:-1].inv() * Rs[1:]).as_rotvec() / dt                       # body angular rate
        w_body = np.vstack([w_body, w_body[-1]])
        a_world = np.gradient(np.gradient(P, dt, axis=0), dt, axis=0)
        f_body = Rs.inv().apply(a_world - G)                                     # specific force
        s = self.mount.inv()                                                     # brush -> sensor axes
        gyro = s.apply(w_body) * self.gyro_scale + self.gyro_bias
        acc = s.apply(f_body) * self.acc_scale + self.acc_bias
        a = np.exp(-2 * np.pi * self.lowpass_hz * dt)                            # 1st-order low-pass
        gyro, acc = lfilter([1 - a], [1, -a], gyro, axis=0), lfilter([1 - a], [1, -a], acc, axis=0)
        t_imu = np.arange(t[0], t[-1], 1 / self.fs)
        tq = t_imu - self.delay_s                                                # sensor reports late
        gi = np.stack([np.interp(tq, t, gyro[:, i]) for i in range(3)], 1)
        ai = np.stack([np.interp(tq, t, acc[:, i]) for i in range(3)], 1)
        return t_imu, gi + self.gyro_noise * rng.standard_normal(gi.shape), ai + self.acc_noise * rng.standard_normal(ai.shape)


# ----------------------------------------------------------------------------- calibration
def calibrate_mocap(arm, mocap, n=25, seed=3):
    """Robot-to-mocap calibration: move to n random poses, read the marker in the mocap
    frame, and solve for the mocap frame pose AND the marker offset on the brush jointly."""
    rng = np.random.default_rng(seed)
    qs = arm.q_home + rng.uniform(-0.6, 0.6, (n, 6))
    P, Rq = zip(*[arm.tcp(q) for q in qs]); P = np.array(P); Rq = R.concatenate(Rq)
    pm, _ = mocap.measure(P, Rq, seed=seed)

    def resid(x):
        Rwm, pwm, off = R.from_rotvec(x[:3]), x[3:6], x[6:9]
        return (Rwm.inv().apply(P + Rq.apply(off) - pwm) - pm).ravel()
    x = least_squares(resid, np.zeros(9)).x
    return R.from_rotvec(x[:3]), x[3:6], x[6:9], np.sqrt(np.mean(resid(x) ** 2))


def estimate_delay(t_ref, sig_ref, t_s, sig_s, max_lag=0.2):
    """Time offset between two rate magnitudes by cross-correlation (sub-sample)."""
    fs = 1000.0
    tg = np.arange(max(t_ref[0], t_s[0]), min(t_ref[-1], t_s[-1]), 1 / fs)
    a = np.interp(tg, t_ref, sig_ref); b = np.interp(tg, t_s, sig_s)
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(-int(max_lag * fs), int(max_lag * fs) + 1)
    c = np.array([np.dot(a[max(0, -l):len(a) - max(0, l)], b[max(0, l):len(b) - max(0, -l)]) for l in lags])
    k = np.argmax(c)
    if 0 < k < len(c) - 1:                                       # parabolic refinement
        k = k + 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1])
    return (lags[0] + k) / fs


def calibrate_imu(t_ref, w_ref, f_ref, t_imu, gyro, acc):
    """Separate true motion from sensor effects. Reference = brush-head rates from the
    calibrated mocap (or the robot). Returns the estimated delay, mounting rotation,
    nearest axis convention, per-axis scale and bias, low-pass cutoff and noise."""
    delay = estimate_delay(t_ref, np.linalg.norm(w_ref, axis=1), t_imu, np.linalg.norm(gyro, axis=1))
    wr = np.stack([np.interp(t_imu - delay, t_ref, w_ref[:, i]) for i in range(3)], 1)
    fr = np.stack([np.interp(t_imu - delay, t_ref, f_ref[:, i]) for i in range(3)], 1)
    # mounting rotation: Kabsch on bias-removed gyro vs reference rates
    A, B = wr - wr.mean(0), gyro - gyro.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    S = U @ D @ Vt                                                # sensor <- brush
    convention = np.round(S)                                      # nearest signed permutation
    misalign_deg = np.degrees(R.from_matrix(S @ convention.T).magnitude())

    def best_fit(ref_b, meas):
        ref_s = ref_b @ S.T
        sc, bi, fc, res = np.zeros(3), np.zeros(3), None, None
        best = (np.inf, None)
        for cut in np.arange(5, 60, 1.0):                         # low-pass cutoff grid search
            a = np.exp(-2 * np.pi * cut * np.mean(np.diff(t_imu)))
            filt = lfilter([1 - a], [1, -a], ref_s, axis=0)
            X = [np.linalg.lstsq(np.c_[filt[:, i], np.ones(len(filt))], meas[:, i], rcond=None)[0] for i in range(3)]
            r = meas - np.stack([filt[:, i] * X[i][0] + X[i][1] for i in range(3)], 1)
            e = np.mean(r[50:] ** 2)
            if e < best[0]:
                best = (e, (cut, np.array([x[0] for x in X]), np.array([x[1] for x in X]), r[50:].std(0)))
        return best[1]
    gcut, gscale, gbias, gnoise = best_fit(wr, gyro)
    acut, ascale, abias, anoise = best_fit(fr, acc)
    transport = delay - 1 / (2 * np.pi * gcut)                   # remove the low-pass group delay
    return dict(delay=delay, transport_delay=transport, S=S, convention=convention, misalign_deg=misalign_deg,
                gyro_scale=gscale, gyro_bias=gbias, gyro_noise=gnoise.mean(), lowpass_hz=gcut,
                acc_scale=ascale, acc_bias=abias, acc_noise=anoise.mean())


def apply_calibration(cal, gyro):
    """Map raw gyro samples back to the brush frame with the estimated sensor model."""
    return ((gyro - cal["gyro_bias"]) / cal["gyro_scale"]) @ cal["S"]


def body_rates(t, P, Rs, smooth=0):
    """Reference angular rate (brush frame) and specific force from poses.
    smooth > 0 applies a Savitzky-Golay filter (window in samples) before differentiating,
    which matters for noisy motion-capture data."""
    dt = np.diff(t).mean()
    if smooth:
        from scipy.signal import savgol_filter
        P = savgol_filter(P, smooth, 3, axis=0)
        qv = Rs.as_quat()
        qv *= np.sign(np.sum(qv * qv[[0]], axis=1, keepdims=True))      # same hemisphere
        qv = savgol_filter(qv, smooth, 3, axis=0)
        Rs = R.from_quat(qv / np.linalg.norm(qv, axis=1, keepdims=True))
    w = (Rs[:-1].inv() * Rs[1:]).as_rotvec() / dt
    w = np.vstack([w, w[-1]])
    f = Rs.inv().apply(np.gradient(np.gradient(P, dt, axis=0), dt, axis=0) - G)
    return w, f


def calibrate_imu_protocol(segments, skip_s=0.5):
    """Calibrate an IMU from the validation protocol instead of from free motion.

    segments: list of dicts with t_ref, w_ref, f_ref (reference, brush frame), t_imu, gyro,
    acc and kind ('rotation' or 'static'). Rotation tests excite every gyro axis on
    purpose, and the six static poses give +-1 g on every accelerometer axis, so every
    parameter is identifiable (brushing alone barely rotates the brush about one axis).
    """
    rot = [s for s in segments if s["kind"] == "rotation"]
    delay = float(np.median([estimate_delay(s["t_ref"], np.linalg.norm(s["w_ref"], axis=1),
                                            s["t_imu"], np.linalg.norm(s["gyro"], axis=1)) for s in rot]))
    for s in segments:
        keep = s["t_imu"] > s["t_imu"][0] + skip_s
        s["keep"] = keep
        s["wr"] = np.stack([np.interp(s["t_imu"] - delay, s["t_ref"], s["w_ref"][:, i]) for i in range(3)], 1)
        s["fr"] = np.stack([np.interp(s["t_imu"] - delay, s["t_ref"], s["f_ref"][:, i]) for i in range(3)], 1)
    # mounting rotation from the rotation tests (Kabsch on bias-removed rates)
    A = np.vstack([s["wr"][s["keep"]] for s in rot]); B = np.vstack([s["gyro"][s["keep"]] for s in rot])
    A, B = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    S = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    convention = np.round(S)
    misalign_deg = np.degrees(R.from_matrix(S @ convention.T).magnitude())

    def fit(segs, ref_key, meas_key):
        best = (np.inf, None)
        for cut in np.arange(4, 61, 1.0):
            Xs, Ys = [], []
            for s in segs:
                a = np.exp(-2 * np.pi * cut * np.mean(np.diff(s["t_imu"])))
                filt = lfilter([1 - a], [1, -a], s[ref_key] @ S.T, axis=0, zi=None) if False else \
                    lfilter([1 - a], [1, -a], s[ref_key] @ S.T - (s[ref_key] @ S.T)[0], axis=0) + (s[ref_key] @ S.T)[0]
                Xs.append(filt[s["keep"]]); Ys.append(s[meas_key][s["keep"]])
            X, Y = np.vstack(Xs), np.vstack(Ys)
            coef = [np.linalg.lstsq(np.c_[X[:, i], np.ones(len(X))], Y[:, i], rcond=None)[0] for i in range(3)]
            r = Y - np.stack([X[:, i] * coef[i][0] + coef[i][1] for i in range(3)], 1)
            if np.mean(r ** 2) < best[0]:
                best = (np.mean(r ** 2), (cut, np.array([c[0] for c in coef]), np.array([c[1] for c in coef]), r.std(0).mean()))
        return best[1]
    gcut, gscale, gbias, _ = fit(rot, "wr", "gyro")
    _, ascale, abias, _ = fit(segments, "fr", "acc")
    # sensor noise: measured while the brush is held still, so mocap noise does not leak in
    stat = [s for s in segments if s["kind"] == "static"]
    gnoise = float(np.mean([s["gyro"][s["keep"]].std(0).mean() for s in stat]))
    anoise = float(np.mean([s["acc"][s["keep"]].std(0).mean() for s in stat]))
    return dict(delay=delay, transport_delay=delay - 1 / (2 * np.pi * gcut), S=S, convention=convention,
                misalign_deg=misalign_deg, gyro_scale=gscale, gyro_bias=gbias, gyro_noise=gnoise,
                lowpass_hz=gcut, acc_scale=ascale, acc_bias=abias, acc_noise=anoise)
