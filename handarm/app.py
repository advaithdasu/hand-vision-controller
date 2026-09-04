"""Real-time teleoperation loop: webcam -> hand pose -> IK -> physics -> display.

Keyboard (focus the display window):
  q / ESC   quit
  c         calibrate: take the current hand pose as the neutral reference
  o         toggle orientation control (position-only when off)
  f         manual freeze toggle (independent of the fist clutch)
  r         start / stop recording a demonstration
  x         reset the simulation scene

Calibration: the first detection is usually a hand entering at the frame
edge, so auto-calibration waits until the hand is fully in view and nearly
still for half a second. Pressing `c` calibrates immediately.

Clutching: a fist freezes arm *and gripper* so the operator can reposition
their hand. On release the maps are re-anchored so the arm resumes from
where it froze rather than swinging to the hand's new absolute position.
If an object was held when the clutch engaged, the grasp stays latched
after release until the operator pinches again — otherwise opening the
hand to disengage the clutch would drop the object.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import pose_features as pf
from .config import HOME_Q, AppConfig
from .filters import OneEuroFilter, QuaternionLowPass, ScalarLowPass
from .ik import DLSSolver
from .kinematics import ArmKinematics
from .latency import LatencyProfiler
from .autopilot import Autopilot, SimPlant
from .mapping import TOOL_DOWN_QUAT, HandToRobotMapper
from .overlay import compose_split, draw_gripper_bar, draw_hand_skeleton, draw_hud
from .recorder import Recorder, load_recording
from .sim import ArmSim

WINDOW = "Hand-Controlled Robot Arm"


class TeleopApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.sim = ArmSim(render_size=cfg.sim_render_size)
        self.kin = ArmKinematics(self.sim.model)
        self.solver = DLSSolver(self.kin, cfg.ik)
        self.mapper = HandToRobotMapper(cfg.mapping)

        ctl = cfg.control
        self.pos_filter = OneEuroFilter(ctl.pos_min_cutoff, ctl.pos_beta, ctl.pos_d_cutoff)
        self.ori_filter = QuaternionLowPass(ctl.ori_cutoff, ctl.ori_beta, ctl.ori_d_cutoff)
        self.pinch_filter = ScalarLowPass(cutoff=ctl.pinch_cutoff)
        self.profiler = LatencyProfiler()
        self.recorder = Recorder(cfg.recordings_dir)

        self.frame_aspect = cfg.camera_width / cfg.camera_height
        self._orientation_on = cfg.orientation_control
        self.calibrated = False
        self.calib_progress = 0.0
        self.lost_frames = 999
        self.last_ik = None
        self.last_obs = None
        self._calib_hold = 0.0
        self._calib_prev_center = None
        self.reset_control()

    def reset_control(self) -> None:
        """Return all control state to the home configuration."""
        self.q_cmd = HOME_Q.copy()
        self.target_pos, self.target_quat = self.kin.fk(HOME_Q)
        self.grip_opening = 1.0
        self.gripping = False
        self.grip_latched = False
        self.clutched = False
        self._manual_freeze = False
        self._fist_time = 0.0
        self._open_time = 0.0
        self._rebase_pos_pending = False
        self._rebase_ori_pending = False
        self.mapper.pos_offset = np.zeros(3)
        self.pos_filter.reset()
        self.ori_filter.reset()
        self.pinch_filter.reset()

    @property
    def frozen(self) -> bool:
        return self.clutched or self._manual_freeze

    @property
    def manual_freeze(self) -> bool:
        return self._manual_freeze

    @manual_freeze.setter
    def manual_freeze(self, on: bool) -> None:
        if self._manual_freeze and not on:
            self._mark_rebase()
        self._manual_freeze = on

    @property
    def orientation_on(self) -> bool:
        return self._orientation_on

    @orientation_on.setter
    def orientation_on(self, on: bool) -> None:
        # Turning orientation back on would otherwise snap the wrist from
        # tool-down to the hand's absolute tilt.
        if not self._orientation_on and on:
            self._rebase_ori_pending = True
        self._orientation_on = on

    def _mark_rebase(self) -> None:
        self._rebase_pos_pending = True
        self._rebase_ori_pending = True

    def _hand_scale(self, obs) -> float:
        """Orientation-invariant apparent hand size (see pose_features)."""
        return pf.apparent_scale(obs.image_landmarks, obs.world_landmarks, self.frame_aspect)

    def request_calibration(self) -> None:
        """Take the current hand pose as neutral. With a hand in view this
        is immediate (frozen or not); otherwise the hold-still gate re-arms
        for the next detection."""
        if self.last_obs is not None:
            self.calibrate(self.last_obs)
        else:
            self.calibrated = False
            self._calib_hold = 0.0
            self.calib_progress = 0.0

    def calibrate(self, obs) -> None:
        """Capture the current hand pose as the neutral reference."""
        self.mapper.calibrate(self._hand_scale(obs), pf.palm_frame(obs.world_landmarks))
        self.calibrated = True
        self.calib_progress = 1.0
        self._calib_hold = 0.0
        self._rebase_pos_pending = False
        self._rebase_ori_pending = False
        self.pos_filter.reset()
        self.ori_filter.reset()

    def _update_calibration_gate(self, obs, dt: float) -> None:
        """Auto-calibrate once the hand is fully in view and nearly still."""
        ctl = self.cfg.control
        c = pf.palm_center(obs.image_landmarks)[:2] * np.array([self.frame_aspect, 1.0])
        speed = 0.0
        if self._calib_prev_center is not None and dt > 0:
            speed = float(np.linalg.norm(c - self._calib_prev_center)) / dt
        self._calib_prev_center = c
        steady = (
            pf.fully_in_frame(obs.image_landmarks, ctl.calib_edge_margin)
            and speed < ctl.calib_max_speed
        )
        self._calib_hold = self._calib_hold + dt if steady else 0.0
        self.calib_progress = min(self._calib_hold / ctl.calib_settle_sec, 1.0)
        if self._calib_hold >= ctl.calib_settle_sec:
            self.calibrate(obs)

    def _update_gripper(self, pinch_raw: float, dt: float) -> None:
        """Pinch ratio -> aperture, with hysteresis for the grab flag."""
        ctl = self.cfg.control
        pinch = self.pinch_filter(pinch_raw, dt)
        if self.grip_latched:
            if pinch < ctl.pinch_close:
                self.grip_latched = False  # operator re-pinched: re-arm
            return
        if self.gripping and pinch > ctl.pinch_open:
            self.gripping = False
        elif not self.gripping and pinch < ctl.pinch_close:
            self.gripping = True
        self.grip_opening = float(np.clip(
            (pinch - ctl.pinch_close)
            / (ctl.pinch_open + ctl.aperture_margin - ctl.pinch_close),
            0, 1,
        ))

    # ---------------- per-frame pipeline ----------------

    def process_hand(self, obs, dt: float) -> None:
        """Turn a hand observation into filtered targets and gripper state."""
        ctl = self.cfg.control
        img, world = obs.image_landmarks, obs.world_landmarks
        ext = pf.finger_extensions(world)
        pinch_raw = pf.pinch_ratio(world)

        # Clutch: a held fist freezes tracking so the operator can reposition
        # their hand without dragging the arm along; a held open hand
        # releases. Both are debounced against single-frame misdetections.
        fist = pf.is_fist_from(ext)
        opened = pf.count_extended_from(ext) >= 3
        self._fist_time = self._fist_time + dt if fist else 0.0
        self._open_time = self._open_time + dt if opened else 0.0
        if not self.clutched:
            if self._fist_time >= ctl.clutch_engage_sec:
                self.clutched = True
        elif self._open_time >= ctl.clutch_release_sec:
            self.clutched = False
            self.pinch_filter.reset()
            if self.gripping:
                # Keep the grasp until the operator pinches again, so
                # releasing the clutch doesn't drop a held object.
                self.grip_latched = True
            self._mark_rebase()

        if self.frozen:
            return

        # A fist-shaped frame never drives the gripper, even before the
        # clutch debounce elapses: a tucked thumb can read as a tight pinch.
        if not fist:
            self._update_gripper(pinch_raw, dt)

        if not self.calibrated:
            self._update_calibration_gate(obs, dt)
            if not self.calibrated:
                return

        scale = self._hand_scale(obs)
        xy = pf.palm_center(img)[:2]
        if self._rebase_pos_pending:
            self.mapper.rebase_position(xy, scale, self.target_pos)
            self.pos_filter.reset()
            self._rebase_pos_pending = False
        self.target_pos = self.pos_filter(self.mapper.map_position(xy, scale), dt)

        raw_quat = TOOL_DOWN_QUAT
        if self._orientation_on:
            palm_rot = pf.palm_frame(world)
            if self._rebase_ori_pending:
                self.mapper.rebase_orientation(palm_rot, self.target_quat)
                self.ori_filter.reset()
            raw_quat = self.mapper.map_orientation(palm_rot)
        self._rebase_ori_pending = False
        self.target_quat = self.ori_filter(raw_quat, dt)

    def solve_and_command(self, dt: float) -> None:
        """IK to the current target, then rate-limit the joint command."""
        result = self.solver.solve(self.q_cmd, self.target_pos, self.target_quat)
        self.last_ik = result
        max_dq = self.cfg.control.max_joint_vel * max(dt, 1e-3)
        self.q_cmd = self.q_cmd + np.clip(result.q - self.q_cmd, -max_dq, max_dq)
        self.sim.set_joint_targets(self.q_cmd)
        self.sim.set_gripper(self.grip_opening)
        self.sim.set_target_marker(self.target_pos)

    def hud_lines(self, hand_visible: bool) -> list[tuple[str, bool]]:
        p = self.profiler
        ik = self.last_ik
        if self._manual_freeze:
            mode = "FROZEN (key)"
        elif self.clutched:
            mode = "CLUTCHED (fist) - open your hand to resume"
        else:
            mode = "pos + orientation" if self._orientation_on else "position only"
        grip = "LATCHED" if self.grip_latched else ("CLOSED" if self.gripping else "open")
        lines = [
            (f"FPS {p.fps():5.1f}   end-to-end {p.mean_ms('total'):5.1f} ms", False),
            (f"capture {p.mean_ms('capture'):4.1f}  detect {p.mean_ms('detect'):4.1f}  "
             f"ik {p.mean_ms('ik'):4.1f}  sim {p.mean_ms('sim'):4.1f}  "
             f"draw {p.mean_ms('draw'):4.1f} ms", False),
            (f"mode: {mode}", self.frozen),
            (f"hand: {'tracking' if hand_visible else 'NOT FOUND'}", not hand_visible),
            (f"grip: {grip}  aperture {self.grip_opening:0.2f}",
             self.gripping or self.grip_latched),
        ]
        if ik is not None:
            lines.append(
                (f"ik: {ik.iters} it  err {ik.pos_err*1000:5.1f} mm / "
                 f"{np.degrees(ik.ori_err):4.1f} deg  w {ik.manipulability:0.3f}",
                 not ik.converged)
            )
        if not self.calibrated:
            if hand_visible:
                n = int(round(self.calib_progress * 10))
                lines.append((f"hold still to lock on  [{'#' * n}{'.' * (10 - n)}]", True))
            else:
                lines.append(("show your open hand to the camera (c re-zeros)", True))
        if self.recorder.active:
            lines.append(("REC", True))
        return lines

    # ---------------- main loops ----------------

    def run(self) -> None:
        from .hand_tracker import HandTracker

        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.cfg.camera_index}. "
                "Check macOS camera permissions for your terminal."
            )
        tracker = HandTracker()
        t_start = time.perf_counter()
        t_prev = t_start

        try:
            while True:
                self.profiler.frame_start()
                now = time.perf_counter()
                dt, t_prev = now - t_prev, now

                ok, frame = cap.read()
                if not ok:
                    break
                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)
                self.frame_aspect = frame.shape[1] / frame.shape[0]
                self.profiler.mark("capture")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs = tracker.detect(rgb, int((now - t_start) * 1000))
                self.profiler.mark("detect")

                self.tick(obs, dt)
                self.profiler.mark("ik")

                self.sim.step(dt)
                self.profiler.mark("sim")

                if self.recorder.active:
                    self.recorder.record(self.q_cmd, self.grip_opening)

                if obs is not None:
                    draw_hand_skeleton(frame, obs.image_landmarks)
                draw_gripper_bar(frame, self.grip_opening,
                                 self.gripping or self.grip_latched)
                draw_hud(frame, self.hud_lines(obs is not None))
                view = compose_split(frame, self.sim.render(), self.cfg.display_height)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF  # also pumps the UI event loop
                self.profiler.mark("draw")
                self.profiler.frame_end()

                if not self.handle_key(key):
                    break
        finally:
            self.recorder.stop()
            tracker.close()
            cap.release()
            cv2.destroyAllWindows()

    def tick(self, obs, dt: float) -> None:
        """One control tick without physics: gesture/mapping, then IK unless
        the hand has been lost long enough to hold position."""
        self.last_obs = obs
        if obs is not None:
            self.lost_frames = 0
            self.process_hand(obs, dt)
        else:
            self.lost_frames += 1
            self._fist_time = 0.0
            self._open_time = 0.0
            self._calib_prev_center = None
        hold = self.lost_frames >= self.cfg.control.hold_after_lost_frames
        if not hold:
            self.solve_and_command(dt)

    def tick_scripted(self, dt: float) -> None:
        """One control tick with the targets already set by a script (the
        autopilot): the same IK -> rate-limited command path as a hand
        frame, minus gesture processing. A manual freeze holds the arm."""
        self.last_obs = None
        if not self.frozen:
            self.solve_and_command(dt)

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("o"):
            self.orientation_on = not self.orientation_on
        elif key == ord("f"):
            self.manual_freeze = not self.manual_freeze
        elif key == ord("x"):
            self.sim.reset()
            self.reset_control()
        elif key == ord("c"):
            self.request_calibration()
        elif key == ord("r"):
            if self.recorder.active:
                path = self.recorder.stop()
                print(f"saved recording: {path}")
            else:
                print(f"recording to: {self.recorder.start()}")
        return True

    def run_autopilot(self) -> None:
        """Scripted pick-and-place with no camera: the arm picks each cube
        and drops it in the tray through the live IK -> physics path,
        looping after a rest. Keys: q quits, x resets, f freezes."""
        pilot = Autopilot(self, SimPlant(self.sim, self.kin))
        t_prev = time.perf_counter()
        done_at = None
        while True:
            now = time.perf_counter()
            dt, t_prev = min(now - t_prev, 0.1), now
            pilot.tick(dt)
            self.sim.step(dt)
            if pilot.round_complete:
                done_at = done_at or now
                if now - done_at > 2.5:
                    done_at = None
                    self.sim.reset()
                    self.reset_control()
                    pilot.restart()

            view = cv2.cvtColor(self.sim.render(), cv2.COLOR_RGB2BGR)
            hud = f"AUTOPILOT  {pilot.status}"
            if self._manual_freeze:
                hud = "FROZEN (f)"
            cv2.putText(view, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (60, 220, 120), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                self.manual_freeze = not self.manual_freeze
            elif key == ord("x"):
                self.sim.reset()
                self.reset_control()
                pilot.restart()
        cv2.destroyAllWindows()

    def replay(self, path: Path) -> None:
        """Play a recorded joint trajectory back into the simulation."""
        frames = load_recording(path)
        if not frames:
            print(f"empty recording: {path}")
            return
        print(f"replaying {len(frames)} frames from {path}")
        t_start = time.perf_counter()
        t_prev = t_start
        i = 0
        while i < len(frames):
            now = time.perf_counter()
            dt, t_prev = now - t_prev, now
            while i < len(frames) and frames[i]["t"] <= now - t_start:
                q = np.array(frames[i]["q"])
                self.sim.set_joint_targets(q)
                self.sim.set_gripper(frames[i]["grip"])
                i += 1
            self.sim.step(dt)
            view = cv2.cvtColor(self.sim.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(view, f"REPLAY {i}/{len(frames)}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 120), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
