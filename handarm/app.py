"""Real-time teleoperation loop: webcam -> hand pose -> IK -> physics -> display.

Keyboard (focus the display window):
  q / ESC   quit
  c         calibrate: capture current hand pose as the neutral reference
  o         toggle orientation control (position-only when off)
  f         manual freeze toggle (independent of the fist clutch)
  r         start / stop recording a demonstration
  x         reset the simulation scene

Clutching: a fist freezes arm *and gripper* so the operator can reposition
their hand. If an object was held when the clutch engaged, the grasp stays
latched after release until the operator pinches again — otherwise opening
the hand to disengage the clutch would drop the object.
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
        self.ori_filter = QuaternionLowPass(ctl.ori_cutoff)
        self.pinch_filter = ScalarLowPass(cutoff=8.0)
        self.profiler = LatencyProfiler()
        self.recorder = Recorder(cfg.recordings_dir)

        self.frame_aspect = cfg.camera_width / cfg.camera_height
        self.orientation_on = cfg.orientation_control
        self.calibrated = False
        self.lost_frames = 999
        self.last_ik = None
        self.reset_control()

    def reset_control(self) -> None:
        """Return all control state to the home configuration."""
        self.q_cmd = HOME_Q.copy()
        self.target_pos, self.target_quat = self.kin.fk(HOME_Q)
        self.grip_opening = 1.0
        self.gripping = False
        self.grip_latched = False
        self.clutched = False
        self.manual_freeze = False
        self.pos_filter.reset()
        self.ori_filter.reset()
        self.pinch_filter.reset()

    @property
    def frozen(self) -> bool:
        return self.clutched or self.manual_freeze

    def _hand_scale(self, image_landmarks: np.ndarray) -> float:
        """Apparent hand size in aspect-corrected normalized units.

        MediaPipe normalizes x by image width and y by height, so on a
        16:9 frame a raw wrist->MCP norm would change ~1.8x as the hand
        rotates in-plane; scaling x by the aspect ratio makes the measure
        isotropic (in units of image height).
        """
        lm = image_landmarks.copy()
        lm[:, 0] *= self.frame_aspect
        return pf.hand_scale(lm)

    def calibrate(self, obs) -> None:
        """Capture the current hand pose as the neutral reference."""
        self.mapper.calibrate(
            self._hand_scale(obs.image_landmarks), pf.palm_frame(obs.world_landmarks)
        )
        self.calibrated = True
        self.pos_filter.reset()
        self.ori_filter.reset()

    # ---------------- per-frame pipeline ----------------

    def process_hand(self, obs, dt: float) -> None:
        """Turn a hand observation into filtered targets and gripper state."""
        img, world = obs.image_landmarks, obs.world_landmarks
        ext = pf.finger_extensions(world)
        pinch_raw = pf.pinch_ratio(world)

        # Clutch: a fist freezes tracking so the operator can reposition
        # their hand without dragging the arm along; an open hand releases.
        if pf.is_fist_from(ext, pinch_raw):
            self.clutched = True
        elif self.clutched and pf.count_extended_from(ext) >= 3:
            self.clutched = False
            self.pinch_filter.reset()
            if self.gripping:
                # Keep the grasp until the operator pinches again, so
                # releasing the clutch doesn't drop a held object.
                self.grip_latched = True

        if self.frozen:
            return

        # Gripper: pinch ratio -> aperture, with hysteresis for the grab flag.
        ctl = self.cfg.control
        pinch = self.pinch_filter(pinch_raw, dt)
        if self.grip_latched:
            if pinch < ctl.pinch_close:
                self.grip_latched = False  # operator re-pinched: re-arm
        else:
            if self.gripping and pinch > ctl.pinch_open:
                self.gripping = False
            elif not self.gripping and pinch < ctl.pinch_close:
                self.gripping = True
            self.grip_opening = float(np.clip(
                (pinch - ctl.pinch_close)
                / (ctl.pinch_open + ctl.aperture_margin - ctl.pinch_close),
                0, 1,
            ))

        if not self.calibrated:
            self.calibrate(obs)

        scale = self._hand_scale(img)
        raw_pos = self.mapper.map_position(pf.palm_center(img)[:2], scale)
        self.target_pos = self.pos_filter(raw_pos, dt)
        if self.orientation_on:
            raw_quat = self.mapper.map_orientation(pf.palm_frame(world))
            self.target_quat = self.ori_filter(raw_quat, dt)
        else:
            self.target_quat = self.ori_filter(TOOL_DOWN_QUAT, dt)

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
        if self.manual_freeze:
            mode = "FROZEN (key)"
        elif self.clutched:
            mode = "CLUTCHED (fist)"
        else:
            mode = "pos + orientation" if self.orientation_on else "position only"
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
            lines.append(("show your hand to calibrate (press c to re-zero)", True))
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

                if obs is not None:
                    self.lost_frames = 0
                    self.process_hand(obs, dt)
                else:
                    self.lost_frames += 1

                hold = self.lost_frames >= self.cfg.control.hold_after_lost_frames
                if not (hold and obs is None):
                    self.solve_and_command(dt)
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

                if not self.handle_key(key, obs):
                    break
        finally:
            self.recorder.stop()
            tracker.close()
            cap.release()
            cv2.destroyAllWindows()

    def handle_key(self, key: int, obs) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("o"):
            self.orientation_on = not self.orientation_on
        elif key == ord("f"):
            self.manual_freeze = not self.manual_freeze
        elif key == ord("x"):
            self.sim.reset()
            self.reset_control()
        elif key == ord("c") and obs is not None:
            self.calibrate(obs)
        elif key == ord("r"):
            if self.recorder.active:
                path = self.recorder.stop()
                print(f"saved recording: {path}")
            else:
                print(f"recording to: {self.recorder.start()}")
        return True

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
