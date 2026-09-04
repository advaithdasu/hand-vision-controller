"""CLI entry point: python -m handarm [options]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .app import TeleopApp
from .config import AppConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="handarm",
        description="Hand-controlled simulated 6-DOF robot arm via webcam.",
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--no-orientation", action="store_true",
                        help="position-only control (tool always points down)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="do not mirror the webcam image")
    parser.add_argument("--replay", type=Path, metavar="FILE",
                        help="replay a recorded .jsonl trajectory instead of live control")
    parser.add_argument("--autopilot", action="store_true",
                        help="scripted pick-and-place demo, no camera needed")
    args = parser.parse_args()

    cfg = AppConfig(
        camera_index=args.camera,
        orientation_control=not args.no_orientation,
        mirror=not args.no_mirror,
    )
    app = TeleopApp(cfg)
    if args.replay:
        app.replay(args.replay)
    elif args.autopilot:
        app.run_autopilot()
    else:
        app.run()


if __name__ == "__main__":
    main()
