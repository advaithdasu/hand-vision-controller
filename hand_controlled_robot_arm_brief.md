# Hand-Controlled Simulated Robot Arm via Computer Vision

## What This Project Is

A real-time teleoperation system that uses a standard laptop webcam to detect and track a human hand, then maps that hand's pose directly to the joint angles of a simulated 6-DOF robot arm — which moves in a live 3D physics simulation in response. No hardware required. The operator's hand is the controller.

This sits at the intersection of computer vision, robotics kinematics, and real-time control systems. It's the software core of what powers teleoperated surgical robots, prosthetic limb control, remote manipulation systems, and human-robot collaboration interfaces.

---

## The Problem It Addresses

Controlling a robot arm intuitively is a hard problem. Traditional controllers — joysticks, teach pendants, gantry systems — require training and create a cognitive gap between the operator's intent and the robot's motion. The natural interface is the human hand itself: if the robot can watch your hand and mirror it, the learning curve collapses and the operator can focus entirely on the task.

This project builds that interface in software, demonstrating that the core technical challenge — real-time hand pose estimation feeding a live inverse kinematics solver — is tractable on commodity hardware.

---

## What the System Does

**Hand Pose Detection**

A computer vision pipeline runs continuously on the webcam feed, detecting the operator's hand and extracting its full skeletal pose — the 3D orientation of the palm, the curl state of each finger, and the overall position of the hand in the camera's field of view. This runs in real time, fast enough that there is no perceptible lag between hand movement and system response.

**Pose-to-Command Mapping**

The raw hand pose is translated into a set of control commands for the robot arm. Palm orientation maps to the orientation of the arm's end effector (the "wrist" of the robot). Hand position in the camera frame maps to the target position the end effector should reach in 3D space. Finger state — open, closed, pinch — maps to discrete commands like gripper open/close or mode switching.

**Inverse Kinematics Solver**

Given a target position and orientation for the end effector, an inverse kinematics (IK) solver computes the joint angles required across all six joints of the arm to achieve that pose. This is the mathematically interesting core of the project — IK has no unique solution and requires careful handling of singularities, joint limits, and smooth interpolation between configurations so the arm doesn't snap between positions.

**Physics Simulation**

The robot arm lives inside a physics simulator (PyBullet or MuJoCo). This gives the demo physical realism — the arm has mass, joint constraints, and responds to its own inertia. It also means the system can be extended to include objects the arm interacts with: picking up a block, placing it somewhere, responding to contact forces.

**Live Interface**

The operator sees two things simultaneously: their webcam feed with a skeletal hand overlay showing what the system is detecting, and the 3D simulation window showing the arm responding in real time. The visual feedback loop — move your hand, watch the arm move — is the demo moment that makes this project immediately legible to anyone watching.

---

## Why This Is Technically Impressive

The project stacks three distinct engineering disciplines that rarely appear together in a portfolio:

- **Computer vision** — real-time hand pose estimation from a monocular camera, handling varying lighting, partial occlusion, and motion blur
- **Robotics kinematics** — inverse kinematics on a 6-DOF serial chain, including singularity avoidance and joint limit enforcement
- **Real-time control** — closing the loop between perception and actuation at a frame rate that feels responsive, with smooth interpolation to avoid jerky motion

Each of these is a legitimate specialization on its own. Combining them into a working system demonstrates systems-level thinking that is rare in junior candidates.

---

## The Components to Build

1. **Hand pose estimation module** — real-time detection and 3D keypoint extraction from webcam feed, producing palm orientation and finger state at each frame

2. **Coordinate transform layer** — maps camera-space hand coordinates to robot-space end effector targets, handling the geometric relationship between what the camera sees and where the arm should go

3. **IK solver** — computes per-joint angles from end effector targets; handles joint limits, singularities, and smooth interpolation between frames

4. **Robot arm simulation** — a 6-DOF arm model running in a physics engine, accepting joint angle commands and rendering in real time

5. **Gripper control** — maps finger pinch/open state to gripper commands, enabling pick-and-place style interactions with simulated objects

6. **Unified display** — side-by-side or overlay view of the webcam feed (with hand skeleton) and the simulation window, so the operator sees both simultaneously

---

## Stretch Goals That Elevate the Project

- **Pick-and-place demo** — place a target object in the simulation and have the operator use their hand to guide the arm to pick it up and move it. Turns a tech demo into a task demo.
- **Gesture mode switching** — use specific hand gestures to switch between control modes (position control vs orientation control vs gripper-only), showing the system can handle a richer command vocabulary
- **Latency profiling** — measure and display the end-to-end latency from hand movement to arm response, and optimize it. Shows engineering rigor beyond just "it works."
- **Record and replay** — log a manipulation sequence and play it back on the arm, previewing the idea of learning from demonstration

---

## Scope and Positioning

This is a simulation-only project — no physical hardware needed. The architecture is deliberately designed so that replacing the simulated arm with a real one (via a serial or ROS interface) would be a relatively contained change. That extensibility is worth mentioning explicitly: this is the software stack of a real teleoperation system, running in simulation.

It should live on GitHub with a demo video as the centerpiece of the README — ideally showing the operator's hand and the arm in a split-screen view, picking up a simulated object.

---

## Who This Impresses

- Surgical and medical robotics companies (Intuitive Surgical, Moon Surgical, Vicarious Surgical)
- Industrial automation and collaborative robotics firms (Boston Dynamics, Machina Labs, Sanctuary AI)
- Defense and teleoperation companies working on remote manipulation (Sarcos, RE2 Robotics, Anduril)
- Any deep tech employer who sees robotics, CV, and control systems fluency as a hiring signal

Paired with the swarm tracker project, this establishes a clear technical identity: someone who works at the intersection of perception, estimation, and control — which is exactly the profile that deep tech robotics companies are looking for.
