/**
 * Three.js rendering + Rapier physics for the arm and pick-and-place scene.
 *
 * The arm itself is kinematic (driven by joint angles from IK, mirroring
 * the geometry in assets/scene.xml); the cubes are dynamic Rapier bodies.
 * Grasping is a velocity-driven attachment: when the gripper closes with a
 * cube between the fingers, the cube stays a *dynamic* body that is
 * servoed toward its pose in the tool frame every physics substep. It
 * follows the gripper rigidly in free space, yet still resolves contacts —
 * driving it into the tray wall or the floor stops it there instead of
 * tunneling through — which contact-friction grasping in a browser
 * physics engine cannot deliver without far more tuning.
 *
 * Everything lives in a z-up world to match the robot-frame math.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";

import { CHAIN, fk } from "./kinematics";
import {
  matrixFromQuat,
  matTranspose3,
  orientationError,
  type Quat,
  quatConjugate,
  quatMultiply,
  quatSlerp,
  type Vec3,
  vecAdd,
  vecNorm,
  vecSub,
  matVec3,
} from "./transforms";
import { type FKResult } from "./kinematics";

/** The page's accent color, so a rebrand is one CSS edit (style.css --accent). */
function cssAccent(): number {
  const v = typeof getComputedStyle === "function"
    ? getComputedStyle(document.documentElement).getPropertyValue("--accent").trim()
    : "";
  return v ? new THREE.Color(v).getHex() : 0xf27317;
}

const COLORS = {
  dark: 0x33383f,
  orange: cssAccent(),
  steel: 0xc0c5c9,
  blue: 0x2673e6,
  tray: 0x8c6b47,
  floor1: 0x33363e,
  floor2: 0x3d4049,
  target: 0x34e07a,
};

const CUBE_HALF = 0.02;
const GRASP_RADIUS = 0.055; // tcp-to-cube-center distance for a valid grasp
const GRASP_CLOSED_APERTURE = 0.25; // grip opening below which a grasp engages
const HELD_MAX_SPEED = 4.0; // m/s the servo may use to catch up to the tool
const HELD_MAX_ANGVEL = 25.0; // rad/s
const RELEASE_MAX_SPEED = 1.5; // m/s carried over when the gripper opens
const HIGHLIGHT_INTENSITY = 0.45;
const PHYSICS_HZ = 120;
const CUBE_STARTS: Vec3[] = [
  [0.42, -0.12, CUBE_HALF],
  [0.5, 0.06, CUBE_HALF],
];
const CUBE_COLORS = [COLORS.orange, COLORS.blue];

interface Cube {
  body: RAPIER.RigidBody;
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
  held: boolean;
  /** Cube pose in the tool frame while held. */
  holdOffset?: { pos: Vec3; quat: Quat };
}

export class ArmScene {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private world!: RAPIER.World;
  private jointGroups: THREE.Group[] = [];
  private fingerL!: THREE.Mesh;
  private fingerR!: THREE.Mesh;
  private targetMarker!: THREE.Mesh;
  private floorRing!: THREE.Mesh;
  private dropLine!: THREE.Line;
  private cubes: Cube[] = [];
  private q = [0, 0, 0, 0, 0, 0];
  private grip = 1; // 0 closed .. 1 open
  private accumulator = 0;
  private prevF: FKResult | null = null;
  private candidate: Cube | null = null;
  private contextLost = false;

  static async create(canvas: HTMLCanvasElement): Promise<ArmScene> {
    await RAPIER.init();
    return new ArmScene(canvas);
  }

  private constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = new THREE.Color(0x16181d);
    this.scene.fog = new THREE.Fog(0x16181d, 2.5, 6);
    // Surface a dead GPU context as an error the render loop can report,
    // instead of silently drawing nothing forever.
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.contextLost = true;
    });

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 20);
    this.camera.up.set(0, 0, 1); // z-up world
    this.camera.position.set(1.35, -0.85, 0.75);
    this.camera.lookAt(0.3, 0, 0.25);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444455, 1.1);
    this.scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 2.0);
    dir.position.set(0.5, -0.6, 1.6);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    // Tight shadow frustum around the workspace: the default ±5 m box
    // spends most of the shadow map on empty floor.
    const sc = dir.shadow.camera;
    sc.left = -1.2; sc.right = 1.2; sc.top = 1.2; sc.bottom = -1.2;
    sc.near = 0.1; sc.far = 4;
    dir.shadow.bias = -0.0005;
    this.scene.add(dir);

    this.world = new RAPIER.World({ x: 0, y: 0, z: -9.81 });
    this.world.timestep = 1 / PHYSICS_HZ;
    this.buildFloor();
    this.buildArm();
    this.buildObjects();
    this.buildMarkers();
  }

  // ---------------- scene construction ----------------

  private buildFloor(): void {
    // One instanced mesh per checker color: 2 draw calls instead of 256.
    const n = 8, size = 0.5;
    const geom = new THREE.PlaneGeometry(size, size);
    const mats = [COLORS.floor1, COLORS.floor2].map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.9 }),
    );
    const tiles = mats.map((mat) => {
      const inst = new THREE.InstancedMesh(geom, mat, 2 * n * n);
      inst.receiveShadow = true;
      return inst;
    });
    const counts = [0, 0];
    const m = new THREE.Matrix4();
    for (let i = -n; i < n; i++) {
      for (let j = -n; j < n; j++) {
        const which = (i + j) % 2 ? 0 : 1;
        m.setPosition((i + 0.5) * size, (j + 0.5) * size, 0);
        tiles[which].setMatrixAt(counts[which]++, m);
      }
    }
    this.scene.add(...tiles);
    this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(4, 4, 0.1).setTranslation(0, 0, -0.1).setFriction(1.0),
    );
  }

  private box(w: number, d: number, h: number, color: number): THREE.Mesh {
    const m = new THREE.Mesh(
      new THREE.BoxGeometry(w, d, h),
      new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.15 }),
    );
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  private cylinder(radius: number, height: number, color: number): THREE.Mesh {
    const m = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, height, 32),
      new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.15 }),
    );
    m.rotation.x = Math.PI / 2; // three cylinders are y-axis; we need z-axis
    m.castShadow = true;
    m.receiveShadow = true;
    return m;
  }

  /** Mirror the MJCF: nested groups, one per joint, meshes as in scene.xml. */
  private buildArm(): void {
    const base = new THREE.Group();
    const baseMesh = this.cylinder(0.07, 0.1, COLORS.dark);
    baseMesh.position.z = 0.05;
    base.add(baseMesh);
    this.scene.add(base);

    const linkSpecs: ((g: THREE.Group) => void)[] = [
      (g) => {
        const m = this.cylinder(0.055, 0.08, COLORS.orange);
        m.position.z = 0.04;
        g.add(m);
      },
      (g) => {
        const m = this.box(0.09, 0.07, 0.3, COLORS.steel);
        m.position.z = 0.15;
        g.add(m);
      },
      (g) => {
        const m = this.box(0.075, 0.06, 0.25, COLORS.orange);
        m.position.z = 0.125;
        g.add(m);
      },
      (g) => {
        const m = this.cylinder(0.045, 0.06, COLORS.steel);
        m.position.z = 0.03;
        g.add(m);
      },
      (g) => {
        const m = this.cylinder(0.04, 0.06, COLORS.orange);
        m.position.z = 0.03;
        g.add(m);
      },
      (g) => {
        const palm = this.box(0.08, 0.05, 0.04, COLORS.dark);
        palm.position.z = 0.02;
        g.add(palm);
        this.fingerL = this.box(0.02, 0.012, 0.08, COLORS.steel);
        this.fingerR = this.box(0.02, 0.012, 0.08, COLORS.steel);
        g.add(this.fingerL, this.fingerR);
      },
    ];

    let parent: THREE.Object3D = base;
    for (let i = 0; i < CHAIN.length; i++) {
      const g = new THREE.Group();
      const [ox, oy, oz] = CHAIN[i].offset;
      g.position.set(ox, oy, oz);
      linkSpecs[i](g);
      parent.add(g);
      this.jointGroups.push(g);
      parent = g;
    }
  }

  private buildObjects(): void {
    const trayCenter: Vec3 = [0.36, 0.26, 0];
    const trayParts: [Vec3, Vec3][] = [
      // [half extents, local position]
      [[0.09, 0.09, 0.006], [0, 0, 0.006]],
      [[0.09, 0.006, 0.025], [0, 0.084, 0.025]],
      [[0.09, 0.006, 0.025], [0, -0.084, 0.025]],
      [[0.006, 0.09, 0.025], [0.084, 0, 0.025]],
      [[0.006, 0.09, 0.025], [-0.084, 0, 0.025]],
    ];
    for (const [half, local] of trayParts) {
      const mesh = this.box(half[0] * 2, half[1] * 2, half[2] * 2, COLORS.tray);
      mesh.position.set(
        trayCenter[0] + local[0],
        trayCenter[1] + local[1],
        trayCenter[2] + local[2],
      );
      this.scene.add(mesh);
      this.world.createCollider(
        RAPIER.ColliderDesc.cuboid(...half)
          .setTranslation(mesh.position.x, mesh.position.y, mesh.position.z)
          .setFriction(1.0),
      );
    }

    CUBE_STARTS.forEach((pos, i) => this.spawnCube(pos, CUBE_COLORS[i]));
  }

  private spawnCube(pos: Vec3, color: number): void {
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(...pos).setCcdEnabled(true),
    );
    this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(CUBE_HALF, CUBE_HALF, CUBE_HALF)
        .setDensity(750)
        .setFriction(1.0),
      body,
    );
    const mesh = this.box(CUBE_HALF * 2, CUBE_HALF * 2, CUBE_HALF * 2, color);
    const material = mesh.material as THREE.MeshStandardMaterial;
    material.emissive.setHex(color);
    material.emissiveIntensity = 0;
    this.scene.add(mesh);
    this.cubes.push({ body, mesh, material, held: false });
  }

  private buildMarkers(): void {
    this.targetMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.015),
      new THREE.MeshBasicMaterial({ color: COLORS.target, transparent: true, opacity: 0.6 }),
    );
    this.scene.add(this.targetMarker);

    // Depth cue: a ring on the floor directly under the tool plus a drop
    // line up to it, so the operator can judge x/y without a second view.
    this.floorRing = new THREE.Mesh(
      new THREE.RingGeometry(0.018, 0.026, 32),
      new THREE.MeshBasicMaterial({
        color: COLORS.target, transparent: true, opacity: 0.7, depthWrite: false,
      }),
    );
    this.scene.add(this.floorRing);
    this.dropLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(), new THREE.Vector3(),
      ]),
      new THREE.LineBasicMaterial({
        color: COLORS.target, transparent: true, opacity: 0.35, depthWrite: false,
      }),
    );
    this.dropLine.frustumCulled = false;
    this.scene.add(this.dropLine);
  }

  // ---------------- commands (same interface as the Python ArmSim) ----------------

  setJointTargets(q: number[]): void {
    this.q = [...q];
  }

  /** opening in [0, 1]: 0 = fully closed, 1 = fully open. */
  setGripper(opening: number): void {
    this.grip = Math.min(Math.max(opening, 0), 1);
  }

  setTargetMarker(pos: Vec3): void {
    this.targetMarker.position.set(...pos);
  }

  reset(): void {
    // Reopen the gripper, or a closed grip re-grasps a freshly reset cube
    // that spawns within GRASP_RADIUS on the next step.
    this.grip = 1;
    this.cubes.forEach((cube, i) => {
      this.release(cube, false);
      cube.body.setTranslation(
        { x: CUBE_STARTS[i][0], y: CUBE_STARTS[i][1], z: CUBE_STARTS[i][2] },
        true,
      );
      cube.body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      cube.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      cube.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    });
  }

  /** Current cube centers, world frame, in spawn order. */
  cubePositions(): Vec3[] {
    return this.cubes.map((c) => {
      const t = c.body.translation();
      return [t.x, t.y, t.z];
    });
  }

  /** True if any cube is currently grasped. */
  get holding(): boolean {
    return this.cubes.some((c) => c.held);
  }

  /** True if the open gripper is over a cube it could grasp right now. */
  get canGrasp(): boolean {
    return this.candidate !== null;
  }

  // ---------------- stepping ----------------

  step(wallDt: number): void {
    // Arm pose (kinematic).
    for (let i = 0; i < 6; i++) {
      const g = this.jointGroups[i];
      g.rotation.set(0, 0, 0);
      if (CHAIN[i].axis === "z") g.rotation.z = this.q[i];
      else g.rotation.y = this.q[i];
    }
    const slide = 0.005 + this.grip * 0.035;
    this.fingerL.position.set(0, 0.008 + slide + 0.006, 0.08);
    this.fingerR.position.set(0, -0.008 - slide - 0.006, 0.08);

    const f = fk(this.q);
    const fPrev = this.prevF ?? f;
    this.updateGrasp(f);

    // Physics at a fixed rate with capped catch-up (same policy as ArmSim).
    // The held cube's servo target is interpolated across the substeps so
    // a frame's worth of arm motion is spread over them instead of landing
    // in the first one.
    this.accumulator = Math.min(this.accumulator + wallDt, 0.1);
    const h = 1 / PHYSICS_HZ;
    const n = Math.floor(this.accumulator / h);
    for (let i = 1; i <= n; i++) {
      this.driveHeld(fPrev, f, i / n, h);
      this.world.step();
    }
    this.accumulator -= n * h;
    this.prevF = f;

    for (const cube of this.cubes) {
      const t = cube.body.translation();
      const r = cube.body.rotation();
      cube.mesh.position.set(t.x, t.y, t.z);
      cube.mesh.quaternion.set(r.x, r.y, r.z, r.w);
    }

    this.floorRing.position.set(f.pos[0], f.pos[1], 0.002);
    const line = this.dropLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    line.setXYZ(0, f.pos[0], f.pos[1], 0.002);
    line.setXYZ(1, f.pos[0], f.pos[1], f.pos[2]);
    line.needsUpdate = true;
  }

  /** Servo held cubes toward their tool-frame pose for one substep. */
  private driveHeld(fPrev: FKResult, f: FKResult, alpha: number, h: number): void {
    for (const cube of this.cubes) {
      if (!cube.held || !cube.holdOffset) continue;
      const pos: Vec3 = [
        fPrev.pos[0] + (f.pos[0] - fPrev.pos[0]) * alpha,
        fPrev.pos[1] + (f.pos[1] - fPrev.pos[1]) * alpha,
        fPrev.pos[2] + (f.pos[2] - fPrev.pos[2]) * alpha,
      ];
      const quat = quatSlerp(fPrev.quat, f.quat, alpha);
      const want = vecAdd(pos, matVec3(matrixFromQuat(quat), cube.holdOffset.pos));
      const wantQuat = quatMultiply(quat, cube.holdOffset.quat);

      const t = cube.body.translation();
      let v: Vec3 = [(want[0] - t.x) / h, (want[1] - t.y) / h, (want[2] - t.z) / h];
      const speed = vecNorm(v);
      if (speed > HELD_MAX_SPEED) {
        const s = HELD_MAX_SPEED / speed;
        v = [v[0] * s, v[1] * s, v[2] * s];
      }
      cube.body.setLinvel({ x: v[0], y: v[1], z: v[2] }, true);

      const r = cube.body.rotation();
      const e = orientationError(wantQuat, [r.x, r.y, r.z, r.w]);
      let w: Vec3 = [e[0] / h, e[1] / h, e[2] / h];
      const rate = vecNorm(w);
      if (rate > HELD_MAX_ANGVEL) {
        const s = HELD_MAX_ANGVEL / rate;
        w = [w[0] * s, w[1] * s, w[2] * s];
      }
      cube.body.setAngvel({ x: w[0], y: w[1], z: w[2] }, true);
    }
  }

  private grasp(cube: Cube, f: FKResult, rel: Vec3): void {
    cube.held = true;
    // Store the cube pose in the tool frame.
    const localPos = matVec3(matTranspose3(f.rot), rel) as Vec3;
    const r = cube.body.rotation();
    const localQuat = quatMultiply(quatConjugate(f.quat), [r.x, r.y, r.z, r.w]);
    cube.holdOffset = { pos: localPos, quat: localQuat };
    cube.body.setGravityScale(0, true); // the servo carries it; no sag
    cube.material.emissiveIntensity = 0;
  }

  private release(cube: Cube, keepVelocity: boolean): void {
    if (!cube.held) return;
    cube.held = false;
    cube.holdOffset = undefined;
    cube.body.setGravityScale(1, true);
    // Drop, don't fling: cap the carried-over speed and shed the wrist's
    // spin, which the servo can leave at tens of rad/s.
    const lv = cube.body.linvel();
    let v: Vec3 = keepVelocity ? [lv.x, lv.y, lv.z] : [0, 0, 0];
    const speed = vecNorm(v);
    if (speed > RELEASE_MAX_SPEED) {
      const s = RELEASE_MAX_SPEED / speed;
      v = [v[0] * s, v[1] * s, v[2] * s];
    }
    cube.body.setLinvel({ x: v[0], y: v[1], z: v[2] }, true);
    cube.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }

  private updateGrasp(f: FKResult): void {
    const closed = this.grip < GRASP_CLOSED_APERTURE;
    // Nearest cube within reach of the tool, for the grasp and the glow.
    let nearest: Cube | null = null;
    let nearestRel: Vec3 = [0, 0, 0];
    let nearestDist = GRASP_RADIUS;
    for (const cube of this.cubes) {
      const t = cube.body.translation();
      const rel = vecSub([t.x, t.y, t.z], f.pos);
      const d = vecNorm(rel);
      if (d < nearestDist) {
        nearest = cube;
        nearestRel = rel;
        nearestDist = d;
      }
    }

    if (closed) {
      if (!this.holding && nearest) this.grasp(nearest, f, nearestRel);
    } else {
      for (const cube of this.cubes) this.release(cube, true);
    }

    // Glow the cube a pinch would pick up (open gripper, nothing held).
    const candidate = !closed && !this.holding ? nearest : null;
    if (candidate !== this.candidate) {
      if (this.candidate) this.candidate.material.emissiveIntensity = 0;
      if (candidate) candidate.material.emissiveIntensity = HIGHLIGHT_INTENSITY;
      this.candidate = candidate;
    }
  }

  render(width: number, height: number): void {
    if (this.contextLost) throw new Error("WebGL context lost");
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x !== width || size.y !== height) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.render(this.scene, this.camera);
  }

  /** Free GPU and physics resources so a fresh scene can take the canvas. */
  dispose(): void {
    this.renderer.dispose();
    this.world.free();
  }
}
