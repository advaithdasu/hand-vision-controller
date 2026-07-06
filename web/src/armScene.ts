/**
 * Three.js rendering + Rapier physics for the arm and pick-and-place scene.
 *
 * The arm itself is kinematic (driven by joint angles from IK, mirroring
 * the geometry in assets/scene.xml); the cubes are dynamic Rapier bodies.
 * Grasping is kinematic: when the gripper closes with a cube between the
 * fingers, the cube is attached to the tool frame until the gripper opens —
 * the standard approach for browser demos, where contact-friction grasping
 * is too unstable to be worth the tuning.
 *
 * Everything lives in a z-up world to match the robot-frame math.
 */

import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";

import { CHAIN, fk } from "./kinematics";
import {
  type Quat,
  quatMultiply,
  type Vec3,
  vecAdd,
  vecNorm,
  vecSub,
  matVec3,
} from "./transforms";

const COLORS = {
  dark: 0x33383f,
  orange: 0xf27317,
  steel: 0xc0c5c9,
  blue: 0x2673e6,
  tray: 0x8c6b47,
  floor1: 0x33363e,
  floor2: 0x3d4049,
  target: 0x34e07a,
};

const CUBE_HALF = 0.02;
const GRASP_RADIUS = 0.055; // tcp-to-cube-center distance for a valid grasp

interface Cube {
  body: RAPIER.RigidBody;
  mesh: THREE.Mesh;
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
  private cubes: Cube[] = [];
  private q = [0, 0, 0, 0, 0, 0];
  private grip = 1; // 0 closed .. 1 open
  private accumulator = 0;

  static async create(canvas: HTMLCanvasElement): Promise<ArmScene> {
    await RAPIER.init();
    return new ArmScene(canvas);
  }

  private constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.scene.background = new THREE.Color(0x16181d);

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
    this.scene.add(dir);

    this.world = new RAPIER.World({ x: 0, y: 0, z: -9.81 });
    this.buildFloor();
    this.buildArm();
    this.buildObjects();

    this.targetMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.015),
      new THREE.MeshBasicMaterial({ color: COLORS.target, transparent: true, opacity: 0.6 }),
    );
    this.scene.add(this.targetMarker);
  }

  // ---------------- scene construction ----------------

  private buildFloor(): void {
    const tiles = new THREE.Group();
    const n = 8, size = 0.5;
    for (let i = -n; i < n; i++) {
      for (let j = -n; j < n; j++) {
        const mat = new THREE.MeshStandardMaterial({
          color: (i + j) % 2 ? COLORS.floor1 : COLORS.floor2,
          roughness: 0.9,
        });
        const tile = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
        tile.position.set((i + 0.5) * size, (j + 0.5) * size, 0);
        tile.receiveShadow = true;
        tiles.add(tile);
      }
    }
    this.scene.add(tiles);
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

    this.spawnCube([0.42, -0.12, CUBE_HALF], COLORS.orange);
    this.spawnCube([0.5, 0.06, CUBE_HALF], COLORS.blue);
  }

  private spawnCube(pos: Vec3, color: number): void {
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(...pos),
    );
    this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(CUBE_HALF, CUBE_HALF, CUBE_HALF)
        .setDensity(750)
        .setFriction(1.0),
      body,
    );
    const mesh = this.box(CUBE_HALF * 2, CUBE_HALF * 2, CUBE_HALF * 2, color);
    this.scene.add(mesh);
    this.cubes.push({ body, mesh, held: false });
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
    const starts: Vec3[] = [
      [0.42, -0.12, CUBE_HALF],
      [0.5, 0.06, CUBE_HALF],
    ];
    this.cubes.forEach((cube, i) => {
      cube.held = false;
      cube.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
      cube.body.setTranslation(
        { x: starts[i][0], y: starts[i][1], z: starts[i][2] },
        true,
      );
      cube.body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
      cube.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      cube.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    });
  }

  /** True if any cube is currently grasped. */
  get holding(): boolean {
    return this.cubes.some((c) => c.held);
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

    this.updateGrasp();

    // Physics at a fixed 120 Hz, capped catch-up (same policy as ArmSim).
    this.accumulator = Math.min(this.accumulator + wallDt, 0.1);
    const h = 1 / 120;
    this.world.timestep = h;
    while (this.accumulator >= h) {
      this.world.step();
      this.accumulator -= h;
    }

    // Sync dynamic cube meshes; held cubes follow the tool frame.
    const f = fk(this.q);
    for (const cube of this.cubes) {
      if (cube.held && cube.holdOffset) {
        const worldPos = vecAdd(f.pos, matVec3(f.rot, cube.holdOffset.pos));
        const worldQuat = quatMultiply(f.quat, cube.holdOffset.quat);
        cube.body.setNextKinematicTranslation({
          x: worldPos[0], y: worldPos[1], z: worldPos[2],
        });
        cube.body.setNextKinematicRotation({
          x: worldQuat[0], y: worldQuat[1], z: worldQuat[2], w: worldQuat[3],
        });
        cube.mesh.position.set(...worldPos);
        cube.mesh.quaternion.set(...worldQuat);
      } else {
        const t = cube.body.translation();
        const r = cube.body.rotation();
        cube.mesh.position.set(t.x, t.y, t.z);
        cube.mesh.quaternion.set(r.x, r.y, r.z, r.w);
      }
    }
  }

  private updateGrasp(): void {
    const closed = this.grip < 0.25;
    const f = fk(this.q);
    if (closed && !this.holding) {
      for (const cube of this.cubes) {
        const t = cube.body.translation();
        const d = vecNorm(vecSub([t.x, t.y, t.z], f.pos));
        if (d < GRASP_RADIUS) {
          cube.held = true;
          // Store the cube pose in the tool frame.
          const rel = vecSub([t.x, t.y, t.z], f.pos);
          const rotT = f.rot[0].map((_, c) => f.rot.map((row) => row[c])); // transpose
          const localPos = matVec3(rotT, rel) as Vec3;
          const r = cube.body.rotation();
          const qInv: Quat = [-f.quat[0], -f.quat[1], -f.quat[2], f.quat[3]];
          const localQuat = quatMultiply(qInv, [r.x, r.y, r.z, r.w]);
          cube.holdOffset = { pos: localPos, quat: localQuat };
          cube.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
          break;
        }
      }
    } else if (!closed) {
      for (const cube of this.cubes) {
        if (cube.held) {
          cube.held = false;
          cube.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
          cube.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
        }
      }
    }
  }

  render(width: number, height: number): void {
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x !== width || size.y !== height) {
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.render(this.scene, this.camera);
  }
}
