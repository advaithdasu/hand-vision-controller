/**
 * Signal smoothing — TypeScript port of handarm/filters.py.
 *
 * One Euro filter (Casiez et al., 2012): speed-adaptive cutoff — heavy
 * smoothing when the hand hovers, light smoothing during fast moves.
 */

import { type Quat, quatAngleBetween, quatNormalize, quatSlerp } from "./transforms";

function smoothingFactor(dt: number, cutoff: number): number {
  const r = 2 * Math.PI * cutoff * dt;
  return r / (r + 1);
}

export class OneEuroFilter {
  private x: number[] | null = null;
  private dx: number[] | null = null;

  constructor(
    private minCutoff = 1.0,
    private beta = 0.0,
    private dCutoff = 1.0,
  ) {}

  reset(): void {
    this.x = null;
    this.dx = null;
  }

  apply(x: number[], dt: number): number[] {
    if (this.x === null || this.dx === null) {
      this.x = [...x];
      this.dx = x.map(() => 0);
      return [...x];
    }
    // A zero/negative dt (coarsened clocks) must not wipe filter memory;
    // hold the previous estimate instead of passing raw jitter through.
    if (dt <= 0) return [...this.x];
    const aD = smoothingFactor(dt, this.dCutoff);
    let speedSq = 0;
    for (let i = 0; i < x.length; i++) {
      const dxi = (x[i] - this.x[i]) / dt;
      this.dx[i] = aD * dxi + (1 - aD) * this.dx[i];
      speedSq += this.dx[i] * this.dx[i];
    }
    const cutoff = this.minCutoff + this.beta * Math.sqrt(speedSq);
    const a = smoothingFactor(dt, cutoff);
    for (let i = 0; i < x.length; i++) {
      this.x[i] = a * x[i] + (1 - a) * this.x[i];
    }
    return [...this.x];
  }
}

/**
 * One Euro filter on orientation: slerp toward the target with a cutoff
 * that opens up with angular speed. beta = 0 degrades to a plain
 * first-order low-pass.
 */
export class QuaternionLowPass {
  private q: Quat | null = null;
  private rate = 0; // low-passed angular speed of the input, rad/s

  constructor(
    private minCutoff = 3.0,
    private beta = 0.0,
    private dCutoff = 1.0,
  ) {}

  reset(): void {
    this.q = null;
    this.rate = 0;
  }

  apply(q: Quat, dt: number): Quat {
    q = quatNormalize(q);
    if (this.q === null) {
      this.q = [...q];
      return [...q];
    }
    if (dt <= 0) return [...this.q];
    const aD = smoothingFactor(dt, this.dCutoff);
    this.rate = aD * (quatAngleBetween(this.q, q) / dt) + (1 - aD) * this.rate;
    const cutoff = this.minCutoff + this.beta * this.rate;
    this.q = quatSlerp(this.q, q, smoothingFactor(dt, cutoff));
    return [...this.q];
  }
}

/** Simple exponential moving average with a time-constant cutoff. */
export class ScalarLowPass {
  private y: number | null = null;

  constructor(private cutoff = 5.0) {}

  reset(): void {
    this.y = null;
  }

  apply(y: number, dt: number): number {
    if (this.y === null) {
      this.y = y;
      return y;
    }
    if (dt <= 0) return this.y;
    const a = smoothingFactor(dt, this.cutoff);
    this.y = a * y + (1 - a) * this.y;
    return this.y;
  }
}

/**
 * Dead zone that travels with the signal (a backlash / slop filter): the
 * output holds until the input strays more than `slop` from it, then
 * follows at that distance. Unlike a fixed dead zone around a neutral
 * value this suppresses small wobble anywhere in the range, which is
 * what an axis that picks up unintended cross-coupling needs; the cost
 * is `slop` of lag, and twice that to reverse direction.
 */
export class SlopFilter {
  private y: number | null = null;

  constructor(private slop = 0) {}

  reset(): void {
    this.y = null;
  }

  apply(x: number): number {
    if (this.y === null) {
      this.y = x;
      return x;
    }
    if (x > this.y + this.slop) this.y = x - this.slop;
    else if (x < this.y - this.slop) this.y = x + this.slop;
    return this.y;
  }
}
