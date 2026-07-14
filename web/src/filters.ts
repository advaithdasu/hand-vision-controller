/**
 * Signal smoothing — TypeScript port of handarm/filters.py.
 *
 * One Euro filter (Casiez et al., 2012): speed-adaptive cutoff — heavy
 * smoothing when the hand hovers, light smoothing during fast moves.
 */

import { type Quat, quatNormalize, quatSlerp } from "./transforms";

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

/** First-order low-pass on orientation via slerp toward the target. */
export class QuaternionLowPass {
  private q: Quat | null = null;

  constructor(private cutoff = 3.0) {}

  reset(): void {
    this.q = null;
  }

  apply(q: Quat, dt: number): Quat {
    q = quatNormalize(q);
    if (this.q === null) {
      this.q = [...q];
      return [...q];
    }
    if (dt <= 0) return [...this.q];
    this.q = quatSlerp(this.q, q, smoothingFactor(dt, this.cutoff));
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
