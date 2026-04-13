import { getPhase } from './angleHeuristics.js';

// ─── Constants ─────────────────────────────────────────────────────────────────
// How many consecutive frames a phase must be confirmed before transitioning.
// This prevents single-frame jitter from triggering phase changes.
const PHASE_CONFIRM_FRAMES = 4;

// Minimum elbow angle depth required to count a rep as "full depth".
// If the person never bends their elbows below this, it's a shallow rep.
const MIN_DEPTH_ANGLE = 100;

// Smoothing window for elbow angle (exponential moving average alpha)
const EMA_ALPHA = 0.25;

// How many consecutive frames of "in position" needed before counting begins
const POSITION_CONFIRM_FRAMES = 20;

// ─── RepCounter ────────────────────────────────────────────────────────────────
export class RepCounter {
  constructor() {
    this.reset();
  }

  reset() {
    // Phase state machine
    this.phase = 'UP';            // Current confirmed phase: UP | DOWN | MOVING
    this.pendingPhase = 'UP';     // Phase candidate being confirmed
    this.pendingCount = 0;        // Frames the pending phase has been seen

    // Rep tracking
    this.goodReps = 0;
    this.badReps  = 0;

    // Within-rep issue accumulation
    // Issues seen during the CURRENT rep (cleared after each rep)
    this._repIssues = [];
    this._repIsGood = true;

    // Did the person reach adequate depth in this rep?
    this._reachedDepth = false;

    // Minimum elbow angle seen in the current DOWN phase
    this._minElbowThisRep = 180;

    // Smoothed elbow angle (EMA)
    this._smoothedAngle = null;

    // Session readiness: person must be in position for N frames before counting
    this._inPositionFrames = 0;
    this.isReady = false;
  }

  // ── Main update ──────────────────────────────────────────────────────────────
  // Call this every frame with:
  //   elbowAngle   – raw angle from analyzeFrame (or null if landmarks not visible)
  //   isGoodForm   – boolean from analyzeFrame
  //   issues       – array of issue objects from analyzeFrame
  //   inPosition   – boolean: is the person in a push-up position?
  //
  // Returns a rep event object when a rep completes, otherwise null.
  update(elbowAngle, isGoodForm, issues, inPosition) {
    // ── 1. Position gating ───────────────────────────────────────────────────
    if (!inPosition || elbowAngle === null) {
      // Person not in position — reset readiness counter
      // But don't fully reset state so they can pause and resume
      this._inPositionFrames = Math.max(0, this._inPositionFrames - 2);
      if (this._inPositionFrames === 0) this.isReady = false;
      return null;
    }

    this._inPositionFrames = Math.min(this._inPositionFrames + 1, POSITION_CONFIRM_FRAMES);
    if (this._inPositionFrames >= POSITION_CONFIRM_FRAMES) {
      this.isReady = true;
    }

    if (!this.isReady) return null;

    // ── 2. Smooth the elbow angle ─────────────────────────────────────────────
    if (this._smoothedAngle === null) {
      this._smoothedAngle = elbowAngle;
    } else {
      this._smoothedAngle = EMA_ALPHA * elbowAngle + (1 - EMA_ALPHA) * this._smoothedAngle;
    }
    const angle = this._smoothedAngle;

    // ── 3. Accumulate issues during this rep ──────────────────────────────────
    // Only track high/medium severity to avoid noise from low-severity issues
    if (!isGoodForm) {
      this._repIsGood = false;
    }
    for (const issue of issues) {
      if ((issue.severity === 'high' || issue.severity === 'medium') &&
          !this._repIssues.find(i => i.code === issue.code)) {
        this._repIssues.push(issue);
      }
    }

    // ── 4. Track minimum angle reached ────────────────────────────────────────
    if (angle < this._minElbowThisRep) {
      this._minElbowThisRep = angle;
    }

    // ── 5. Phase candidate detection ──────────────────────────────────────────
    const rawPhase = getPhase(angle);
    let candidatePhase;

    if (rawPhase === 'UP') {
      candidatePhase = 'UP';
    } else if (rawPhase === 'DOWN') {
      candidatePhase = 'DOWN';
    } else {
      // MOVING — keep the last confirmed phase as candidate (don't interrupt)
      candidatePhase = this.phase;
    }

    // ── 6. Phase confirmation ──────────────────────────────────────────────────
    if (candidatePhase !== this.pendingPhase) {
      this.pendingPhase = candidatePhase;
      this.pendingCount = 1;
    } else {
      this.pendingCount++;
    }

    // Only commit phase transition when confirmed for N frames
    let event = null;
    if (this.pendingCount >= PHASE_CONFIRM_FRAMES && candidatePhase !== this.phase) {
      const prevPhase = this.phase;
      this.phase = candidatePhase;

      // ── 7. Rep completion (DOWN → UP) ────────────────────────────────────────
      if (prevPhase === 'DOWN' && this.phase === 'UP') {
        event = this._finalizeRep();
      }

      // ── 8. Rep start (UP → DOWN) — begin fresh issue tracking ────────────────
      if (prevPhase === 'UP' && this.phase === 'DOWN') {
        this._beginRep();
      }
    }

    return event;
  }

  // ── Begin a new rep (called when transitioning UP → DOWN) ──────────────────
  _beginRep() {
    this._repIssues    = [];
    this._repIsGood    = true;
    this._reachedDepth = false;
    this._minElbowThisRep = 180;
  }

  // ── Finalize a rep (called when transitioning DOWN → UP) ───────────────────
  _finalizeRep() {
    // Check range of motion: did the person go deep enough?
    if (this._minElbowThisRep > MIN_DEPTH_ANGLE) {
      this._repIsGood = false;
      this._repIssues.push({
        code: 'SHALLOW_REP',
        message: `Shallow rep — try to lower until your elbows reach ~90° (you reached ~${Math.round(this._minElbowThisRep)}°)`,
        severity: 'high'
      });
    }

    let event;
    if (this._repIsGood && this._repIssues.length === 0) {
      this.goodReps++;
      event = { type: 'GOOD_REP' };
    } else {
      this.badReps++;
      event = { type: 'BAD_REP', issues: [...this._repIssues] };
    }

    // Reset per-rep state
    this._repIssues    = [];
    this._repIsGood    = true;
    this._reachedDepth = false;
    this._minElbowThisRep = 180;

    return event;
  }

  // ── Convenience getters ────────────────────────────────────────────────────
  get smoothedAngle() { return this._smoothedAngle; }
}