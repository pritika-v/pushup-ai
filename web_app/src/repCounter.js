import { getPhase } from './angleHeuristics.js';

// ─── Tuning constants ─────────────────────────────────────────────────────────

// Frames a phase must be held continuously before it's committed.
// Prevents single-frame jitter from flipping the state machine.
const PHASE_CONFIRM_FRAMES = 4;

// Minimum elbow bend required at the bottom of a rep to count as full depth.
// If the person never gets to 100° in the DOWN phase, rep is marked shallow.
const MIN_DEPTH_ANGLE = 100;

// Exponential moving average smoothing for elbow angle.
// Lower = more smoothing (more lag). Higher = more responsive (more jitter).
const EMA_ALPHA = 0.30;

// Consecutive in-position frames required before counting begins.
// Prevents counting during the walk-up / getting-into-position phase.
const POSITION_CONFIRM_FRAMES = 15;

// ─── RepCounter ───────────────────────────────────────────────────────────────
export class RepCounter {
  constructor() {
    this.reset();
  }

  reset() {
    this.phase         = 'UP';
    this.pendingPhase  = 'UP';
    this.pendingCount  = 0;

    this.goodReps = 0;
    this.badReps  = 0;

    this._repIssues        = [];
    this._repIsGood        = true;
    this._minElbowThisRep  = 180;

    this._smoothedAngle    = null;

    this._inPositionFrames = 0;
    this.isReady           = false;
  }

  // ── update() ─────────────────────────────────────────────────────────────
  // Call every frame.
  //   elbowAngle  – from analyzeFrame (number or null)
  //   isGoodForm  – boolean
  //   issues      – array of issue objects
  //   inPosition  – boolean from isInPushupPosition()
  //
  // Returns a rep event object on rep completion, or null.
  update(elbowAngle, isGoodForm, issues, inPosition) {

    // ── 1. Position gating ──────────────────────────────────────────────────
    if (!inPosition || elbowAngle === null) {
      // Decay the in-position counter (faster decay so we react quickly)
      this._inPositionFrames = Math.max(0, this._inPositionFrames - 3);
      if (this._inPositionFrames === 0) {
        this.isReady = false;
      }
      return null;
    }

    this._inPositionFrames = Math.min(this._inPositionFrames + 1, POSITION_CONFIRM_FRAMES);
    if (this._inPositionFrames >= POSITION_CONFIRM_FRAMES) {
      this.isReady = true;
    }

    if (!this.isReady) return null;

    // ── 2. Smooth the angle ──────────────────────────────────────────────────
    if (this._smoothedAngle === null) {
      this._smoothedAngle = elbowAngle;
    } else {
      this._smoothedAngle = EMA_ALPHA * elbowAngle + (1 - EMA_ALPHA) * this._smoothedAngle;
    }
    const angle = this._smoothedAngle;

    // ── 3. Track minimum angle in this rep ──────────────────────────────────
    if (angle < this._minElbowThisRep) {
      this._minElbowThisRep = angle;
    }

    // ── 4. Accumulate form issues (deduplicated, high+medium only) ───────────
    if (!isGoodForm) this._repIsGood = false;
    for (const issue of issues) {
      if (issue.severity !== 'low' && !this._repIssues.find(i => i.code === issue.code)) {
        this._repIssues.push(issue);
      }
    }

    // ── 5. Determine phase candidate ────────────────────────────────────────
    const rawPhase = getPhase(angle);
    // Only transition between UP and DOWN — MOVING keeps existing phase
    const candidatePhase = rawPhase === 'UP' ? 'UP' : rawPhase === 'DOWN' ? 'DOWN' : this.phase;

    // ── 6. Confirm phase transition ─────────────────────────────────────────
    if (candidatePhase !== this.pendingPhase) {
      this.pendingPhase = candidatePhase;
      this.pendingCount = 1;
    } else {
      this.pendingCount++;
    }

    let event = null;
    if (this.pendingCount >= PHASE_CONFIRM_FRAMES && candidatePhase !== this.phase) {
      const prev = this.phase;
      this.phase = candidatePhase;

      if (prev === 'UP' && this.phase === 'DOWN') {
        // Rep starting — reset per-rep tracking
        this._repIssues       = [];
        this._repIsGood       = true;
        this._minElbowThisRep = 180;
      }

      if (prev === 'DOWN' && this.phase === 'UP') {
        // Rep completed — evaluate
        event = this._finalizeRep();
      }
    }

    return event;
  }

  _finalizeRep() {
    // Check depth
    if (this._minElbowThisRep > MIN_DEPTH_ANGLE) {
      this._repIsGood = false;
      this._repIssues.push({
        code: 'SHALLOW_REP',
        message: `Shallow rep — lower until elbows reach ~90° (you reached ~${Math.round(this._minElbowThisRep)}°)`,
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
    this._repIssues       = [];
    this._repIsGood       = true;
    this._minElbowThisRep = 180;

    return event;
  }

  get smoothedAngle() { return this._smoothedAngle; }
}