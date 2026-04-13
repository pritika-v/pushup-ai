// ─── Utilities ────────────────────────────────────────────────────────────────

export function angle3Points(A, B, C) {
  // Returns angle at B in degrees
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const magAB = Math.hypot(AB.x, AB.y);
  const magCB = Math.hypot(CB.x, CB.y);
  if (magAB === 0 || magCB === 0) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (magAB * magCB)))) * (180 / Math.PI);
}

export function midpoint(A, B) {
  return { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2 };
}

// ─── Landmark visibility guard ────────────────────────────────────────────────
// Returns true if all required landmarks are visible enough to trust
export function landmarksVisible(landmarks, indices, threshold = 0.5) {
  return indices.every(i => landmarks[i] && landmarks[i].visibility >= threshold);
}

// ─── Push-up Position Detector ────────────────────────────────────────────────
// Returns true if the person appears to be in a plank/push-up position at all.
// This is used to gate the rep counter — don't count unless in position.
export function isInPushupPosition(landmarks) {
  // Require key landmarks visible
  const required = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  if (!landmarksVisible(landmarks, required, 0.45)) return false;

  // Both shoulders and hips must be below nose (person is horizontal / face-down)
  const nose = landmarks[0];
  const leftShoulder  = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip       = landmarks[23];
  const rightHip      = landmarks[24];

  // In MediaPipe, y=0 is top of frame, y=1 is bottom.
  // When lying horizontal the shoulder y and hip y should be fairly close.
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
  const hipY      = (leftHip.y + rightHip.y) / 2;

  // The vertical spread (y difference) between shoulders and hips should be small
  // when horizontal — less than 0.25 of frame height
  const verticalSpread = Math.abs(shoulderY - hipY);
  if (verticalSpread > 0.30) return false;

  // Shoulders should be roughly at or below mid-frame (not standing up)
  // and not at the very bottom (lying on back)
  if (shoulderY < 0.20 || shoulderY > 0.90) return false;

  // Nose should be near shoulder height (not towering above — indicates standing)
  if (Math.abs(nose.y - shoulderY) > 0.35) return false;

  return true;
}

// ─── Push-up Form Analyzer ────────────────────────────────────────────────────
// Analyzes a single frame of landmarks and returns angles + form issues.
// Issues are ONLY returned for problems that are clearly outside normal range —
// thresholds are intentionally generous to avoid false positives.
export function analyzeFrame(landmarks) {
  const issues = [];

  // Select dominant side based on combined visibility score
  const leftVis  = (landmarks[11]?.visibility ?? 0) + (landmarks[23]?.visibility ?? 0);
  const rightVis = (landmarks[12]?.visibility ?? 0) + (landmarks[24]?.visibility ?? 0);
  const side = leftVis >= rightVis ? 'left' : 'right';

  const shoulder = side === 'left' ? landmarks[11] : landmarks[12];
  const elbow    = side === 'left' ? landmarks[13] : landmarks[14];
  const wrist    = side === 'left' ? landmarks[15] : landmarks[16];
  const hip      = side === 'left' ? landmarks[23] : landmarks[24];
  const knee     = side === 'left' ? landmarks[25] : landmarks[26];
  const ankle    = side === 'left' ? landmarks[27] : landmarks[28];
  const ear      = side === 'left' ? landmarks[7]  : landmarks[8];

  // Guard: skip analysis if key landmarks are low-confidence
  const keySide = side === 'left' ? [11, 13, 15, 23, 25, 27] : [12, 14, 16, 24, 26, 28];
  if (!landmarksVisible(landmarks, keySide, 0.45)) {
    return { elbowAngle: null, backAngle: null, kneeAngle: null, issues: [], isGoodForm: true };
  }

  // ── Core angles ───────────────────────────────────────────────────────────
  const elbowAngle = angle3Points(shoulder, elbow, wrist);
  const backAngle  = angle3Points(shoulder, hip, ankle);
  const kneeAngle  = angle3Points(hip, knee, ankle);

  // ── 1. BODY ALIGNMENT (back/hips) ─────────────────────────────────────────
  // Ideal: shoulder-hip-ankle ≈ 180°. Allow generous range (145°+).
  // Distinguish hip sag vs hip raised using relative y position.
  if (backAngle < 145) {
    const shoulderAnkleY = (shoulder.y + ankle.y) / 2;
    if (hip.y > shoulderAnkleY + 0.04) {
      issues.push({
        code: 'HIP_SAG',
        message: 'Hips are sagging — squeeze your core and glutes to keep a straight line',
        severity: 'high'
      });
    } else if (hip.y < shoulderAnkleY - 0.06) {
      issues.push({
        code: 'HIP_RAISED',
        message: 'Hips are too high (pike position) — lower your hips to form a straight plank',
        severity: 'high'
      });
    }
  }

  // ── 2. ELBOW FLARE ─────────────────────────────────────────────────────────
  // Check if elbow deviates significantly from the shoulder-to-wrist midpoint.
  // Threshold widened to 0.12 to avoid false flare on shorter people / side angles.
  const elbowExpectedX = (shoulder.x + wrist.x) / 2;
  const flare = Math.abs(elbow.x - elbowExpectedX);
  if (flare > 0.12) {
    issues.push({
      code: 'ELBOW_FLARE',
      message: 'Elbows are flaring out — tuck them closer to your sides at ~45°',
      severity: 'medium'
    });
  }

  // ── 3. WRIST PLACEMENT ─────────────────────────────────────────────────────
  // Wrist should be roughly under the shoulder. Threshold: 0.18 (generous for different body sizes)
  const wristDeviation = Math.abs(wrist.x - shoulder.x);
  if (wristDeviation > 0.18) {
    if (wrist.x > shoulder.x) {
      issues.push({
        code: 'HANDS_TOO_FAR',
        message: 'Hands are too far forward — place them under your shoulders',
        severity: 'medium'
      });
    } else {
      issues.push({
        code: 'HANDS_TOO_BACK',
        message: 'Hands are too far back — move them forward under your shoulders',
        severity: 'medium'
      });
    }
  }

  // ── 4. LEG STRAIGHTNESS ────────────────────────────────────────────────────
  // Only flag if legs are noticeably bent (< 150°). Knees slightly soft are fine.
  if (kneeAngle < 150) {
    issues.push({
      code: 'BENT_KNEES',
      message: 'Legs are bent — straighten your knees fully',
      severity: 'medium'
    });
  }

  // ── 5. HEAD / NECK POSITION ────────────────────────────────────────────────
  // Ear should stay roughly in line with shoulder (neutral neck).
  // Only flag if deviation is significant.
  const neckAngle = angle3Points(ear, shoulder, hip);
  if (neckAngle < 145) {
    if (ear.y < shoulder.y - 0.10) {
      issues.push({
        code: 'HEAD_UP',
        message: 'Head is craning upward — keep your neck neutral, eyes slightly ahead',
        severity: 'low'
      });
    } else if (ear.y > shoulder.y + 0.06) {
      issues.push({
        code: 'HEAD_DOWN',
        message: 'Chin is tucking too far — keep a neutral neck position',
        severity: 'low'
      });
    }
  }

  // ── 6. SHOULDER SHRUG ──────────────────────────────────────────────────────
  // Shoulders creeping up toward ears — threshold tightened to 0.03 to avoid
  // false positives on different head sizes / camera angles
  const shoulderEarDist = Math.abs(ear.y - shoulder.y);
  if (shoulderEarDist < 0.03) {
    issues.push({
      code: 'SHOULDER_SHRUG',
      message: 'Shoulders are shrugged up — depress and retract your shoulder blades',
      severity: 'medium'
    });
  }

  // ── 7. LUMBAR SAG (lower back hyperextension) ──────────────────────────────
  // Project hip onto the shoulder-ankle line and measure how far it drops below.
  const dX = ankle.x - shoulder.x;
  if (Math.abs(dX) > 0.01) {
    const t = (hip.x - shoulder.x) / dX;
    const lineY = shoulder.y + t * (ankle.y - shoulder.y);
    const sagAmount = hip.y - lineY; // positive = hip below the line (sag)
    if (sagAmount > 0.07) {
      issues.push({
        code: 'LOWER_BACK_SAG',
        message: 'Lower back is sagging — engage your abs and glutes to maintain a plank',
        severity: 'high'
      });
    }
  }

  // ── 8. SHALLOW REP (incomplete range of motion) ────────────────────────────
  // Flagged in repCounter when the person never reaches adequate depth in a rep.
  // No per-frame flag here — that would spam the UI.

  return {
    elbowAngle,
    backAngle,
    kneeAngle,
    issues,
    isGoodForm: issues.filter(i => i.severity === 'high').length === 0
  };
}

// ─── Phase classifier ─────────────────────────────────────────────────────────
// UP   = arms extended (top of push-up)
// DOWN = arms bent at bottom
// MOVING = in between
export function getPhase(elbowAngle) {
  if (elbowAngle === null) return 'UNKNOWN';
  if (elbowAngle > 155) return 'UP';
  if (elbowAngle < 90)  return 'DOWN';
  return 'MOVING';
}