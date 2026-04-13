// ─── Utilities ────────────────────────────────────────────────────────────────

export function angle3Points(A, B, C) {
  const AB = { x: A.x - B.x, y: A.y - B.y };
  const CB = { x: C.x - B.x, y: C.y - B.y };
  const dot = AB.x * CB.x + AB.y * CB.y;
  const magAB = Math.hypot(AB.x, AB.y);
  const magCB = Math.hypot(CB.x, CB.y);
  if (magAB === 0 || magCB === 0) return null;
  return Math.acos(Math.max(-1, Math.min(1, dot / (magAB * magCB)))) * (180 / Math.PI);
}

export function midpoint(A, B) {
  return { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2 };
}

// ─── Visibility helper ────────────────────────────────────────────────────────
function vis(lm, idx, threshold = 0.3) {
  return lm[idx] && (lm[idx].visibility ?? 1) >= threshold;
}

// ─── Push-up Position Detection ───────────────────────────────────────────────
//
// KEY DESIGN DECISION: Do NOT use absolute y-coordinates to infer horizontal
// posture. Camera angle varies hugely across devices and setups (floor-level
// on mobile, desk-level on laptop, diagonal, etc.). Y-based checks that work
// for one camera angle break for another.
//
// Instead we check:
//  1. Core arm landmarks are visible enough to trust.
//  2. Elbow angle is within the push-up working range (50°–178°).
//     Outside this range the person is standing/sitting with arms in a
//     position that cannot produce push-up elbow angles.
//  3. Wrists are not way above the head (y check with a very loose threshold
//     to handle all camera angles).
//
// This approach works regardless of camera height or angle.
//
export function isInPushupPosition(landmarks) {
  // Need at least one arm's key joints visible
  const leftArmOk  = vis(landmarks, 11, 0.3) && vis(landmarks, 13, 0.3) && vis(landmarks, 15, 0.3);
  const rightArmOk = vis(landmarks, 12, 0.3) && vis(landmarks, 14, 0.3) && vis(landmarks, 16, 0.3);
  if (!leftArmOk && !rightArmOk) return false;

  // Pick the more visible side
  const leftScore  = (landmarks[11]?.visibility ?? 0) + (landmarks[13]?.visibility ?? 0) + (landmarks[15]?.visibility ?? 0);
  const rightScore = (landmarks[12]?.visibility ?? 0) + (landmarks[14]?.visibility ?? 0) + (landmarks[16]?.visibility ?? 0);
  const side = leftScore >= rightScore ? 'left' : 'right';

  const shoulder = side === 'left' ? landmarks[11] : landmarks[12];
  const elbow    = side === 'left' ? landmarks[13] : landmarks[14];
  const wrist    = side === 'left' ? landmarks[15] : landmarks[16];

  // Compute elbow angle
  const ea = angle3Points(shoulder, elbow, wrist);
  if (ea === null) return false;

  // Push-up elbow range: from deep bottom (~50°) to just short of full lock-out (178°)
  // This filters out: arms-at-side (180°), arms-overhead, random bent-arm poses
  if (ea < 48 || ea > 178) return false;

  // Wrist must not be dramatically above the shoulder.
  // In any push-up camera setup the wrist is near-floor-level.
  // Allow wrist to be up to 0.4 units above shoulder (very generous for
  // unusual camera angles), but reject if way higher.
  const yDiff = wrist.y - shoulder.y; // positive = wrist lower in frame (normal)
  if (yDiff < -0.40) return false;

  return true;
}

// ─── Frame Analyzer ───────────────────────────────────────────────────────────
export function analyzeFrame(landmarks) {
  const issues = [];

  // Pick dominant side
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

  // Need at least shoulder + elbow + wrist to compute meaningful angles
  const armIdx = side === 'left' ? [11, 13, 15] : [12, 14, 16];
  if (!armIdx.every(i => vis(landmarks, i, 0.3))) {
    return { elbowAngle: null, backAngle: null, kneeAngle: null, issues: [], isGoodForm: true };
  }

  const elbowAngle = angle3Points(shoulder, elbow, wrist);
  if (elbowAngle === null) {
    return { elbowAngle: null, backAngle: null, kneeAngle: null, issues: [], isGoodForm: true };
  }

  const hipVis   = vis(landmarks, side === 'left' ? 23 : 24, 0.25);
  const ankleVis = vis(landmarks, side === 'left' ? 27 : 28, 0.25);
  const kneeVis  = vis(landmarks, side === 'left' ? 25 : 26, 0.25);

  const backAngle  = (hipVis && ankleVis) ? angle3Points(shoulder, hip, ankle) : null;
  const kneeAngle  = (hipVis && kneeVis && ankleVis) ? angle3Points(hip, knee, ankle) : null;

  // ── 1. Body alignment ────────────────────────────────────────────────────
  if (backAngle !== null && backAngle < 145) {
    const midY = ankle ? (shoulder.y + ankle.y) / 2 : shoulder.y;
    if (hip.y > midY + 0.04) {
      issues.push({ code: 'HIP_SAG', message: 'Hips sagging — squeeze core and glutes to keep a straight line', severity: 'high' });
    } else if (hip.y < midY - 0.06) {
      issues.push({ code: 'HIP_RAISED', message: 'Hips too high (pike) — lower hips to form a straight plank', severity: 'high' });
    }
  }

  // ── 2. Elbow flare ───────────────────────────────────────────────────────
  const elbowExpectedX = (shoulder.x + wrist.x) / 2;
  if (Math.abs(elbow.x - elbowExpectedX) > 0.12) {
    issues.push({ code: 'ELBOW_FLARE', message: 'Elbows flaring out — keep elbows ~45° from torso', severity: 'medium' });
  }

  // ── 3. Bent knees ────────────────────────────────────────────────────────
  if (kneeAngle !== null && kneeAngle < 150) {
    issues.push({ code: 'BENT_KNEES', message: 'Legs not straight — fully extend knees', severity: 'medium' });
  }

  // ── 4. Head/neck ─────────────────────────────────────────────────────────
  const earIdx = side === 'left' ? 7 : 8;
  if (vis(landmarks, earIdx, 0.3) && hipVis) {
    const neckAngle = angle3Points(ear, shoulder, hip);
    if (neckAngle !== null && neckAngle < 140) {
      if (ear.y < shoulder.y - 0.10) {
        issues.push({ code: 'HEAD_UP', message: 'Head craning up — keep neck neutral', severity: 'low' });
      } else if (ear.y > shoulder.y + 0.06) {
        issues.push({ code: 'HEAD_DOWN', message: 'Chin tucking — maintain neutral neck position', severity: 'low' });
      }
    }
  }

  // ── 5. Lumbar sag ────────────────────────────────────────────────────────
  if (backAngle !== null && ankleVis) {
    const dX = ankle.x - shoulder.x;
    if (Math.abs(dX) > 0.01) {
      const t = (hip.x - shoulder.x) / dX;
      const lineY = shoulder.y + t * (ankle.y - shoulder.y);
      if (hip.y - lineY > 0.07) {
        issues.push({ code: 'LOWER_BACK_SAG', message: 'Lower back sagging — engage abs and glutes', severity: 'high' });
      }
    }
  }

  return {
    elbowAngle,
    backAngle,
    kneeAngle,
    issues,
    isGoodForm: issues.filter(i => i.severity === 'high').length === 0
  };
}

// ─── Phase classifier ─────────────────────────────────────────────────────────
export function getPhase(elbowAngle) {
  if (elbowAngle === null) return 'UNKNOWN';
  if (elbowAngle > 155) return 'UP';
  if (elbowAngle < 90)  return 'DOWN';
  return 'MOVING';
}