import * as tf from '@tensorflow/tfjs';
import { analyzeFrame, isInPushupPosition } from './angleHeuristics.js';
import { RepCounter } from './repCounter.js';
import { PushupModel } from './modelInference.js';

console.log('JS LOADED');

// 17 keypoints × 4 values = 68 features for GRU model
const KEYPOINT_INDICES = [
  11, 12, 13, 14, 15, 16,  // shoulders, elbows, wrists
  23, 24, 25, 26, 27, 28,  // hips, knees, ankles
  0,                        // nose
  7,  8,                    // ears
  9, 10                     // mouth corners
];

// Run GRU anomaly scoring every N frames (it's expensive)
const ANOMALY_INTERVAL = 10;

// ── State ─────────────────────────────────────────────────────────────────────
let repCounter  = null;
let gruModel    = null;
let frameCount  = 0;
let lastAnomaly = null;
let appRunning  = false;

// ── DOM refs (resolved after DOMContentLoaded) ────────────────────────────────
let elGoodReps, elBadReps, elElbow, elPhase, elAnomalyScore,
    elFeedback, elStatus, elStartBtn, elResetBtn;

function resolveDOM() {
  elGoodReps    = document.getElementById('good-reps');
  elBadReps     = document.getElementById('bad-reps');
  elElbow       = document.getElementById('elbow-angle');
  elPhase       = document.getElementById('phase');
  elAnomalyScore = document.getElementById('anomaly-score');
  elFeedback    = document.getElementById('form-feedback');
  elStatus      = document.getElementById('status');
  elStartBtn    = document.getElementById('start-btn');
  elResetBtn    = document.getElementById('reset-btn');
}

// ── App startup ───────────────────────────────────────────────────────────────
async function startApp() {
  if (appRunning) return;

  resolveDOM();

  elStartBtn.disabled    = true;
  elStartBtn.textContent = 'Starting…';
  if (elStatus) elStatus.textContent = '⏳ Requesting camera…';

  const video = document.getElementById('video');

  // Camera
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error('Camera error:', err);
    alert('Could not access camera. Check permissions and try again.');
    elStartBtn.disabled    = false;
    elStartBtn.textContent = 'Start';
    if (elStatus) elStatus.textContent = '❌ Camera access denied';
    return;
  }

  // Init counters / model
  repCounter  = new RepCounter();
  frameCount  = 0;
  lastAnomaly = null;

  gruModel = new PushupModel();
  await gruModel.load(); // non-fatal if it fails

  // MediaPipe Pose
  const pose = new window.Pose({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  });

  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults(onPoseResults);

  appRunning             = true;
  elStartBtn.textContent = 'Running';
  if (elStatus) {
    elStatus.textContent  = '📍 Get into push-up position';
    elStatus.className    = 'status-waiting';
  }

  // Main loop — requestAnimationFrame drives MediaPipe at display refresh rate
  async function loop() {
    if (!appRunning) return;
    await pose.send({ image: video });
    requestAnimationFrame(loop);
  }
  loop();
}

// ── Per-frame handler ─────────────────────────────────────────────────────────
async function onPoseResults(results) {
  if (!results.poseLandmarks) return;

  const landmarks = results.poseLandmarks;
  frameCount++;

  // 1. Position detection (camera-angle-agnostic)
  const inPosition = isInPushupPosition(landmarks);

  // 2. Angle + form analysis
  const analysis = analyzeFrame(landmarks);

  // 3. GRU model — only feed frames when in position
  if (inPosition && analysis.elbowAngle !== null) {
    const frameData = KEYPOINT_INDICES.flatMap(i => [
      landmarks[i]?.x          ?? 0,
      landmarks[i]?.y          ?? 0,
      landmarks[i]?.z          ?? 0,
      landmarks[i]?.visibility ?? 0
    ]);
    gruModel.addFrame(frameData);

    if (frameCount % ANOMALY_INTERVAL === 0) {
      lastAnomaly = await gruModel.getAnomalyScore();
    }
  } else {
    gruModel.clearBuffer();
    lastAnomaly = null;
  }

  // 4. Merge anomaly into form quality
  let isGood = analysis.isGoodForm;
  if (lastAnomaly?.isAnomaly) isGood = false;

  // 5. Rep counter
  const repEvent = repCounter.update(
    analysis.elbowAngle,
    isGood,
    analysis.issues,
    inPosition
  );

  // 6. Update UI
  updateUI(analysis, repEvent, inPosition);
}

// ── UI update ─────────────────────────────────────────────────────────────────
// UI is updated every frame for live data (angle, phase, status).
// Feedback text is only updated when a rep completes, to avoid flicker.
function updateUI(analysis, repEvent, inPosition) {
  if (!repCounter) return;

  // Live counters
  if (elGoodReps) elGoodReps.textContent = repCounter.goodReps;
  if (elBadReps)  elBadReps.textContent  = repCounter.badReps;

  // Elbow angle — show smoothed value when available
  const displayAngle = repCounter.smoothedAngle ?? analysis.elbowAngle;
  if (elElbow) {
    elElbow.textContent = displayAngle !== null ? Math.round(displayAngle) + '°' : '--';
  }

  // Phase label
  if (elPhase) {
    if (!inPosition) {
      elPhase.textContent = 'NOT IN POSITION';
    } else if (!repCounter.isReady) {
      elPhase.textContent = 'GET READY…';
    } else {
      elPhase.textContent = repCounter.phase;
    }
  }

  // Anomaly score
  if (elAnomalyScore) {
    elAnomalyScore.textContent = lastAnomaly ? lastAnomaly.score.toFixed(4) : '--';
  }

  // Status banner
  if (elStatus) {
    if (!inPosition) {
      elStatus.textContent = '📍 Get into push-up position to begin';
      elStatus.className   = 'status-waiting';
    } else if (!repCounter.isReady) {
      elStatus.textContent = '⏳ Hold position… getting ready';
      elStatus.className   = 'status-waiting';
    } else {
      elStatus.textContent = '🟢 Counting — go!';
      elStatus.className   = 'status-ready';
    }
  }

  // Feedback — only update on rep event (prevents constant flicker)
  if (repEvent && elFeedback) {
    if (repEvent.type === 'GOOD_REP') {
      elFeedback.innerHTML = '<span class="good">✓ Great rep!</span>';
    } else {
      const top = repEvent.issues
        .sort((a, b) => severityRank(b.severity) - severityRank(a.severity))
        .slice(0, 3)
        .map(i => `• ${i.message}`)
        .join('<br>');
      elFeedback.innerHTML = `<span class="bad">✗ Bad rep:</span><br>${top}`;
    }
  }
}

function severityRank(s) {
  return { high: 3, medium: 2, low: 1 }[s] ?? 0;
}

// ── Button wiring ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  resolveDOM();

  elStartBtn?.addEventListener('click', () => {
    console.log('Start clicked');
    startApp();
  });

  elResetBtn?.addEventListener('click', () => {
    if (!repCounter) return;
    repCounter.reset();
    lastAnomaly = null;
    if (elGoodReps)   elGoodReps.textContent  = '0';
    if (elBadReps)    elBadReps.textContent    = '0';
    if (elFeedback)   elFeedback.textContent   = '';
    if (elStatus)     elStatus.textContent     = '📍 Get into push-up position to begin';
    console.log('Reset');
  });
});