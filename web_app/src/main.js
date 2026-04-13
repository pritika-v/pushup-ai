import * as tf from '@tensorflow/tfjs';
import { analyzeFrame, isInPushupPosition } from './angleHeuristics.js';
import { RepCounter } from './repCounter.js';
import { PushupModel } from './modelInference.js';

console.log("JS LOADED");

// ── Keypoint indices to extract for GRU model ─────────────────────────────────
// 17 keypoints × 4 values (x, y, z, visibility) = 68 features
const KEYPOINT_INDICES = [
  11, 12, 13, 14, 15, 16,   // shoulders, elbows, wrists
  23, 24, 25, 26, 27, 28,   // hips, knees, ankles
  0,                         // nose
  7, 8,                      // ears
  9, 10                      // mouth (left/right — used as face width proxy)
];

// ── Anomaly scoring throttle ───────────────────────────────────────────────────
// Don't run the GRU model every single frame — only every N frames.
const ANOMALY_SCORE_INTERVAL = 10;

// ── Module-level state ────────────────────────────────────────────────────────
let repCounter = null;
let gruModel   = null;
let frameCount = 0;       // total frames processed since start
let lastAnomaly = null;   // most recent anomaly result (cached between intervals)

// ─── App startup ──────────────────────────────────────────────────────────────
async function startApp() {
  // Disable the button while loading
  const btn = document.getElementById('start-btn');
  btn.disabled = true;
  btn.textContent = 'Starting…';

  const video = document.getElementById('video');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error('Camera access failed:', err);
    alert('Could not access your camera. Please allow camera permissions and try again.');
    btn.disabled = false;
    btn.textContent = 'Start';
    return;
  }

  // Init state
  repCounter = new RepCounter();
  frameCount = 0;
  lastAnomaly = null;

  // Load GRU model (non-fatal if it fails — anomaly detection is optional)
  gruModel = new PushupModel();
  await gruModel.load();

  // Init MediaPipe Pose
  const pose = new window.Pose({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  });

  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults(onPoseResults);

  btn.textContent = 'Running…';

  // Main loop
  async function loop() {
    await pose.send({ image: video });
    requestAnimationFrame(loop);
  }

  loop();
}

// ─── Per-frame processing ─────────────────────────────────────────────────────
async function onPoseResults(results) {
  if (!results.poseLandmarks) return;

  const landmarks = results.poseLandmarks;
  frameCount++;

  // ── 1. Check if person is in push-up position ─────────────────────────────
  const inPosition = isInPushupPosition(landmarks);

  // ── 2. Analyze form angles ────────────────────────────────────────────────
  const analysis = analyzeFrame(landmarks);

  // ── 3. Feed keypoints to GRU model (only when in position) ───────────────
  if (inPosition && analysis.elbowAngle !== null) {
    const frameData = KEYPOINT_INDICES.flatMap(i => [
      landmarks[i].x,
      landmarks[i].y,
      landmarks[i].z,
      landmarks[i].visibility
    ]);
    gruModel.addFrame(frameData);

    // Run anomaly scoring periodically (not every frame — expensive)
    if (frameCount % ANOMALY_SCORE_INTERVAL === 0) {
      lastAnomaly = await gruModel.getAnomalyScore();
    }
  } else {
    // Person not in position — clear GRU buffer so stale frames don't corrupt scoring
    gruModel.clearBuffer();
    lastAnomaly = null;
  }

  // ── 4. Override isGoodForm if anomaly detected ────────────────────────────
  let isGood = analysis.isGoodForm;
  if (lastAnomaly && lastAnomaly.isAnomaly) {
    isGood = false;
  }

  // ── 5. Update rep counter ──────────────────────────────────────────────────
  const repEvent = repCounter.update(
    analysis.elbowAngle,
    isGood,
    analysis.issues,
    inPosition
  );

  // ── 6. Update UI ───────────────────────────────────────────────────────────
  updateUI(analysis, lastAnomaly, repEvent, inPosition);
}

// ─── UI update ────────────────────────────────────────────────────────────────
function updateUI(analysis, anomaly, repEvent, inPosition) {
  // Rep counts
  document.getElementById('good-reps').textContent = repCounter.goodReps;
  document.getElementById('bad-reps').textContent  = repCounter.badReps;

  // Elbow angle — show smoothed value
  const displayAngle = repCounter.smoothedAngle ?? analysis.elbowAngle;
  document.getElementById('elbow-angle').textContent =
    displayAngle !== null ? Math.round(displayAngle) + '°' : '--';

  // Current phase
  document.getElementById('phase').textContent = inPosition
    ? (repCounter.isReady ? repCounter.phase : 'GET READY…')
    : 'NOT IN POSITION';

  // Anomaly score
  const anomalyEl = document.getElementById('anomaly-score');
  if (anomalyEl) {
    anomalyEl.textContent = anomaly ? anomaly.score.toFixed(4) : '--';
  }

  // ── Form feedback — ONLY update when a rep just completed ─────────────────
  // This avoids the constant flicker of per-frame feedback changes.
  if (repEvent) {
    const fb = document.getElementById('form-feedback');
    if (repEvent.type === 'GOOD_REP') {
      fb.innerHTML = '<span class="good">✓ Great rep! Keep it up.</span>';
    } else {
      const topIssues = repEvent.issues
        .sort((a, b) => severityRank(b.severity) - severityRank(a.severity))
        .slice(0, 3); // show top 3 issues max
      const msgs = topIssues.map(i => `• ${i.message}`).join('<br>');
      fb.innerHTML = `<span class="bad">✗ Rep counted with form issues:</span><br>${msgs}`;
    }
  }

  // ── Position / readiness status ───────────────────────────────────────────
  const statusEl = document.getElementById('status');
  if (statusEl) {
    if (!inPosition) {
      statusEl.textContent = '📍 Get into push-up position to begin';
      statusEl.className = 'status-waiting';
    } else if (!repCounter.isReady) {
      statusEl.textContent = '⏳ Hold position… getting ready';
      statusEl.className = 'status-waiting';
    } else {
      statusEl.textContent = '🟢 Ready — start your reps!';
      statusEl.className = 'status-ready';
    }
  }
}

// Helper: convert severity string to sort number
function severityRank(severity) {
  return { high: 3, medium: 2, low: 1 }[severity] ?? 0;
}

// ─── Button wiring ────────────────────────────────────────────────────────────
document.getElementById('start-btn').addEventListener('click', () => {
  console.log('Start button clicked');
  startApp();
});

// Optional: reset button
const resetBtn = document.getElementById('reset-btn');
if (resetBtn) {
  resetBtn.addEventListener('click', () => {
    if (repCounter) {
      repCounter.reset();
      document.getElementById('good-reps').textContent  = '0';
      document.getElementById('bad-reps').textContent   = '0';
      document.getElementById('form-feedback').textContent = '';
      console.log('Rep counter reset');
    }
  });
}