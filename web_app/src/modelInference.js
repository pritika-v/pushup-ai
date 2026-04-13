import * as tf from '@tensorflow/tfjs';

// ─── PushupModel ──────────────────────────────────────────────────────────────
// GRU-based autoencoder for push-up form anomaly detection.
// Only accumulates frames and scores when the person is actively in position.
export class PushupModel {
  constructor() {
    this.model = null;
    this.sequenceBuffer = [];
    this.SEQ_LENGTH = 30;   // frames needed before scoring
    this.FEATURES   = 68;   // 17 keypoints × 4 values

    // Anomaly threshold — tuned to avoid false positives from slight jitter.
    // Increase this value if you're getting too many false anomalies.
    this.ANOMALY_THRESHOLD = 0.08;
  }

  async load() {
    try {
      this.model = await tf.loadLayersModel('/models/gru_model/model.json');
      console.log('✅ GRU anomaly model loaded');
    } catch (err) {
      console.warn('⚠️ GRU model failed to load — anomaly detection disabled', err);
      this.model = null;
    }
  }

  // Add a single frame of keypoint data.
  // Only call this when the person IS in push-up position.
  addFrame(frame) {
    if (frame.length !== this.FEATURES) {
      console.warn(`Expected ${this.FEATURES} features, got ${frame.length}`);
      return;
    }
    this.sequenceBuffer.push(frame);
    if (this.sequenceBuffer.length > this.SEQ_LENGTH) {
      this.sequenceBuffer.shift();
    }
  }

  // Clear the buffer (call when person leaves push-up position)
  clearBuffer() {
    this.sequenceBuffer = [];
  }

  // Returns anomaly info object, or null if not enough data / model not loaded
  async getAnomalyScore() {
    if (!this.model || this.sequenceBuffer.length < this.SEQ_LENGTH) {
      return null;
    }

    let input;
    let output;
    try {
      input  = tf.tensor([this.sequenceBuffer]); // [1, SEQ_LENGTH, FEATURES]
      output = this.model.predict(input);

      const pred = await output.array();

      let error = 0;
      for (let t = 0; t < this.SEQ_LENGTH; t++) {
        for (let f = 0; f < this.FEATURES; f++) {
          const diff = this.sequenceBuffer[t][f] - pred[0][t][f];
          error += diff * diff;
        }
      }
      error /= (this.SEQ_LENGTH * this.FEATURES);

      return {
        score: error,
        isAnomaly: error > this.ANOMALY_THRESHOLD
      };
    } catch (err) {
      console.warn('Anomaly score computation failed:', err);
      return null;
    } finally {
      // Always dispose tensors to avoid memory leaks
      if (input)  input.dispose();
      if (output) output.dispose();
    }
  }
}