import * as tf from '@tensorflow/tfjs';

export class PushupModel {
  constructor() {
    this.model          = null;
    this.sequenceBuffer = [];
    this.SEQ_LENGTH     = 30;
    this.FEATURES       = 68;
    // Raised threshold to reduce false anomalies from normal movement variation
    this.ANOMALY_THRESHOLD = 0.08;
  }

  async load() {
    try {
      this.model = await tf.loadLayersModel('/models/gru_model/model.json');
      console.log('✅ GRU model loaded');
    } catch (err) {
      console.warn('⚠️ GRU model not loaded — anomaly detection disabled:', err.message);
      this.model = null;
    }
  }

  // Only call when person IS in push-up position
  addFrame(frame) {
    if (frame.length !== this.FEATURES) return;
    this.sequenceBuffer.push(frame);
    if (this.sequenceBuffer.length > this.SEQ_LENGTH) {
      this.sequenceBuffer.shift();
    }
  }

  // Call when person leaves position — prevents stale frames poisoning scores
  clearBuffer() {
    this.sequenceBuffer = [];
  }

  async getAnomalyScore() {
    if (!this.model || this.sequenceBuffer.length < this.SEQ_LENGTH) return null;

    let input = null;
    let output = null;
    try {
      input  = tf.tensor([this.sequenceBuffer]);
      output = this.model.predict(input);
      const pred = await output.array();

      let error = 0;
      for (let t = 0; t < this.SEQ_LENGTH; t++) {
        for (let f = 0; f < this.FEATURES; f++) {
          const d = this.sequenceBuffer[t][f] - pred[0][t][f];
          error += d * d;
        }
      }
      error /= (this.SEQ_LENGTH * this.FEATURES);

      return { score: error, isAnomaly: error > this.ANOMALY_THRESHOLD };
    } catch (err) {
      console.warn('Anomaly score error:', err);
      return null;
    } finally {
      if (input)  input.dispose();
      if (output) output.dispose();
    }
  }
}