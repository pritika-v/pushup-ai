import * as tf from '@tensorflow/tfjs';

let model;

export async function loadModel() {
  model = await tf.loadLayersModel('/models/gru_model/model.json');
  console.log("✅ Model loaded");
}

export function predict(sequence) {
  const input = tf.tensor([sequence]); // shape [1, 30, 68]
  const output = model.predict(input);
  return output;
}