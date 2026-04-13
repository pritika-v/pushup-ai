import subprocess, os, numpy as np

os.makedirs('../web_app/public/models/lstm', exist_ok=True)
os.makedirs('../web_app/public/models/gru', exist_ok=True)

# Convert LSTM
subprocess.run([
    'tensorflowjs_converter',
    '--input_format=keras',
    'models/lstm_autoencoder.keras',
    '../web_app/public/models/lstm'
], check=True)

# Convert GRU
subprocess.run([
    'tensorflowjs_converter',
    '--input_format=keras',
    'models/gru_autoencoder.keras',
    '../web_app/public/models/gru'
], check=True)

# Export thresholds and scaler as JSON for JS
lstm_thresh = float(np.load('models/lstm_threshold.npy'))
gru_thresh  = float(np.load('models/gru_threshold.npy'))
scaler_min  = np.load('data/landmarks/scaler_min.npy').tolist()
scaler_max  = np.load('data/landmarks/scaler_max.npy').tolist()

import json
config = {
    'lstm_threshold': lstm_thresh,
    'gru_threshold':  gru_thresh,
    'scaler_min': scaler_min,
    'scaler_max': scaler_max,
    'sequence_length': 30,
    'feature_count': 76
}
with open('../web_app/public/models/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Conversion complete. Config written.")