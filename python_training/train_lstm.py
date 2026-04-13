import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
import pickle, os

# Load data
data = np.load('data/landmarks/good_pushup_sequences.npy')  # (N, 30, 76)
N, T, F = data.shape
print(f"Dataset: {N} sequences, {T} timesteps, {F} features")

# Normalize per-feature across all sequences
data_reshaped = data.reshape(-1, F)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped).reshape(N, T, F)

# Save scaler for use in JS (we'll export min/max manually)
with open('data/landmarks/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
np.save('data/landmarks/scaler_min.npy', scaler.data_min_)
np.save('data/landmarks/scaler_max.npy', scaler.data_max_)

# Train/val split (90/10)
split = int(0.9 * N)
X_train = data_scaled[:split]
X_val   = data_scaled[split:]

# ── LSTM Autoencoder ──────────────────────────────────────────────────────────
inputs = tf.keras.Input(shape=(T, F))

# Encoder
enc = layers.LSTM(128, activation='tanh', return_sequences=True)(inputs)
enc = layers.Dropout(0.2)(enc)
enc = layers.LSTM(64, activation='tanh', return_sequences=False)(enc)
encoded = layers.Dense(32, activation='relu')(enc)  # bottleneck

# Decoder — RepeatVector brings back the time dimension
dec = layers.RepeatVector(T)(encoded)
dec = layers.LSTM(64, activation='tanh', return_sequences=True)(dec)
dec = layers.Dropout(0.2)(dec)
dec = layers.LSTM(128, activation='tanh', return_sequences=True)(dec)
outputs = layers.TimeDistributed(layers.Dense(F))(dec)

model = Model(inputs, outputs, name='lstm_autoencoder')
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.ModelCheckpoint('models/lstm_best.keras', save_best_only=True)
]

history = model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100, batch_size=32,
    callbacks=callbacks
)

# ── Compute anomaly threshold ─────────────────────────────────────────────────
# Reconstruction error on training set — threshold = mean + 2*std
X_pred = model.predict(X_train)
mse_per_seq = np.mean(np.mean((X_train - X_pred)**2, axis=2), axis=1)
threshold = float(np.mean(mse_per_seq) + 2 * np.std(mse_per_seq))
print(f"\nLSTM anomaly threshold: {threshold:.6f}")
np.save('models/lstm_threshold.npy', threshold)

model.save('models/lstm_autoencoder.keras')
print("LSTM model saved.")