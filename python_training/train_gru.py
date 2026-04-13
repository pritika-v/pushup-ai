import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

data = np.load('data/landmarks/good_pushup_sequences.npy')
N, T, F = data.shape

from sklearn.preprocessing import MinMaxScaler
data_reshaped = data.reshape(-1, F)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_reshaped).reshape(N, T, F)

split = int(0.9 * N)
X_train = data_scaled[:split]
X_val   = data_scaled[split:]

# ── GRU Autoencoder ───────────────────────────────────────────────────────────
inputs = tf.keras.Input(shape=(T, F))

enc = layers.GRU(128, activation='tanh', return_sequences=True, reset_after=False)(inputs)
enc = layers.Dropout(0.2)(enc)
enc =layers.GRU(64, activation='tanh', return_sequences=False, reset_after=False)(enc)
encoded = layers.Dense(32, activation='relu')(enc)

dec = layers.RepeatVector(T)(encoded)
dec = layers.GRU(64, activation='tanh', return_sequences=True, reset_after=False)(dec)
dec = layers.Dropout(0.2)(dec)
dec =layers.GRU(128, activation='tanh', return_sequences=True, reset_after=False)(dec)
outputs = layers.TimeDistributed(layers.Dense(F))(dec)

model = Model(inputs, outputs, name='gru_autoencoder')
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    tf.keras.callbacks.ModelCheckpoint('models/gru_best.keras', save_best_only=True)
]

model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=100, batch_size=32,
    callbacks=callbacks
)

X_pred = model.predict(X_train)
mse_per_seq = np.mean(np.mean((X_train - X_pred)**2, axis=2), axis=1)
threshold = float(np.mean(mse_per_seq) + 2 * np.std(mse_per_seq))
np.save('models/gru_threshold.npy', threshold)

# model.save('models/gru_autoencoder.keras')
#model.save('models/gru_saved_model', save_format='tf')  # ADD THIS
#print(f"GRU saved. Threshold: {threshold:.6f}")

model.save('models/gru_autoencoder.h5')  
    


print(f"GRU saved. Threshold: {threshold:.6f}")