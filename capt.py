import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Étape 1 : Charger les données à partir du fichier CSV
data = pd.read_csv('data_Humid.csv')
data = data['_value'].values.reshape(-1, 1)

# Normalisation des données entre 0 et 1
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Étape 2 : Division des données
train_size = int(len(data) * 0.8)  # 80% pour l'entraînement, 20% pour le test
train_data = data[:train_size]
test_data = data[train_size:]

# Étape 3 : Création des séquences d'entraînement
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length + 1])
    return np.array(sequences)

seq_length = 5  # Longueur de la séquence, ajustez-la en fonction de vos besoins
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

X_train = train_sequences[:, :-1]
y_train = train_sequences[:, -1]
X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

# Étape 4 : Création et entraînement du modèle LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=100, batch_size=16)

# Étape 5 : Prédiction du prochain mois
num_predictions = 5
last_sequence = train_data[-seq_length:]
predictions = []

for _ in range(num_predictions):
    last_sequence = last_sequence.reshape((1, seq_length, 1))
    pred = model.predict(last_sequence)
    predictions.append(pred[0, 0])
    last_sequence = np.concatenate((last_sequence[:, 1:, :], pred), axis=1)

# Inverser la normalisation pour obtenir les valeurs prédites en échelle réelle
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print("Prédictions pour le prochain mois : ", predictions)
