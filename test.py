import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from keras.src.saving.legacy.saved_model.load import metrics
from tensorflow_probability import optimizer

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# mescolo i numpy
indices = np.random.permutation(len(x_train))

x_train = x_train[indices]
y_train = y_train[indices]

# divido
parts = 5 # N client
size_part = len(x_train) // parts

dataset = []
for i in range(parts):
    start = i * size_part
    end = start + size_part
    x_part = x_train[start:end]
    y_part = y_train[start:end]
    dataset.append((x_part, y_part))

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


# creo il dataset
tf_dataset = [
    tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y)
    ))
    .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
    for x, y in dataset
]

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    return model

def client_update(model, c_dataset, epochs=2):
    # alleno il modello per ogni client sul proprio dataset
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(c_dataset, epochs=epochs)

    return model.get_weights()

def server_update_weights(g_model, c_weights, ds_sizes):
    new_weights = []

    # Calcola la media pesata dei pesi
    total_size = sum(ds_sizes)
    weights_scaled = [size / total_size for size in ds_sizes] # n_k / n -- Assicura che i client con dataset più grandi abbiano un impatto maggiore nell'aggiornamento dei pesi globali.

    for weights_list_tuple in zip(*c_weights): # zip aggrega i pesi dello stesso layer
        weighted_avg = np.average(weights_list_tuple, axis=0, weights=weights_scaled) # media pesata
        new_weights.append(weighted_avg)

    g_model.set_weights(new_weights) # setto i nuovi valori dei pesi nel modello globale

    for i, layer_weights in enumerate(g_model.get_weights()):
        print(f"\033[94mPesi Layer {i} (primi 5 valori):", layer_weights.flatten()[:5], "\033[0m") # stampa in blu


# modello globale
global_model = create_model()

num_rounds = 10 # numero di round

"""for i, client_dataset in enumerate(tf_dataset):
    for batch_x, batch_y in client_dataset.take(1):  # primo batch
        print(f"Client {i} - First batch X shape: {batch_x.shape}, Y shape: {batch_y.shape}")
        print(f"Labels: {batch_y.numpy()}")
    print("-" * 50)

for i, (x, y) in enumerate(dataset):
    print(f"Client {i} - Classi uniche nei dati: {np.unique(y)}")"""

for round_num in range(num_rounds):
    print(f"\033[93mRound {round_num + 1} \033[0m") # stampa in giallo

    client_weights = [] # lista dei pesi restituiti dai client

    for client_data in tf_dataset:
        client_model = tf.keras.models.clone_model(global_model) # clona del modello globale
        client_model.set_weights(global_model.get_weights()) # inizializzo il modello client con i pesi del modello globale

        updated_weights = client_update(client_model, client_data)

        client_weights.append(updated_weights)

    # server aggiorna i pesi
    dataset_sizes = [len(x) for x, y in dataset]
    server_update_weights(global_model, client_weights, dataset_sizes)

    global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Valuta il modello globale dopo il round
    loss, accuracy = global_model.evaluate(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32))
    print(f"\033[92mServer Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} \033[0m") # stampa in verde