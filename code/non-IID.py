import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from keras.src.saving.legacy.saved_model.load import metrics
from tensorflow_probability import optimizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def create_non_iid_clients(x, y, num_clients=5, shards_per_client=2):
    num_shards = num_clients * shards_per_client
    num_imgs_per_shard = len(y) // num_shards

    # Ordina gli indici per etichetta
    sorted_indices = np.argsort(y)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Crea gli shard
    shards = []
    for i in range(num_shards):
        start = i * num_imgs_per_shard
        end = start + num_imgs_per_shard
        shards.append((x_sorted[start:end], y_sorted[start:end]))

    # Assegna gli shard ai client
    all_indices = np.arange(num_shards)
    np.random.shuffle(all_indices)

    client_data = []
    for i in range(num_clients):
        client_shards = all_indices[i*shards_per_client : (i+1)*shards_per_client]
        x_client = np.concatenate([shards[s][0] for s in client_shards])
        y_client = np.concatenate([shards[s][1] for s in client_shards])
        client_data.append((x_client, y_client))

    return client_data


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# mescolo i numpy
indices = np.random.permutation(len(x_train))

parts = 5 # N client
dataset = create_non_iid_clients(x_train, y_train, num_clients=parts, shards_per_client=5)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


# creo il dataset
tf_dataset = [
    tf.data.Dataset.from_tensor_slices((
        tf.convert_to_tensor(x),
        tf.convert_to_tensor(y)
    ))
    .map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=len(x))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
    for x, y in dataset
]

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model

def client_update(model, c_dataset, epochs=2):
    # alleno il modello per ogni client sul proprio dataset
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
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

   # for i, layer_weights in enumerate(g_model.get_weights()):
    #    print(f"\033[94mPesi Layer {i} (primi 5 valori):", layer_weights.flatten()[:5], "\033[0m") # stampa in blu


# modello globale
global_model = create_model()

"""for i, client_dataset in enumerate(tf_dataset):
    for batch_x, batch_y in client_dataset.take(1):  # primo batch
        print(f"Client {i} - First batch X shape: {batch_x.shape}, Y shape: {batch_y.shape}")
        print(f"Labels: {batch_y.numpy()}")
    print("-" * 50)
"""
for i, (x, y) in enumerate(dataset):
    print(f"Client {i} - Classi uniche nei dati: {np.unique(y)}")

# disegno i grafici di distribuzione delle classi
num_classes = len(np.unique(y_train))  # numero di classi
client_class_counts = np.zeros((parts, num_classes), dtype=int)  # matrice distribuzione

# Popolo la matrice con il numero di immagini per ogni classe in ciascun client
for i, (_, y_client) in enumerate(dataset):
    class_counts = Counter(y_client)
    for cls in range(num_classes):
        client_class_counts[i, cls] = class_counts.get(cls, 0)

cmap = plt.get_cmap('RdYlGn')  # Colormap dal rosso al verde
norm = plt.Normalize(vmin=0, vmax=num_classes - 1)  # Normalizza i valori per il colormap

# Crea il grafico
fig, ax = plt.subplots(figsize=(12, 6))
left = np.zeros(parts)  # Inizializza le barre a zero

for cls in range(num_classes):
    color = cmap(norm(cls))  # Assegna un colore fisso alla classe (dal rosso al verde)

    # barre per ogni client
    for i in range(parts):
        client_count = client_class_counts[i, cls]
        ax.barh(
            y=i,
            width=client_count,
            left=left[i],
            color=color
        )
    # Aggiorna l'inizio della barra per la classe successiva
    left += client_class_counts[:, cls]

ax.set_yticks(np.arange(parts))
ax.set_yticklabels([f"Client {i}" for i in range(parts)])
ax.set_xlabel("Numero di immagini")
ax.set_title("Distribuzione delle classi per client")

# legenda per le classi
handles = []
for cls in range(num_classes):
    color = cmap(norm(cls))  # Colore per quella classe
    handles.append(plt.Rectangle((0, 0), 1, 1, color=color))  # Rappresenta la classe con un rettangolo
ax.legend(handles, [f"Classe {cls}" for cls in range(num_classes)], title="Classi", bbox_to_anchor=(1.05, 1),
          loc='upper left')

# Visualizza il grafico
plt.tight_layout()
plt.show()

rounds = []
accuracies = []

num_rounds = 10 # numero di round

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
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(normalize_img).batch(32)
    loss, accuracy = global_model.evaluate(test_dataset)

    rounds.append(round_num)
    accuracies.append(accuracy)

    print(f"\033[92mServer Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} \033[0m") # stampa in verde


# esporto i risultati in un file csv
df_iid = pd.DataFrame({
    'round': rounds,
    'accuracy': accuracies,
    'client': parts
})

df_iid.to_csv('non-iid_data_10_rnds.csv', index=False)
