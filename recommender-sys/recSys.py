from collections import defaultdict

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Reshape, Dense, Dropout
import collections
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

pd.set_option('display.max_columns', None) # Mostra tutte le colonne
pd.set_option('display.width', None) # occupa tutta la larghezza della console
# pd.set_option('display.max_colwidth', None) # non impone un limite sulla larghezza delle colonne

reviews = pd.read_json("/home/luca/Università/Tirocinio/Prove Tesi/Dataset/All_Beauty.jsonl", lines=True)
meta = pd.read_json("/home/luca/Università/Tirocinio/Prove Tesi/Dataset/meta_All_Beauty.jsonl", lines=True)

# COMANDI UTILI PER ANALISI DATASET #
# --------------------------------- #
# df_sorted = df.sort_values(by="user_id") # ordina per user_id
# print(df_sorted.head(5))

# df_user_count = df_sorted.groupby('user_id').size().reset_index(name='n_ratings') # raggruppa per user_id, contando il numero di recensioni per user
# print(df_user_count.sort_values(by='n_ratings', ascending=False).head(20))

# print(df.columns.values) # stampa colonne dataset
# --------------------------------- #

# print(reviews.columns.values)
# ⮕ ['rating' 'title' 'text' 'images' 'asin' 'parent_asin' 'user_id' 'timestamp' 'helpful_vote' 'verified_purchase']
# print(meta.columns.values)
# ⮕ ['main_category' 'title' 'average_rating' 'rating_number' 'features' 'description' 'price' 'images' 'videos' 'store' 'categories' 'details'
#     'parent_asin' 'bought_together']

# snellisco i dataset e li unisco
reviews = reviews[['user_id', 'asin', 'text', 'rating']]
reviews.columns = ['user_id', 'product_id', 'review_text', 'rating']

meta = meta[['parent_asin', 'title', 'description', 'features', 'categories']]
meta.columns = ['product_id', 'title', 'description', 'features', 'categories']

df = pd.merge(reviews, meta, on='product_id', how='inner')

def build_full_text(row):
    # Concatenazione robusta con fallback su stringhe vuote se NaN
    parts = [
        str(row['review_text']),
        str(row['title']),
        str(row['description']),
        ' '.join(row['features']) if isinstance(row['features'], list) else str(row['features']),
        ' '.join(row['categories'][0]) if isinstance(row['categories'], list) and len(row['categories']) > 0 else str(row['categories']),
    ]
    return ' | '.join([p for p in parts if p.strip() != ''])  # filtra vuoti e concatena

df['full_text'] = df.apply(build_full_text, axis=1)

# print(df.columns.values)
# ⮕ ['user_id' 'product_id' 'review_text' 'rating' 'title' 'description']

# print(meta.columns.values)
# print(df.head(3))

# print("lunghezza ds: ", len(df))
# print("utenti: ", len(df['user_id'].unique()))

user_review_counts = df['user_id'].value_counts()
# print(user_review_counts.head(20))

def clean_text(x):
    if isinstance(x, list):
        return ' '.join(map(str, x))
    elif isinstance(x, dict):
        return ' '.join([f"{k} {v}" for k, v in x.items()])
    elif pd.isna(x):
        return ''
    else:
        return str(x)

# -------------------------------
# Filtro gli utenti più attivi (n. recensioni >= 5)
# -------------------------------

active_users = user_review_counts[user_review_counts >= 5].index
df_filtered = df[df['user_id'].isin(active_users)]

# -------------------------------
# Filtro i prodotti che n. recensioni >= 3
# -------------------------------

product_review_counts = df_filtered['product_id'].value_counts()
active_products = product_review_counts[product_review_counts >= 3].index
df_filtered = df_filtered[df_filtered['product_id'].isin(active_products)]
df_filtered = df_filtered[(df_filtered['title'].str.len() > 10)]

# print("lunghezza nuovo ds: ", len(df_filtered))

# -------------------------------
# DIVIDO IL DATASET PRIMA DI COSTRUIRE I PROFILI (PER EVITARE DATA LEAKAGE)
# -------------------------------

train_df, test_df = train_test_split(
    df_filtered, test_size=0.25, stratify=(df_filtered['rating'] >= 4), random_state=42
)

train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=(train_df['rating'] >= 4), random_state=42
)

# -------------------------------
# Costruzione dei profili utente - embedding, rappresentano il 'campo semantico' degli utenti
# -------------------------------

sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

print("Costruzione profili utente...")

texts = train_df['full_text'].tolist() # review + title + description + features + categories (riferiti ai prodotti che l'utente ha recensito)
review_embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True) # alleno SBERT

train_df.loc[:,'review_emb'] = review_embs.tolist() # aggiungo lista embedding al df


user_profiles = {}
for user in train_df['user_id'].unique(): # per ogni utente nella lista unica degli utenti
    embs = train_df[(train_df['user_id'] == user)]['review_emb'].tolist() # prendo l'embedding dell'utente
                                                                                                               # in cui la recensione di di almeno
                                                                                                               # 4 stelle
    if embs:
        user_profiles[user] = np.mean(embs, axis=0) # media embedding per creare il profilo

# -------------------------------
# Costruzione profili prodotto
# -------------------------------

print("Costruzione profili prodotto...")
# df_filtered['product_text'] = df_filtered['title'].fillna('') + ' ' + df_filtered['description'].fillna('')

train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('') + ' ' + train_df['features'].apply(clean_text) + ' ' +  train_df['categories'].apply(clean_text)# crea una nuova colonna nel df
                                                                                                            # title + description


product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato

print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT

product_profiles = dict(zip(product_texts['product_id'], product_texts['product_emb'])) # crea dizionario ID: embedding

# -------------------------------
# Preparazione dati modello
# -------------------------------

def prepare_xy(df, user_profiles, product_profiles):
    x = []
    y = []

    for _, row in df.iterrows():
        user_emb = user_profiles.get(row['user_id'])
        prod_emb = product_profiles.get(row['product_id'])

        if user_emb is not None and prod_emb is not None:
            diff = np.abs(user_emb - prod_emb)
            prod = user_emb * prod_emb

            input_vect = np.concatenate([user_emb, prod_emb, diff, prod])
            x.append(input_vect)
            y.append(1 if row['rating'] >= 4 else 0)
    return np.array(x), np.array(y)


x_train, y_train = prepare_xy(train_df, user_profiles, product_profiles)
x_val, y_val = prepare_xy(val_df, user_profiles, product_profiles)
x_test, y_test = prepare_xy(test_df, user_profiles, product_profiles)


y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)


print("Train distrib:", np.bincount(y_train.astype(int)))
print("Val distrib:", np.bincount(y_val.astype(int)))

model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(256, activation='relu', input_shape=(1536,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

def focal_loss(gamma=2., alpha=0.75):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * tf.pow(1. - pt, gamma) * bce
    return loss

model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss = focal_loss(), # tf.keras.losses.SquaredHinge(), tf.keras.losses.BinaryCrossentropy()
               metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
              )

early_stop = EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=25,
    batch_size=32,
    # validation_split=0.1,
    verbose=1,
    # class_weight=class_weights_dict,
    callbacks=[early_stop]
)

loss, accuracy, auc = model.evaluate(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32), verbose=1)

print(f"\033[92mLoss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f} \033[0m")  # stampa in verde

y_pred = model.predict(x_test).flatten()
y_pred_labels = (y_pred > 0.54).astype(int)
print(classification_report(y_test, y_pred_labels))

# Salva il modello
model.save("rec_model_focal_loss_4.h5")

# Salva i dizionari user_profiles e product_profiles
with open("user_profiles.pkl", "wb") as f:
    pickle.dump(user_profiles, f)

with open("product_profiles.pkl", "wb") as f:
    pickle.dump(product_profiles, f)

# Salva anche il DataFrame filtrato
df_filtered.to_pickle("df_filtered.pkl")

