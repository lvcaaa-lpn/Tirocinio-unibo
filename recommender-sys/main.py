import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

meta = meta[['parent_asin', 'title', 'description']]
meta.columns = ['product_id', 'title', 'description']

df = pd.merge(reviews, meta, on='product_id', how='inner')
# print(df.columns.values)
# ⮕ ['user_id' 'product_id' 'review_text' 'rating' 'title' 'description']

# print(meta.columns.values)
# print(df.head(3))

# print("lunghezza ds: ", len(df))
# print("utenti: ", len(df['user_id'].unique()))

user_review_counts = df['user_id'].value_counts()
# print(user_review_counts.head(20))

active_users = user_review_counts[user_review_counts >= 5].index
df_filtered = df[df['user_id'].isin(active_users)]

# print("lunghezza nuovo ds: ", len(df_filtered))

# -------------------------------
# Costruzione dei profili utente - embedding, rappresentano il 'campo semantico' degli utenti
# -------------------------------

sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

print("Costruzione profili utente...")

texts = df_filtered['review_text'].tolist() # lista recensioni
review_embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True) # alleno SBERT

df_filtered.loc[:,'review_emb'] = review_embs.tolist() # aggiungo lista embedding al df

user_profiles = {}
for user in df_filtered['user_id'].unique(): # per ogni utente nella lista unica degli utenti
    embs = df_filtered[(df_filtered['user_id'] == user) & (df_filtered['rating'] >= 4)]['review_emb'].tolist() # prendo l'embedding dell'utente
                                                                                                               # in cui la recensione di di almeno
                                                                                                               # 4 stelle
    if embs:
        user_profiles[user] = np.mean(embs, axis=0) # media embedding per creare il profilo

"""
# Salvataggio user_profiles
with open("user_profiles.pkl", "wb") as f:
    pickle.dump(user_profiles, f)
print(f"Profili utente salvati: {len(user_profiles)}")
"""

# -------------------------------
# Costruzione profili prodotto
# -------------------------------

print("Costruzione profili prodotto...")
# df_filtered['product_text'] = df_filtered['title'].fillna('') + ' ' + df_filtered['description'].fillna('')

df_filtered.loc[:,'description'] = df_filtered['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
df_filtered.loc[:,'product_text'] = df_filtered['title'].fillna('') + ' ' + df_filtered['description'].fillna('') # crea una nuova colonna nel df
                                                                                                            # title + description


product_texts = df_filtered.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                          # product_text associato

print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT

product_profiles = dict(zip(product_texts['product_id'], product_texts['product_emb'])) # crea dizionario ID: embedding

"""
with open("product_profiles.pkl", "wb") as f:
    pickle.dump(product_profiles, f)
print(f"Profili prodotto salvati: {len(product_profiles)}")
"""
# -------------------------------
# Salvataggio del DataFrame completo
# -------------------------------

# print("Salvataggio DataFrame completo con review_emb...")
# df.to_pickle("df_with_review_emb.pkl")

# -------------------------------
# Preparazione dati modello
# -------------------------------

x = []
y = []

for _, row in df_filtered.iterrows():
    user_emb = user_profiles.get(row['user_id'])
    prod_emb = product_profiles.get(row['product_id'])

    if user_emb is not None and prod_emb is not None:
        input_vect = np.concatenate([user_emb, prod_emb])
        x.append(input_vect)
        y.append(1 if row['rating'] >= 4 else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=104, stratify=y, shuffle=True)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(768,)), # 384 + 384 = 768 (da SBERT)
    tf.keras.layers.Dropout(0.3), # spegne 30% dei neuroni in questo layer a ogni epoch -> evita overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # spegne 20% dei neuroni in questo layer a ogni epoch -> evita overfitting
    tf.keras.layers.Dense(1, activation='sigmoid') # output binario
])

model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=['accuracy']
              )

model.fit(
    x_train, y_train,
    epochs=12,
    batch_size=32,
    #validation_split=0.2,
    verbose=1
)

loss, accuracy = model.evaluate(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32), verbose=1)

print(f"\033[92mLoss: {loss:.4f}, Accuracy: {accuracy:.4f} \033[0m")  # stampa in verde

