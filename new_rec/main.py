# prova con focal_loss al posto di smote
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Reshape, Dense, Dropout
import collections
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from scipy.spatial.distance import cosine
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import GridSearchCV




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

meta = meta[['parent_asin', 'title', 'description', 'price', 'categories']]
meta.columns = ['product_id', 'title', 'description', 'price', 'categories']

df = pd.merge(reviews, meta, on='product_id', how='inner')
df['price_tier'] = pd.qcut(df['price'], q=4, labels=['low', 'medium', 'high', 'luxury'], duplicates='drop')
# print(df.columns.values)
# ⮕ ['user_id' 'product_id' 'review_text' 'rating' 'title' 'description']

# print(meta.columns.values)
# print(df.head(3))

# print("lunghezza ds: ", len(df))
# print("utenti: ", len(df['user_id'].unique()))

user_review_counts = df['user_id'].value_counts()
# print(user_review_counts.head(20))

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

def get_last_category_safely(category_list):
    """
    Estrae l'ultima categoria da una lista in modo sicuro.
    Se la lista non è valida o è vuota, restituisce 'Unknown'.
    """
    if isinstance(category_list, list) and category_list: # Controlla se è una lista E se non è vuota
        return category_list[-1]
    return 'Unknown' # Valore di default

# Applica la funzione sicura
df_filtered['specific_category'] = df_filtered['categories'].apply(get_last_category_safely)


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


user_favorite_price_tier = {}
for user in train_df['user_id'].unique():
    positive_reviews = train_df[(train_df['user_id'] == user) & (train_df['rating'] >= 4)]
    if not positive_reviews.empty:
        tier_counts = Counter(positive_reviews['price_tier'])
        if tier_counts:
            user_favorite_price_tier[user] = tier_counts.most_common(1)[0][0]



# Ora, per ogni utente, troviamo la sua categoria specifica preferita

user_favorite_category = {}
for user in train_df['user_id'].unique():
    positive_reviews = train_df[(train_df['user_id'] == user) & (train_df['rating'] >= 4)]
    if not positive_reviews.empty:
        cat_counts = Counter(positive_reviews['specific_category'])
        if cat_counts:
            user_favorite_category[user] = cat_counts.most_common(1)[0][0]

# -------------------------------
# Costruzione dei profili utente - embedding, rappresentano il 'campo semantico' degli utenti
# -------------------------------

sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# --- CALCOLO EMBEDDING PER IL TRAINING SET ---
print("Calcolo embedding delle recensioni per il Training Set...")
train_texts = train_df['review_text'].tolist()
train_review_embs = sbert_model.encode(train_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
train_df['review_emb'] = train_review_embs.tolist()


# --- CALCOLO EMBEDDING PER IL VALIDATION SET ---
print("Calcolo embedding delle recensioni per il Validation Set...")
val_texts = val_df['review_text'].tolist()
val_review_embs = sbert_model.encode(val_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
val_df['review_emb'] = val_review_embs.tolist()


# --- CALCOLO EMBEDDING PER IL TEST SET ---
print("Calcolo embedding delle recensioni per il Test Set...")
test_texts = test_df['review_text'].tolist()
test_review_embs = sbert_model.encode(test_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
test_df['review_emb'] = test_review_embs.tolist()


print("Costruzione profili utente...")

#texts = train_df['review_text'].tolist() # lista recensioni
#review_embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True) # alleno SBERT

"""
user_profiles = {}
for user in train_df['user_id'].unique(): # per ogni utente nella lista unica degli utenti
    embs = train_df[(train_df['user_id'] == user)]['review_emb'].tolist() # prendo l'embedding dell'utente
                                                                                                               # in cui la recensione di di almeno
                                                                                                               # 4 stelle
    if embs:
        user_profiles[user] = np.mean(embs, axis=0) # media embedding per creare il profilo
"""

# Costruisci i due tipi di profili
positive_profiles = {}
negative_profiles = {}

# Usiamo il train_df per costruire i profili
users_in_train = train_df['user_id'].unique()

for user in users_in_train:
    # Profilo Positivo (come prima)
    positive_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] >= 4)
    ]['review_emb'].tolist()
    if positive_embs:
        positive_profiles[user] = np.mean(positive_embs, axis=0)

    # Profilo Negativo (la novità)
    negative_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] <= 2) # Scegli una soglia adatta
    ]['review_emb'].tolist()
    if negative_embs:
        negative_profiles[user] = np.mean(negative_embs, axis=0)

# Gestione degli utenti che non hanno recensioni negative (importante!)
# Creiamo un profilo "negativo generico" come media di tutti gli embedding negativi
all_negative_embs = train_df[train_df['rating'] <= 2]['review_emb'].tolist()
if all_negative_embs:
    generic_negative_profile = np.mean(all_negative_embs, axis=0)
else:
    # Fallback nel caso non ci siano proprio recensioni negative nel training set
    generic_negative_profile = np.zeros_like(review_embs[0])

# Fallback anche per i positivi, sebbene meno probabile
all_positive_embs = train_df[train_df['rating'] >= 4]['review_emb'].tolist()
generic_positive_profile = np.mean(all_positive_embs, axis=0)

# -------------------------------
# Costruzione profili prodotto
# -------------------------------

print("Costruzione profili prodotto...")
# df_filtered['product_text'] = df_filtered['title'].fillna('') + ' ' + df_filtered['description'].fillna('')

train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('')# crea una nuova colonna nel df
                                                                                                            # title + description
product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato

df_filtered = df_filtered.merge(
    product_texts,  # contiene ['product_id', 'product_text']
    on='product_id',
    how='left'      # o 'inner' se sei sicuro che tutti i product_id combaciano
)

print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT

product_profiles = dict(zip(product_texts['product_id'], product_texts['product_emb'])) # crea dizionario ID: embedding


# -------------------------------
# Preparazione dati modello
# -------------------------------

def prepare_xy_v3(df, positive_profiles, negative_profiles, product_profiles,
                  user_fav_tier, user_fav_cat, generic_positive_profile, generic_negative_profile):
    """
    Versione adattata che include feature strutturate (prezzo e categoria).

    Args:
        df (pd.DataFrame): Il DataFrame arricchito che contiene 'price_tier' e 'specific_category'.
        positive_profiles (dict): Dizionario dei profili positivi degli utenti.
        negative_profiles (dict): Dizionario dei profili negativi degli utenti.
        product_profiles (dict): Dizionario dei profili dei prodotti.
        user_fav_tier (dict): Dizionario della fascia di prezzo preferita per ogni utente.
        user_fav_cat (dict): Dizionario della categoria specifica preferita per ogni utente.
        generic_positive_profile (np.array): Fallback per il profilo positivo.
        generic_negative_profile (np.array): Fallback per il profilo negativo.
    """
    x = []
    y = []

    # Funzione helper per la similarità (invariata)
    def cosine_similarity(v1, v2):
        if v1 is None or v2 is None or np.all(v1 == 0) or np.all(v2 == 0):
            return 0
        # La distanza coseno è 1 - similarità. Per evitare errori con vettori identici, clippiamo.
        dist = cosine(v1, v2)
        return 1 - np.clip(dist, 0.0, 2.0)

    for _, row in df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']

        # Recupera l'embedding del prodotto
        prod_emb = product_profiles.get(product_id)
        review_emb = np.array(row['review_emb'])

        if prod_emb is not None:
            # --- PARTE SEMANTICA (invariata) ---
            pos_profile = positive_profiles.get(user_id, generic_positive_profile)
            neg_profile = negative_profiles.get(user_id, generic_negative_profile)

            sim_to_positive = cosine_similarity(prod_emb, pos_profile)
            sim_to_negative = cosine_similarity(prod_emb, neg_profile)
            sim_difference = sim_to_positive - sim_to_negative

            user_pos_emb = positive_profiles.get(user_id, np.zeros_like(prod_emb))

            # ### INIZIO MODIFICA ###
            # --- PARTE STRUTTURATA (nuova) ---

            # Recupera le preferenze dell'utente e le feature del prodotto
            user_pref_tier = user_fav_tier.get(user_id, 'None')
            product_price_tier = row['price_tier']

            user_pref_cat = user_fav_cat.get(user_id, 'None')
            product_specific_category = row['specific_category']

            # Calcola le feature di match
            price_match = 1 if product_price_tier == user_pref_tier else 0
            category_match = 1 if product_specific_category == user_pref_cat else 0

            # --- VETTORE DI INPUT ARRICCHITO ---
            input_vect = np.concatenate([
                user_pos_emb,  # Embedding di ciò che gli piace
                prod_emb,  # Embedding del prodotto target
                review_emb,
                [sim_to_positive],  # Feature semantica #1
                [sim_to_negative],  # Feature semantica #2
                [sim_difference],  # Feature semantica #3
                [price_match],  # NUOVA Feature strutturata #1
                [category_match]  # NUOVA Feature strutturata #2
            ])
            # ### FINE MODIFICA ###

            x.append(input_vect)
            y.append(1 if row['rating'] >= 4 else 0)

    return np.array(x), np.array(y)


# Usa la nuova funzione per creare i dati
x_train, y_train = prepare_xy_v3(
    train_df,
    positive_profiles, negative_profiles, product_profiles,
    user_favorite_price_tier, user_favorite_category,
    generic_positive_profile, generic_negative_profile
)

x_val, y_val = prepare_xy_v3(
    val_df,
    positive_profiles, negative_profiles, product_profiles,
    user_favorite_price_tier, user_favorite_category,
    generic_positive_profile, generic_negative_profile
)
x_test, y_test = prepare_xy_v3(
    test_df,
    positive_profiles, negative_profiles, product_profiles,
    user_favorite_price_tier, user_favorite_category,
    generic_positive_profile, generic_negative_profile
)

"""
smote = SMOTE(random_state=42)
x_train_res, y_train_res = SMOTETomek().fit_resample(x_train, y_train)


# Suddivido il nuovo x_train_res in train/val
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_res, y_train_res, test_size=0.2, random_state=42, stratify=y_train_res
)
"""
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)


print("Train distrib:", np.bincount(y_train.astype(int)))
print("Val distrib:", np.bincount(y_val.astype(int)))
"""
input_dim = x_train.shape[1]

# Modello più semplice e regolarizzato
model = tf.keras.models.Sequential([
    # Dimensione input corretta in base a x_train.shape[1]
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4), # Aumenta il dropout

    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4), # Aumenta il dropout

    tf.keras.layers.Dense(1, activation='sigmoid')
])


def focal_loss_corrected(gamma=2.0, alpha=0.25):  # alpha è il peso per la classe POSITIVA (maggioritaria)
    def loss(y_true, y_pred):
        # Assicurati che y_pred sia clippato per evitare log(0)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Calcolo della cross-entropia binaria
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Calcolo di p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        # Calcolo del fattore di modulazione (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # Calcolo del fattore di pesatura alpha
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        # Loss finale
        focal_loss_value = alpha_factor * modulating_factor * bce

        return tf.reduce_mean(focal_loss_value)

    return loss

model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss = focal_loss_corrected(gamma=2.0, alpha=0.20),
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
"""

# Per sicurezza, controlliamo la versione che hai installato
# print("Versione di XGBoost installata:", xgb.__version__)

# Usa i migliori parametri che hai trovato prima
best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}

# Ricorda il peso delle classi
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Inizializza il classificatore
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    **best_params
)

# Esegui la ricerca sul set di training
xgb_clf.fit(x_train, y_train)

y_pred_best = xgb_clf.predict(x_test)

print("\n--- Risultati XGBoost Ottimizzato ---")
print(classification_report(y_test, y_pred_best))

# 1. Salva il modello XGBoost addestrato
xgb_clf.save_model("xgb_model_final.json")

# 2. Salva i profili e le preferenze
with open("positive_profiles.pkl", "wb") as f:
    pickle.dump(positive_profiles, f)
with open("negative_profiles.pkl", "wb") as f:
    pickle.dump(negative_profiles, f)
with open("product_profiles.pkl", "wb") as f:
    pickle.dump(product_profiles, f)
with open("user_favorite_price_tier.pkl", "wb") as f:
    pickle.dump(user_favorite_price_tier, f)
with open("user_favorite_category.pkl", "wb") as f:
    pickle.dump(user_favorite_category, f)
with open("generic_positive_profile.pkl", "wb") as f:
    pickle.dump(generic_positive_profile, f)
with open("generic_negative_profile.pkl", "wb") as f:
    pickle.dump(generic_negative_profile, f)

# 3. Salva il DataFrame completo e arricchito per il test
# Uniamo train, val e test per avere un unico df di lookup
df_final_for_testing = pd.concat([train_df, val_df, test_df])
df_final_for_testing.to_pickle("df_final_for_testing.pkl")
