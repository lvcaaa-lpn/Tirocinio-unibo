from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import random
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


# DIVIDO GLI UTENTI (CLIENT)
all_users = df_filtered['user_id'].unique()
# Divido prima in training+validation (90%) e test (10%)
train_val_users, test_users = train_test_split(all_users, test_size=0.15, random_state=42)
# Divido il primo gruppo in training (80% del totale) e validation (10% del totale)
train_users, val_users = train_test_split(train_val_users, test_size=(0.10/0.85), random_state=42)

# Creo i tre set di dati
train_data_full = df_filtered[df_filtered['user_id'].isin(train_users)].copy()
val_data_full = df_filtered[df_filtered['user_id'].isin(val_users)].copy()
test_data_full = df_filtered[df_filtered['user_id'].isin(test_users)].copy()


def get_last_category_safely(category_list):
    """
    Estrae l'ultima categoria da una lista in modo sicuro.
    Se la lista non è valida o è vuota, restituisce 'Unknown'.
    """
    if isinstance(category_list, list) and category_list: # Controlla se è una lista e se non è vuota
        return category_list[-1]
    return 'Unknown' # Valore di default

# Applica la funzione sicura
train_data_full['specific_category'] = train_data_full['categories'].apply(get_last_category_safely)
test_data_full['specific_category'] = test_data_full['categories'].apply(get_last_category_safely)
val_data_full['specific_category'] = val_data_full['categories'].apply(get_last_category_safely)

user_favorite_price_tier = {}
for user in train_data_full['user_id'].unique():
    positive_reviews = train_data_full[(train_data_full['user_id'] == user) & (train_data_full['rating'] >= 4)]
    if not positive_reviews.empty:
        tier_counts = Counter(positive_reviews['price_tier'])
        if tier_counts:
            user_favorite_price_tier[user] = tier_counts.most_common(1)[0][0]



# Per ogni utente, troviamo la sua categoria specifica preferita
user_favorite_category = {}
for user in train_data_full['user_id'].unique():
    positive_reviews = train_data_full[(train_data_full['user_id'] == user) & (train_data_full['rating'] >= 4)]
    if not positive_reviews.empty:
        cat_counts = Counter(positive_reviews['specific_category'])
        if cat_counts:
            user_favorite_category[user] = cat_counts.most_common(1)[0][0]

# -------------------------------
# Costruzione dei profili utente - embedding, rappresentano il 'campo semantico' degli utenti
# -------------------------------

sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
EMBEDDING_DIM = sbert_model.get_sentence_embedding_dimension()
print(f"Dimensione embedding SBERT: {EMBEDDING_DIM}")

# sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# --- CALCOLO EMBEDDING PER IL TRAINING SET ---
print("Calcolo embedding delle recensioni per il Training Set...")
train_texts = train_data_full['review_text'].tolist()
train_review_embs = sbert_model.encode(train_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
train_data_full['review_emb'] = train_review_embs.tolist()

# --- CALCOLO EMBEDDING PER IL TEST SET ---
print("Calcolo embedding delle recensioni per il Test Set...")
test_texts = test_data_full['review_text'].tolist()
test_review_embs = sbert_model.encode(test_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
test_data_full['review_emb'] = test_review_embs.tolist()

# --- CALCOLO EMBEDDING PER IL VAL SET ---
print("Calcolo embedding delle recensioni per il Test Set...")
test_texts = val_data_full['review_text'].tolist()
val_review_embs = sbert_model.encode(test_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
val_data_full['review_emb'] = val_review_embs.tolist()


print("Costruzione profili utente...")

# Costruisco i due tipi di profili
positive_profiles = {}
negative_profiles = {}

# Usiamo il train_df per costruire i profili
users_in_train = train_data_full['user_id'].unique()

for user in users_in_train:
    # Profilo Positivo (come prima)
    positive_embs = train_data_full[
        (train_data_full['user_id'] == user) & (train_data_full['rating'] >= 4)
    ]['review_emb'].tolist()
    if positive_embs:
        positive_profiles[user] = np.mean(positive_embs, axis=0)

    # Profilo Negativo (la novità)
    negative_embs = train_data_full[
        (train_data_full['user_id'] == user) & (train_data_full['rating'] <= 2) # Scegli una soglia adatta
    ]['review_emb'].tolist()
    if negative_embs:
        negative_profiles[user] = np.mean(negative_embs, axis=0)

generic_fallback_embedding = np.zeros(EMBEDDING_DIM)

# Gestione degli utenti che non hanno recensioni negative (importante!)
all_negative_embs = train_data_full[train_data_full['rating'] <= 2]['review_emb'].tolist()
if all_negative_embs:
    generic_negative_profile = np.mean([np.array(e) for e in all_negative_embs], axis=0)
else:
    generic_negative_profile = generic_fallback_embedding

# Fallback anche per i positivi, sebbene meno probabile
all_positive_embs = train_data_full[train_data_full['rating'] >= 4]['review_emb'].tolist()
if all_positive_embs:
    generic_positive_profile = np.mean([np.array(e) for e in all_positive_embs], axis=0)
else:
    generic_positive_profile = generic_fallback_embedding

# -------------------------------
# Costruzione profili prodotto
# -------------------------------

print("Costruzione profili prodotto...")

train_data_full.loc[:,'description'] = train_data_full['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_data_full.loc[:,'product_text'] = train_data_full['title'].fillna('') + ' ' + train_data_full['description'].fillna('')# crea una nuova colonna nel df
                                                                                                            # title + description
product_texts = train_data_full.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato

df_filtered = df_filtered.merge(
    product_texts,  # contiene ['product_id', 'product_text']
    on='product_id',
    how='left'
)

print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT

product_profiles = dict(zip(product_texts['product_id'], product_texts['product_emb'])) # crea dizionario ID: embedding


# CREO I "DATA SILOS" DEI CLIENT DI TRAINING
# Il valore è il dataframe completo di quel client, già con gli embedding.
client_data_silos = {user_id: group for user_id, group in train_data_full.groupby('user_id')}

# -------------------------------
# Preparazione dati modello
# -------------------------------

def prepare_xy_v3(df, positive_profiles, negative_profiles, product_profiles,
                  user_fav_tier, user_fav_cat, generic_positive_profile, generic_negative_profile,
                  generic_fallback_emb):
    """
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
        dist = cosine(v1, v2)
        return 1 - np.clip(dist, 0.0, 2.0)

    for _, row in df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']

        # --- RECUPERO ROBUSTO DEGLI EMBEDDING ---
        # Se l'embedding del prodotto non esiste, usiamo un fallback di zeri
        prod_emb = np.array(product_profiles.get(product_id, generic_fallback_emb))

        # L'embedding della recensione dovrebbe sempre esistere, ma per sicurezza...
        review_emb = np.array(row['review_emb'])

        # L'embedding del profilo positivo dell'utente
        user_pos_emb = np.array(positive_profiles.get(user_id, generic_fallback_emb))

        # Il profilo positivo e negativo per il calcolo della similarità
        pos_profile = positive_profiles.get(user_id, generic_positive_profile)
        neg_profile = negative_profiles.get(user_id, generic_negative_profile)

        sim_to_positive = cosine_similarity(prod_emb, pos_profile)
        sim_to_negative = cosine_similarity(prod_emb, neg_profile)
        sim_difference = sim_to_positive - sim_to_negative

        user_pref_tier = user_fav_tier.get(user_id, 'None')
        product_price_tier = row['price_tier']

        user_pref_cat = user_fav_cat.get(user_id, 'None')
        # Gestisci il caso in cui 'specific_category' potrebbe mancare
        product_specific_category = row.get('specific_category', 'Unknown')

        price_match = 1 if product_price_tier == user_pref_tier else 0
        category_match = 1 if product_specific_category == user_pref_cat else 0

        # --- VETTORE DI INPUT ARRICCHITO (invariato) ---
        input_vect = np.concatenate([
            user_pos_emb,
            prod_emb,
            review_emb,
            [sim_to_positive],
            [sim_to_negative],
            [sim_difference],
            [price_match],
            [category_match]
        ])

        x.append(input_vect)
        y.append(1 if row['rating'] >= 4 else 0)

    # Converti la lista di vettori in un array 2D NumPy
    # Se x è vuoto, restituisci array vuoti con la forma corretta
    if not x:
        num_features = (EMBEDDING_DIM * 3) + 5  # 3 embedding + 5 features scalari
        return np.empty((0, num_features)), np.empty(0)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


print("\n--- Inizio Simulazione Apprendimento Federato con XGBoost ---")

# Parametri della simulazione
FEDERATED_ROUNDS = 5  # Quante volte fare il "giro" di tutti i client
CLIENTS_PER_ROUND = len(train_users)  # Per semplicità, usiamo tutti i client ad ogni round

# Deriviamo le etichette per l'intero training set da train_data_full
y_train_labels_for_weight = (train_data_full['rating'] >= 4).astype(int)

# Contiamo le classi positive (1) e negative (0)
count_neg = np.sum(y_train_labels_for_weight == 0)
count_pos = np.sum(y_train_labels_for_weight == 1)

# Calcoliamo il peso. Aggiungiamo un controllo per evitare la divisione per zero.
if count_pos > 0:
    calculated_scale_pos_weight = (count_neg / count_pos) * 1.5
else:
    calculated_scale_pos_weight = 1 # Valore di default se non ci sono campioni positivi

print(f"\033[93mDistribuzione classi nel training set: Negativi={count_neg}, Positivi={count_pos}\033[0m")
print(f"\033[93mCalcolato scale_pos_weight globale: {calculated_scale_pos_weight:.2f}\033[0m")

print("\n--- Inizio Simulazione Apprendimento Federato (Tree-by-Tree) con XGBoost ---")

# PREPARO TUTTI I DATI DI TRAINING IN ANTICIPO
# Questo è molto più efficiente che farlo ad ogni round per ogni client
print("Preparazione dei dati di training per tutti i client...")
all_client_features = []
all_client_labels = []

informativi_clients_count = 0
for user_id in train_users:
    client_df = client_data_silos.get(user_id)
    if client_df is None:
        continue

    # Preparo i dati per il singolo client.
    # Questo calcolo avviene sul dispositivo. I dati grezzi non lasciano mai il client.
    x_local, y_local = prepare_xy_v3(
        client_df,
        positive_profiles, negative_profiles, product_profiles,
        user_favorite_price_tier, user_favorite_category,
        generic_positive_profile, generic_negative_profile,
        generic_fallback_embedding
    )

    # Salva solo se il client ha dati informativi
    if len(x_local) > 0 and len(np.unique(y_local)) >= 2:
        all_client_features.append(x_local)
        all_client_labels.append(y_local)
        informativi_clients_count += 1

print(f"Dati preparati per {informativi_clients_count}/{len(train_users)} client informativi.")

# DEFINISCO I PARAMETRI DEL MODELLO
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,  # learning_rate
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'scale_pos_weight': calculated_scale_pos_weight,
    'random_state': 42,
    'gamma': 2,    # Richiede un miglioramento maggiore per fare uno split
    'lambda': 1.5, # Regolarizzazione L2
    'alpha': 0.5   # Regolarizzazione L1
}

# COMBINA I DATI E CREA UNA SOLA DMATRIX DI TRAINING
print("Combinazione dei dati in un'unica DMatrix di training...")
# Controlla se ci sono dati prima di procedere
if not all_client_features:
    raise ValueError("Nessun client con dati di training informativi trovato. Impossibile continuare.")

# Combina tutti i dati in un unico grande array NumPy. Questo simula l'invio delle statistiche dei client al server
full_train_x = np.vstack(all_client_features)
full_train_y = np.hstack(all_client_labels)

# Crea la DMatrix
dtrain = xgb.DMatrix(full_train_x, label=full_train_y)
print(f"DMatrix di training creata con {dtrain.num_row()} campioni e {dtrain.num_col()} feature.")

# LOOP DI TRAINING (ALBERO PER ALBERO)
NUM_TREES = 300
global_model = None

x_val, y_val = prepare_xy_v3(val_data_full,
    positive_profiles, negative_profiles, product_profiles,
    user_favorite_price_tier, user_favorite_category,
    generic_positive_profile, generic_negative_profile,
    generic_fallback_embedding)

dval = xgb.DMatrix(x_val, label=y_val)

# Definiamo la "watch list" per il monitoraggio
watchlist = [(dtrain, 'train'), (dval, 'eval')]

print(f"Inizio addestramento di {NUM_TREES} alberi...")
# Con l'API Core, possiamo addestrare tutti gli alberi in una sola chiamata.
# Il loop manuale non è necessario, a meno che non si vogliano fare operazioni complesse
# tra un albero e l'altro. Per il nostro caso, una singola chiamata è più efficiente.

global_model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,  # Addestra tutti gli alberi
    evals=watchlist,
    early_stopping_rounds=30,  # Se l'AUC su 'eval' non migliora per 30 round, fermati
    verbose_eval=10  # Stampa l'AUC ogni 10 alberi
)

print("\n\n\033[92mAddestramento Federato (Simulato) Completato\033[0m")

# VALUTAZIONE FINALE SUL TEST SET (UTENTI MAI VISTI)
print("\n\033[94mValutazione sul Test Set (utenti mai visti prima)\033[0m")

# Prepara i dati di test
x_test, y_test = prepare_xy_v3(
    test_data_full,
    positive_profiles, negative_profiles, product_profiles,
    user_favorite_price_tier, user_favorite_category,
    generic_positive_profile, generic_negative_profile,
    generic_fallback_embedding
)

# Converti i dati di test in una DMatrix, che è il formato richiesto dal modello Booster
dtest = xgb.DMatrix(x_test, label=y_test)

# Usa direttamente il modello Booster (`global_model`) per la predizione.
# global_model.predict(dtest) restituisce le probabilità grezze per la classe positiva.
y_pred_proba_final = global_model.predict(dtest)

# Per ottenere le predizioni di classe (0 o 1), applichiamo una soglia di 0.5
y_pred_final = (y_pred_proba_final > 0.5).astype(int)

# Ora possiamo usare y_test, y_pred_final, e y_pred_proba_final per il report
from sklearn.metrics import classification_report, roc_auc_score

print("\n--- Risultati sul Test Set ---")
print(classification_report(y_test, y_pred_final))
print(f"AUC sul Test Set: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
