# Recommender system
## Introduzione
In questo progetto si simula un sistema di raccomandazione di prodotti Amazon.  
Per realizzarlo mi sono basato sul dataset 'https://amazon-reviews-2023.github.io/', contenente diverse recensioni di prodotti scritte da molteplici utenti, compreso di immagini e descrizione dei prodotti.

## Come funziona
Banalmente il dataset contenente le recensioni degli utenti è composto dalle seguenti colonne:
['rating' 'title' 'text' 'images' 'asin' 'parent_asin' 'user_id' 'timestamp' 'helpful_vote' 'verified_purchase']

```python
reviews = pd.read_json("All_Beauty.jsonl", lines=True)
print(reviews.columns.values)
```
Output: 
```bash
 ['reviewerID' 'asin' 'reviewerName' 'helpful' 'reviewText' 'overall' 'summary' 'unixReviewTime' 'reviewTime'] 
``` 

Le colonne utili ai fini della raccomandazione sono quelle che rappresentano l'ID utente, l'ID dei prodotti, la recensione e il voto (stelle), ovvero:

```bash
['user_id', 'asin', 'text', 'rating']
```

Lo stesso approccio si applica al dataset dei prodotti (il cosiddetto meta), dal quale si vanno ad strapolare le colonne

```bash
['parent_asin', 'title', 'description']
```  

---

Il dataset utilizzato contiene migliaia di recensioni, rendendo il lavoro computazione molto complesso:

```python
print("lunghezza df: ", len(df))
print("utenti: ", len(df['user_id'].unique()))
```
Output:
```bash
lunghezza df:  640340
utenti:  578813

```

É pertanto necessario snellire i dati, filtrando gli utenti più attivi:

```python
user_review_counts = df['user_id'].value_counts()

active_users = user_review_counts[user_review_counts >= 5].index
df_filtered = df[df['user_id'].isin(active_users)]
```

---

### SBERT - SentenceTransformers
Il prossimo passaggio è realizzare delle variabili che rappresentino il significato degli utenti e dei prodotti, utilizzando recensioni e descrizioni.  
Per realizzare ciò si sfrutta un modello preallenato, denominato SBERT — Sentence Transformers.  
Dalla documentazione ufficiale (https://sbert.net)

> Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models (quickstart). This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

SBERT serve quindi a creare degli embedding che rappresentano il significato semantico di una frase.  
Ciò che si vuole ottenere sono dei dati x = [user_emb | product_emb] e y = 0/1 a seconda del rating (1 se rating>=4, 0 altrimenti).

```python
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
x = []
y = []

for _, row in df_filtered.iterrows():
    user_emb = user_profiles.get(row['user_id'])
    prod_emb = product_profiles.get(row['product_id'])

    if user_emb is not None and prod_emb is not None:
        input_vect = np.concatenate([user_emb, prod_emb])
        x.append(input_vect)
        y.append(1 if row['rating'] >= 4 else 0)
```

> [!NOTE]  
> Il modello di sbert utilizzato — *paraphrase-MiniLM-L3-v2* — potrebbe essere sostituito da un modello più preciso, ovvero *paraphrase-MPNet-base-v2*, considerato uno dei migliori modelli SBERT in termini di performance semantica e accuratezza.  
Tuttavia il grande carico computazionale derivato non permette il suo utilizzo (quasi 2 ore di attesa solo per il calcolo degli embedding degli utenti!).  
> 
>Una strategia potrebbe essere quella di utilizzare il modeelo più pesante e salvare successivamente i dati, in modo da non doverli ricalcolare ogni volta.
>
> Tuttavia, per semplicitò, è stato scelto di utilizzare un modello più leggero, anche se meno performante.

---

### Divisione dati test/train
Dati i vettori x e y, bisogna costruire i dati di test e training.  
Sfruttando la libreria **sklearn** si suddividono i dati in:
- 75% per training
- 25% per test

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=104, stratify=y, shuffle=True)
```

---

### Creazione modello e addestramento
Per il modello si utilizza una MLP creata in questo modo:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(768,)), # 384 + 384 = 768 (da SBERT)
    tf.keras.layers.Dropout(0.3), # spegne 30% dei neuroni in questo layer a ogni epoch -> evita overfitting
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # spegne 20% dei neuroni in questo layer a ogni epoch -> evita overfitting
    #tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # output binario
])

model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
               loss = tf.keras.losses.BinaryCrossentropy(),
               metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
              )

model.fit(
    x_train, y_train,
    epochs=12,
    batch_size=32,
    # validation_split=0.2,
    verbose=1,
    class_weight=class_weights_dict
)
```

Si utilizza il parametro **AUC** — Area under the curve — che indica la capacità del modello di distinguere le classi (utile nella classificazione binaria).  
In pratica si ha un grafico in cui:
- sull’asse X → il **False Positive Rate** (FPR).  
  Misura quanti dei casi negativi reali il modello ha classificato erroneamente come positivi.
- sull’asse Y → il **True Positive Rate** (TPR), anche detto recall.   
Misura quanti dei casi positivi reali il modello ha identificato correttamente.

L'AUC è l'area sotto la **ROC curve** (Receiver Operating Characteristic).  
Un AUC superiore a 0.85 indica una buona accuratezza, anche se accuracy potrebbe indicare il contrario.

---

### Risultati
I risultati ottenuti allenando il modello sono i seguenti:

```bash
Epoch 1/12
312/312 [==============================] - 1s 2ms/step - loss: 0.6832 - accuracy: 0.5520 - auc: 0.5809
Epoch 2/12
312/312 [==============================] - 1s 2ms/step - loss: 0.6557 - accuracy: 0.6073 - auc: 0.6560
Epoch 3/12
312/312 [==============================] - 1s 2ms/step - loss: 0.6245 - accuracy: 0.6455 - auc: 0.7060
Epoch 4/12
312/312 [==============================] - 1s 2ms/step - loss: 0.5998 - accuracy: 0.6677 - auc: 0.7366
Epoch 5/12
312/312 [==============================] - 1s 2ms/step - loss: 0.5691 - accuracy: 0.6912 - auc: 0.7722
Epoch 6/12
312/312 [==============================] - 1s 2ms/step - loss: 0.5412 - accuracy: 0.7152 - auc: 0.7982
Epoch 7/12
312/312 [==============================] - 1s 2ms/step - loss: 0.5074 - accuracy: 0.7412 - auc: 0.8290
Epoch 8/12
312/312 [==============================] - 1s 2ms/step - loss: 0.4796 - accuracy: 0.7615 - auc: 0.8497
Epoch 9/12
312/312 [==============================] - 1s 2ms/step - loss: 0.4427 - accuracy: 0.7853 - auc: 0.8753
Epoch 10/12
312/312 [==============================] - 1s 2ms/step - loss: 0.4289 - accuracy: 0.7957 - auc: 0.8837
Epoch 11/12
312/312 [==============================] - 1s 2ms/step - loss: 0.3927 - accuracy: 0.8149 - auc: 0.9035
Epoch 12/12
312/312 [==============================] - 1s 2ms/step - loss: 0.3754 - accuracy: 0.8269 - auc: 0.9126
104/104 [==============================] - 0s 1ms/step - loss: 0.6089 - accuracy: 0.7359 - auc: 0.6966
Loss: 0.6089, Accuracy: 0.7359, AUC: 0.6966 
```

Si vede come il modello venga allenato, raggiungendo una loss di 0.6 con un'accuracy del 73%.