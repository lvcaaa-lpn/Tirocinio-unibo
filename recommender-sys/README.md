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
['parent_asin', 'title', 'description', 'features', 'categories']
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

É pertanto necessario snellire i dati, filtrando gli utenti più attivi (ovvero gli utenti che hanno scritto 5+ recensioni) e i prodotti che sono stati recensiti più di 3 volte:

```python
user_review_counts = df['user_id'].value_counts()

active_users = user_review_counts[user_review_counts >= 5].index
df_filtered = df[df['user_id'].isin(active_users

product_review_counts = df_filtered['product_id'].value_counts()
active_products = product_review_counts[product_review_counts >= 3].index
df_filtered = df_filtered[df_filtered['product_id'].isin(active_products)]
df_filtered = df_filtered[(df_filtered['title'].str.len() > 10)] # Prendo i prodotti con un titolo di 10+ caratteri
```

---

### Divisione dataset
Adesso è importante dividere il dataset in dati di train e test.  
Viene fatto adesso per evitare Data Leakage, fenomeno che avviene quando il modello utilizza nel training informazioni a cui non dovrebbe avere accesso. In parole semplici: il modello "bara", raggiungendo performance elevatissime nel training e sbagliando nel test.

```python
train_df, test_df = train_test_split(
    df_filtered, test_size=0.25, stratify=(df_filtered['rating'] >= 4), random_state=42
)

train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=(train_df['rating'] >= 4), random_state=42
)
```
---

### SBERT - SentenceTransformers
Il prossimo passaggio è realizzare delle variabili che rappresentino il significato degli utenti e dei prodotti, utilizzando recensioni e descrizioni, combinate con le features e categories dei prodotti. 
Per realizzare ciò si sfrutta un modello preallenato, denominato SBERT — Sentence Transformers.  
Dalla documentazione ufficiale (https://sbert.net)

> Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models (quickstart) or to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models (quickstart). This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

SBERT serve quindi a creare degli embedding che rappresentano il significato semantico di una frase.  
Ciò che si vuole ottenere sono dei dati x = [user_emb | product_emb] e y = 0/1 a seconda del rating (1 se rating>=4, 0 altrimenti).

```python
sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

print("Costruzione profili utente...")

texts = train_df['full_text'].tolist() # review + title + description + features + categories (riferiti ai prodotti che l'utente ha recensito)
review_embs = sbert_model.encode(texts, batch_size=64, show_progress_bar=True) # alleno SBERT

train_df.loc[:,'review_emb'] = review_embs.tolist() # aggiungo lista embedding al df


user_profiles = {}
for user in train_df['user_id'].unique(): # per ogni utente nella lista unica degli utenti
    embs = train_df[(train_df['user_id'] == user)]['review_emb'].tolist() # prendo l'embedding dell'utente
                                                                                                              
    if embs:
        user_profiles[user] = np.mean(embs, axis=0) # media embedding per creare il profilo


print("Costruzione profili prodotto...")
train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('') + ' ' + train_df['features'].apply(clean_text) + ' ' +  train_df['categories'].apply(clean_text)# crea una nuova colonna nel df
                                                                                                            # title + description


product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato


print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT
```

> [!NOTE]  
> Il modello di sbert utilizzato — *paraphrase-MiniLM-L3-v2* — potrebbe essere sostituito da un modello più preciso, ovvero *paraphrase-MPNet-base-v2*, considerato uno dei migliori modelli SBERT in termini di performance semantica e accuratezza.  
Tuttavia il grande carico computazionale derivato non permette il suo utilizzo (quasi 2 ore di attesa solo per il calcolo degli embedding degli utenti!).  
> 
>Una strategia potrebbe essere quella di utilizzare il modeelo più pesante e salvare successivamente i dati, in modo da non doverli ricalcolare ogni volta.
>
> Tuttavia, per semplicitò, è stato scelto di utilizzare un modello più leggero, anche se meno performante.

---
### Preparazione dati per il modello
In questa sezione, creo una funzione per preparare gli input da dare "in pasto" al modello.

```python
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
```

---

### Utilizzo della focal loss
Il dataset che sto utilizzando è un po' sbilanciato. In particolare contiene pochi esempi di 0, quindi il modello fatica a classificare questa classe.  
Per gestire questa problematica, ho fatto ricorso alla Focal Loss.  
Essenzialmente si tratta di una modifica della binary cross-entropy (BCE), pensata per:
- gestire dataset sbilanciati,
- penalizzare meno gli esempi facili (che il modello classifica correttamente con alta confidenza),
- focalizzarsi sugli esempi difficili o meno rappresentati (lo 0 nel mio caso).

La formula della Focal Loss è:

$$
\text{FocalLoss}(p_t) = - \alpha (1 - p_t)^\gamma \log(p_t)
$$

Dove:

- $p_t$ è la probabilità predetta del vero target (es: `y_pred` se `y_true=1`, oppure `1 - y_pred` se `y_true=0`)
- $\alpha \in [0,1]$: bilancia il peso tra classi (es: più alta è più si focalizza sulla classe minoritaria)
- $\gamma \geq 0$: controlla il focus sugli esempi difficili

```python

def focal_loss(gamma=2., alpha=0.75):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * tf.pow(1. - pt, gamma) * bce
    return loss


```
### Creazione del modello con Early Stop
Oltre alla creazione del modello, introduciamo un Early Stop.  
Si tratta di una tecnica utilizzataper prevenire l'overfitting, cioè quando il modello apprende troppo bene i dati di training fino a non essere più in grado di generalizzare su nuovi dati.  
Quindi il modello andrà a valutare le proprie performance (monitor) ad ogni epoca, e quando nota che stanno peggiorando si blocca. Ovviamente tutto questo considerando un margine di epoche (patience).

Per il modello si utilizza una MLP creata in questo modo:

```python
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

model.compile( optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss = focal_loss(), # tf.keras.losses.SquaredHinge(), tf.keras.losses.BinaryCrossentropy()
               metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
              )

early_stop = EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True) # Early Stop su val_auc con patience=3

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

```

Si utilizza il parametro **AUC** — Area under the curve — che indica la capacità del modello di distinguere le classi (utile nella classificazione binaria).  
In pratica si ha un grafico in cui:
- sull’asse X → il **False Positive Rate** (FPR).  
  Misura quanti dei casi negativi reali il modello ha classificato erroneamente come positivi.
- sull’asse Y → il **True Positive Rate** (TPR), anche detto recall.   
Misura quanti dei casi positivi reali il modello ha identificato correttamente.

L'AUC è l'area sotto la **ROC curve** (Receiver Operating Characteristic).  
Un AUC superiore a 0.85 indica una buona accuratezza, anche se accuracy potrebbe indicare il contrario.

Oltre a questo parametro, utilizzo un `classification_report` per valutare le performance del modello.
I suoi valori sono:
- precision: di tutte le volte che il modello ha predetto una certa classe, quante volte aveva ragione?
- recall: di tutte le volte che la classe vera era X, quante volte l'ha trovata?
- f1-score: media armonica di precision e recall
- support: quanti esempi veri ci sono per ciascuna classe nel test set?
  
---

### Risultati
I risultati ottenuti allenando il modello sono i seguenti:

```bash
Epoch 1/25
145/145 [==============================] - 2s 6ms/step - loss: 0.2433 - accuracy: 0.6239 - auc: 0.5225 - val_loss: 0.1431 - val_accuracy: 0.3602 - val_auc: 0.5749
Epoch 2/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1861 - accuracy: 0.6886 - auc: 0.5609 - val_loss: 0.1097 - val_accuracy: 0.7270 - val_auc: 0.6174
Epoch 3/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1691 - accuracy: 0.7241 - auc: 0.5920 - val_loss: 0.0947 - val_accuracy: 0.8047 - val_auc: 0.6532
Epoch 4/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1514 - accuracy: 0.7295 - auc: 0.6116 - val_loss: 0.0898 - val_accuracy: 0.8180 - val_auc: 0.6879
Epoch 5/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1389 - accuracy: 0.7462 - auc: 0.6446 - val_loss: 0.0941 - val_accuracy: 0.8038 - val_auc: 0.6841
Epoch 6/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1319 - accuracy: 0.7464 - auc: 0.6621 - val_loss: 0.0943 - val_accuracy: 0.8114 - val_auc: 0.6892
Epoch 7/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1211 - accuracy: 0.7754 - auc: 0.6861 - val_loss: 0.0942 - val_accuracy: 0.8142 - val_auc: 0.6892
Epoch 8/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1243 - accuracy: 0.7721 - auc: 0.6747 - val_loss: 0.0951 - val_accuracy: 0.8104 - val_auc: 0.6972
Epoch 9/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1190 - accuracy: 0.7758 - auc: 0.6923 - val_loss: 0.0954 - val_accuracy: 0.8142 - val_auc: 0.6911
Epoch 10/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1166 - accuracy: 0.7730 - auc: 0.6966 - val_loss: 0.0923 - val_accuracy: 0.8237 - val_auc: 0.7007
Epoch 11/25
145/145 [==============================] - 1s 6ms/step - loss: 0.1088 - accuracy: 0.7942 - auc: 0.7216 - val_loss: 0.0939 - val_accuracy: 0.8199 - val_auc: 0.6986
Epoch 12/25
145/145 [==============================] - 1s 5ms/step - loss: 0.1064 - accuracy: 0.7881 - auc: 0.7214 - val_loss: 0.0920 - val_accuracy: 0.8114 - val_auc: 0.7100
Epoch 13/25
145/145 [==============================] - 1s 4ms/step - loss: 0.0955 - accuracy: 0.7979 - auc: 0.7580 - val_loss: 0.0931 - val_accuracy: 0.8161 - val_auc: 0.7110
Epoch 14/25
145/145 [==============================] - 1s 4ms/step - loss: 0.0984 - accuracy: 0.7936 - auc: 0.7536 - val_loss: 0.0935 - val_accuracy: 0.8218 - val_auc: 0.7072
Epoch 15/25
145/145 [==============================] - 1s 5ms/step - loss: 0.0954 - accuracy: 0.8029 - auc: 0.7598 - val_loss: 0.0934 - val_accuracy: 0.8152 - val_auc: 0.7024
Epoch 16/25
145/145 [==============================] - 1s 6ms/step - loss: 0.0882 - accuracy: 0.8070 - auc: 0.7837 - val_loss: 0.0948 - val_accuracy: 0.8180 - val_auc: 0.7079
56/56 [==============================] - 0s 2ms/step - loss: 0.0942 - accuracy: 0.8188 - auc: 0.7010
Loss: 0.0942, Accuracy: 0.8188, AUC: 0.7010 
56/56 [==============================] - 0s 1ms/step

              precision    recall  f1-score   support

         0.0       0.45      0.37      0.41       334
         1.0       0.86      0.90      0.88      1438

    accuracy                           0.80      1772
   macro avg       0.66      0.63      0.64      1772
weighted avg       0.78      0.80      0.79      1772
```

Si vede come il modello venga allenato, raggiungendo una loss di 0.0942 con un'accuracy del 81.88% e un auc di 0.70.  
Si nota tuttavia, che il modello ha precision e recall basse sulla classe 0:
- quando ha predetto 0, aveva ragione il 45% delle volte
- il modello ha trovato il 37% degli 0

---

### Prove del modello
Una volta salvato il modello, prendo un utente a caso e per ogni prodotto che ha recensito utilizzo il modello per vedere cosa avrebbe recensito.

Ad esempio, l'utente A ha recensito:
- prodotto P1 → 5 stelle
- prodotto P2 → 3 stelle

Con questi dati, mando in input gli embedding dell'utente A e dei prodotto al mio modello e vedo cosa predice.

Ecco i risultati testando su un utente random:

```bash
Confronto rating reali vs predetti per l'utente AFGCJIO7DNVCCZPG4KVMKKXVPJLQ:

product_id  true_rating  predicted_rating
B08KT7FCYY            2                 0
B08BLKFGND            2                 0
B08KGVBW41            3                 0
B086N2SY91            3                 1
B09WN3KBDF            3                 0
B09GVHT2D3            3                 1
B08QHP717Z            4                 1
B08PPB3W6H            4                 1
B09NFQ69KT            4                 0
B09NFKDCPG            4                 1
B08PTZL95G            5                 1
B00BRN5TQY            5                 1
B08DK74M1P            5                 0
B08G149DSD            5                 1
B089ZQ8Y95            5                 1
```

Si nota come su 6 recensioni negative, il modelo abbia sbagliato 2 volte, e su 9 recensioni positive abbia sbagliato 2 volte.
