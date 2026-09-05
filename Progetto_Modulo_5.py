"""
====================================================================================================
PROGETTO FINALE: CHEFBOT AI - DALLO SCATTO ALLA RICETTA
====================================================================================================

OBIETTIVO:
Creare un assistente intelligente capace di collegare tre mondi differenti:

1. COMPUTER VISION
   Una foto del piatto viene analizzata da MobileNetV1.
   Il modello identifica quale piatto è presente nell'immagine.

2. KNOWLEDGE RETRIEVAL
   Una volta ottenuto il nome del piatto, utilizziamo questa etichetta
   come "chiave" per interrogare un database JSON contenente:
   - ingredienti principali
   - calorie stimate
   - descrizione
   - parole chiave semantiche

3. NLP / SEMANTIC SEARCH
   L'utente può scrivere frasi naturali come:

       "voglio qualcosa di fresco e leggero"

   La frase viene trasformata in un embedding.
   Anche le descrizioni dei piatti vengono trasformate in embeddings.

   Attraverso la similarità coseno troviamo il piatto semanticamente
   più vicino alla richiesta.

"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ================================================================================================
# 1. CONFIGURAZIONE GENERALE
# ================================================================================================

# Dimensione richiesta normalmente da MobileNet.
IMG_SIZE = (224, 224)

# Numero di immagini elaborate contemporaneamente.
BATCH_SIZE = 32

# Numero massimo di epoche.
EPOCHS = 10

# Percorsi dei file che creeremo.
MODEL_PATH = "chefbot_mobilenet.keras"
KNOWLEDGE_PATH = "chefbot_knowledge.json"

# ================================================================================================
# 2. CLASSI FOOD101 SELEZIONATE
# ================================================================================================

"""
Food101 contiene 101 categorie.
Per il nostro progetto utilizziamo un subset di 10 classi.
La Dense finale del modello avrà quindi 10 neuroni.
"""

SELECTED_CLASSES = [
    "pizza",
    "hamburger",
    "sushi",
    "ramen",
    "caesar_salad",
    "cheesecake",
    "tiramisu",
    "waffles",
    "baklava",
    "churros"
]

NUM_CLASSES = len(SELECTED_CLASSES)


# ================================================================================================
# 3. CREAZIONE DEL KNOWLEDGE DATABASE in JSON
# ================================================================================================

def create_knowledge_database(path=KNOWLEDGE_PATH):

    """
    Crea il database testuale di ChefBot.
    La chiave principale del dizionario deve corrispondere ESATTAMENTE
    all'etichetta utilizzata dal classificatore.
    Può quindi interrogare:
        knowledge["pizza"]
    """

    knowledge = {

        "pizza": {
            "nome": "Pizza",
            "ingredienti_principali": [
                "farina",
                "pomodoro",
                "mozzarella",
                "olio extravergine di oliva"
            ],
            "calorie_stimate": 270,
            "descrizione":
                "Piatto italiano a base di impasto cotto al forno, "
                "generalmente condito con pomodoro e mozzarella.",
            "parole_chiave": [
                "italiano",
                "salato",
                "caldo",
                "formaggio",
                "pomodoro",
                "comfort food"
            ]
        },

        "hamburger": {
            "nome": "Hamburger",
            "ingredienti_principali": [
                "pane",
                "carne di manzo",
                "insalata",
                "pomodoro",
                "formaggio"
            ],
            "calorie_stimate": 550,
            "descrizione":
                "Panino sostanzioso con carne, verdure e spesso formaggio "
                "o salse. Ideale come pasto ricco e saporito.",
            "parole_chiave": [
                "carne",
                "panino",
                "sostanzioso",
                "ricco",
                "salato",
                "comfort food"
            ]
        },

        "sushi": {
            "nome": "Sushi",
            "ingredienti_principali": [
                "riso",
                "pesce",
                "alghe nori",
                "verdure"
            ],
            "calorie_stimate": 200,
            "descrizione":
                "Piatto giapponese fresco a base di riso, pesce e verdure. "
                "Può essere una scelta relativamente leggera.",
            "parole_chiave": [
                "fresco",
                "leggero",
                "pesce",
                "giapponese",
                "riso",
                "freddo"
            ]
        },

        "ramen": {
            "nome": "Ramen",
            "ingredienti_principali": [
                "noodles",
                "brodo",
                "carne",
                "uovo",
                "verdure"
            ],
            "calorie_stimate": 500,
            "descrizione":
                "Zuppa giapponese calda e saporita composta da noodles "
                "serviti in un brodo ricco con diversi condimenti.",
            "parole_chiave": [
                "caldo",
                "zuppa",
                "brodo",
                "giapponese",
                "comfort food",
                "inverno"
            ]
        },

        "caesar_salad": {
            "nome": "Caesar Salad",
            "ingredienti_principali": [
                "lattuga",
                "parmigiano",
                "crostini",
                "salsa Caesar"
            ],
            "calorie_stimate": 300,
            "descrizione":
                "Insalata fresca e croccante a base di lattuga, parmigiano "
                "e crostini. Adatta a chi cerca qualcosa di fresco e leggero.",
            "parole_chiave": [
                "fresco",
                "leggero",
                "insalata",
                "verdure",
                "croccante",
                "estate"
            ]
        },

        "cheesecake": {
            "nome": "Cheesecake",
            "ingredienti_principali": [
                "formaggio cremoso",
                "biscotti",
                "burro",
                "zucchero"
            ],
            "calorie_stimate": 380,
            "descrizione":
                "Dessert cremoso e dolce composto da una base di biscotti "
                "e uno strato di formaggio zuccherato.",
            "parole_chiave": [
                "dolce",
                "dessert",
                "cremoso",
                "zucchero",
                "formaggio"
            ]
        },

        "tiramisu": {
            "nome": "Tiramisù",
            "ingredienti_principali": [
                "mascarpone",
                "savoiardi",
                "caffè",
                "cacao",
                "zucchero"
            ],
            "calorie_stimate": 420,
            "descrizione":
                "Dessert italiano morbido e cremoso preparato con mascarpone, "
                "caffè, cacao e savoiardi.",
            "parole_chiave": [
                "dolce",
                "dessert",
                "caffè",
                "cremoso",
                "italiano",
                "cacao"
            ]
        },

        "waffles": {
            "nome": "Waffles",
            "ingredienti_principali": [
                "farina",
                "uova",
                "latte",
                "burro",
                "zucchero"
            ],
            "calorie_stimate": 310,
            "descrizione":
                "Dolce morbido all'interno e leggermente croccante all'esterno, "
                "spesso accompagnato da frutta, miele o sciroppo.",
            "parole_chiave": [
                "dolce",
                "colazione",
                "miele",
                "sciroppo",
                "frutta"
            ]
        },

        "baklava": {
            "nome": "Baklava",
            "ingredienti_principali": [
                "pasta fillo",
                "pistacchi",
                "noci",
                "miele",
                "zucchero"
            ],
            "calorie_stimate": 430,
            "descrizione":
                "Dessert molto dolce e croccante composto da sottili strati "
                "di pasta fillo, frutta secca e miele.",
            "parole_chiave": [
                "dolce",
                "miele",
                "dessert",
                "pistacchi",
                "frutta secca",
                "croccante"
            ]
        },

        "churros": {
            "nome": "Churros",
            "ingredienti_principali": [
                "farina",
                "acqua",
                "zucchero",
                "olio",
                "cannella"
            ],
            "calorie_stimate": 360,
            "descrizione":
                "Dolce fritto croccante, spesso ricoperto di zucchero "
                "e cannella e accompagnato da cioccolato.",
            "parole_chiave": [
                "dolce",
                "dessert",
                "fritto",
                "zucchero",
                "cioccolato",
                "croccante"
            ]
        }
    }

    # Scriviamo fisicamente il dizionario in un file JSON.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            knowledge,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"[KNOWLEDGE] Database creato: {path}")

    return knowledge


# ================================================================================================
# 4. CARICAMENTO DEL KNOWLEDGE DATABASE
# ================================================================================================

def load_knowledge_database(path=KNOWLEDGE_PATH):

    """
    Legge il JSON dal disco e lo trasforma nuovamente
    in un dizionario Python.
    """

    with open(path, "r", encoding="utf-8") as file:
        knowledge = json.load(file)

    return knowledge


# ================================================================================================
# 5. CARICAMENTO FOOD101
# ================================================================================================

def load_food101():

    """
    Scarichiamo Food101 attraverso TensorFlow Datasets.

    """     

    print("\n[DATASET] Caricamento Food101...")

    (ds_train, ds_validation), ds_info = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    return ds_train, ds_validation, ds_info


# ================================================================================================
# 6. CREAZIONE DEL SUBSET DI 10 CLASSI
# ================================================================================================

def prepare_subset(ds_train, ds_validation, ds_info):

    food101_names = ds_info.features["label"].names

    # Trova gli ID delle classi a cui siamo interessati, in SELECTED CLASSES
    selected_ids = [
        food101_names.index(class_name)
        for class_name in SELECTED_CLASSES
    ]

    selected_ids_tensor = tf.constant(
        selected_ids,
        dtype=tf.int64
    )

    print("\nClassi selezionate:")

    for new_id, (name, original_id) in enumerate(
        zip(SELECTED_CLASSES, selected_ids)
    ):
        print(
            f"Nuova classe {new_id}: "
            f"{name:15s} <- Food101 ID {original_id}"
        )

    # --------------------------------------------------------------------------------------------
    # FILTRO
    # --------------------------------------------------------------------------------------------

    def keep_selected(image, label):

        """
        Confrontiamo la label corrente con tutte le label
        che abbiamo deciso di utilizzare.
        """

        matches = tf.equal(
            label,
            selected_ids_tensor
        )

        # True se almeno un ID coincide.
        return tf.reduce_any(matches)

    # --------------------------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------------------------

    def preprocess(image, label):

        """
        1. Troviamo la nuova label 0-9.
        2. Ridimensioniamo l'immagine a 224x224.
        3. Convertiamo i pixel in float32.
        """

        matches = tf.equal(
            label,
            selected_ids_tensor
        )

        new_label = tf.argmax(
            tf.cast(matches, tf.int32),
            output_type=tf.int32
        )

        image = tf.image.resize(
            image,
            IMG_SIZE
        )

        image = tf.cast(
            image,
            tf.float32
        )

        return image, new_label

    # --------------------------------------------------------------------------------------------
    # PIPELINE TRAINING
    # --------------------------------------------------------------------------------------------

    train_subset = (
        ds_train
        .filter(keep_selected)
        .map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .shuffle(2000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # --------------------------------------------------------------------------------------------
    # PIPELINE VALIDATION
    # --------------------------------------------------------------------------------------------

    validation_subset = (
        ds_validation
        .filter(keep_selected)
        .map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_subset, validation_subset


# ================================================================================================
# 8. COSTRUZIONE DEL CLASSIFICATORE MOBILENET
# ================================================================================================

def build_vision_model():

    # --------------------------------------------------------------------------------------------
    # BASE PRE-ADDESTRATA
    # --------------------------------------------------------------------------------------------

    base_model = tf.keras.applications.MobileNet(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),

        # Rimuove il classificatore ImageNet originale.
        include_top=False,

        # Utilizza pesi già addestrati.
        weights="imagenet"
    )

    # Congeliamo MobileNet.
    base_model.trainable = False

    # --------------------------------------------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------------------------------------------

    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10)
        ],
        name="data_augmentation"
    )

    # --------------------------------------------------------------------------------------------
    # NUOVO MODELLO
    # --------------------------------------------------------------------------------------------

    inputs = layers.Input(
        shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Durante il training alteriamo leggermente le immagini.
    x = data_augmentation(inputs)

    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1
    )(x)

    # MobileNet estrae le caratteristiche visive.
    x = base_model(
        x,
        training=False
    )

    """
    MobileNet produce diverse feature map.

    GlobalAveragePooling2D le comprime in un unico vettore
    di caratteristiche.
    """

    x = layers.GlobalAveragePooling2D()(x)

    # Riduce overfitting.
    x = layers.Dropout(0.30)(x)

    # Piccolo classificatore personalizzato.
    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.20)(x)

    # 10 neuroni = 10 piatti.
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="ChefBot_MobileNet"
    )

    # --------------------------------------------------------------------------------------------
    # COMPILAZIONE
    # --------------------------------------------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model


# ================================================================================================
# 8. TRAINING
# ================================================================================================

def train_model(model, train_ds, validation_ds):

    # Definizione CALLBACKS: Early Stopping e Reduce LR on Plateau
    callbacks = [

        EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Salviamo il modello addestrato.
    model.save(MODEL_PATH)

    print(
        f"\n[MODEL] Modello salvato in: {MODEL_PATH}"
    )

    return history


# ================================================================================================
# 9. VISUALIZZAZIONE TRAINING
# ================================================================================================

def plot_training(history):

    # --------------------------------------------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("ChefBot - Accuracy")
    plt.legend()

    plt.show()

    # --------------------------------------------------------------------------------------------
    # LOSS
    # --------------------------------------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("ChefBot - Loss")
    plt.legend()

    plt.show()


# ================================================================================================
# 10. VALUTAZIONE DEL CLASSIFICATORE
# ================================================================================================

def evaluate_model(model, validation_ds):

    print("\n" + "=" * 80)
    print("VALUTAZIONE COMPUTER VISION")
    print("=" * 80)

    loss, accuracy = model.evaluate(
        validation_ds,
        verbose=1
    )

    print(f"\nValidation Accuracy: {accuracy:.2%}")
    print(f"Validation Loss: {loss:.4f}")

    # Recuperiamo tutte le etichette vere.
    y_true = np.concatenate([
        labels.numpy()
        for images, labels in validation_ds
    ])

    # Predizioni probabilistiche.
    probabilities = model.predict(
        validation_ds,
        verbose=1
    )

    # Prendiamo la classe con probabilità maggiore.
    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    print("\nCLASSIFICATION REPORT:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=SELECTED_CLASSES,
            digits=3
        )
    )


# ================================================================================================
# 12. PREDIZIONE DI UNA NUOVA FOTO
# ================================================================================================

def predict_dish(model, image_path):

    """
    Riceve il percorso di un'immagine caricata dall'utente.
    Esempio:
        image_path = "foto_pizza.jpg"
    Restituisce:
        label
        confidence
        immagine
    """

    # Apriamo l'immagine con PIL.
    image = Image.open(image_path).convert("RGB")

    # Ridimensionamento.
    resized_image = image.resize(IMG_SIZE)

    # PIL -> NumPy.
    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_batch,
        verbose=0
    )[0]

    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]

    predicted_label = SELECTED_CLASSES[
        predicted_index
    ]

    return (predicted_label, float(confidence), image)


# ================================================================================================
# 12. KNOWLEDGE LOOKUP
# ================================================================================================

def lookup_dish(label, knowledge):

    """
    Questa funzione rappresenta il ponte:

             COMPUTER VISION
                    │
                    ▼
                "pizza"
                    │
                    ▼
                JSON LOOKUP
                    │
                    ▼
        ingredienti / calorie / descrizione
    """

    if label not in knowledge:
        return None

    return knowledge[label]


# ================================================================================================
# 14. ANALISI COMPLETA DELLA FOTO
# ================================================================================================

def analyze_food_image(model,image_path,knowledge):

    predicted_label, confidence, image = predict_dish(model, image_path)
    dish_info = lookup_dish(predicted_label,knowledge)

    print("\n" + "=" * 80)
    print("CHEFBOT - ANALISI IMMAGINE")
    print("=" * 80)

    print(f"\nPiatto riconosciuto: "f"{dish_info['nome']}")

    print(f"Confidenza modello: "f"{confidence:.2%}")

    print("\nIngredienti principali:")

    for ingredient in dish_info["ingredienti_principali"]: print(f"  - {ingredient}")

    print("\nCalorie stimate:",dish_info["calorie_stimate"],"kcal")
    print("\nDescrizione:",dish_info["descrizione"])

    # Visualizzazione.
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.title(
        f"{dish_info['nome']} "
        f"({confidence:.1%})"
    )
    plt.axis("off")
    plt.show()

    return predicted_label, dish_info


# ================================================================================================
# 15. CREAZIONE DEL MOTORE SEMANTICO
# ================================================================================================

def create_semantic_engine(knowledge):

    print("\n[NLP] Caricamento modello embeddings...")

    semantic_model = SentenceTransformer("sentence-transformers/""paraphrase-multilingual-MiniLM-L12-v2")
    dish_labels = list(knowledge.keys())
    dish_documents = []

    # --------------------------------------------------------------------------------------------
    # CREAZIONE DEL TESTO SEMANTICO PER OGNI PIATTO
    # --------------------------------------------------------------------------------------------

    for label in dish_labels:

        dish = knowledge[label]

        ingredients = ", ".join(dish["ingredienti_principali"])

        keywords = ", ".join(dish["parole_chiave"])

        document = (
            f"{dish['nome']}. "
            f"{dish['descrizione']} "
            f"Ingredienti principali: {ingredients}. "
            f"Caratteristiche: {keywords}."
        )

        dish_documents.append(
            document
        )

    # --------------------------------------------------------------------------------------------
    # EMBEDDINGS DEL DATABASE
    # --------------------------------------------------------------------------------------------

    dish_embeddings = semantic_model.encode(
        dish_documents,

        # Restituisce array NumPy.
        convert_to_numpy=True,

        # Porta i vettori a lunghezza 1.
        normalize_embeddings=True
    )

    print(f"[NLP] Creati embeddings per "f"{len(dish_labels)} piatti.")

    return (semantic_model,dish_labels,dish_embeddings)


# ================================================================================================
# 15. RICERCA SEMANTICA PER UMORE
# ================================================================================================

def search_by_mood(query,semantic_model,dish_labels,dish_embeddings,knowledge,top_k=3):

    # --------------------------------------------------------------------------------------------
    # QUERY -> EMBEDDING
    # --------------------------------------------------------------------------------------------

    query_embedding = semantic_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        dish_embeddings
    )[0]

    # Ordiniamo gli indici dal punteggio più alto.
    best_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    print("\n" + "=" * 80)
    print("CHEFBOT - RICERCA PER UMORE")
    print("=" * 80)

    print(f'\nRichiesta: "{query}"\n')

    for position, index in enumerate(best_indices,start=1):

        label = dish_labels[index]
        score = float(similarities[index])
        dish = knowledge[label]

        results.append(
            {
                "label": label,
                "score": score,
                "dish": dish
            }
        )

        print(f"{position}. {dish['nome']}")
        print(f"   Similarità: {score:.3f}")
        print(f"   {dish['descrizione']}")

    return results


# ================================================================================================
# 16. INTERFACCIA CHEFBOT
# ================================================================================================

def run_chefbot(
    model,
    knowledge,
    semantic_model,
    dish_labels,
    dish_embeddings
):

    """
    Piccola interfaccia testuale.

    Permette di scegliere:

        1 -> riconoscimento foto
        2 -> ricerca semantica
        0 -> uscita
    """

    while True:

        print("\n" + "=" * 80)
        print("                         CHEFBOT AI")
        print("=" * 80)

        print("\n1 - Analizza una foto")
        print("2 - Cerca un piatto per umore")
        print("0 - Esci")

        choice = input(
            "\nScelta: "
        ).strip()

        # ----------------------------------------------------------------------------------------
        # COMPUTER VISION
        # ----------------------------------------------------------------------------------------

        if choice == "1":

            image_path = input("\nPercorso immagine: ").strip()

            if not os.path.exists(image_path):
                print("\n[ERRORE] ""Immagine non trovata.")

                continue
            analyze_food_image(model,image_path,knowledge)

        # ----------------------------------------------------------------------------------------
        # NLP
        # ----------------------------------------------------------------------------------------

        elif choice == "2":

            query = input("\nDi cosa hai voglia? ")
            search_by_mood(
                query,
                semantic_model,
                dish_labels,
                dish_embeddings,
                knowledge,
                top_k=3
            )

        # ----------------------------------------------------------------------------------------
        # EXIT
        # ----------------------------------------------------------------------------------------

        elif choice == "0":

            print("\nChefBot terminato.")
            break

        else:

            print("\nScelta non valida.")


# ================================================================================================
# 17. MAIN
# ================================================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("CHEFBOT AI - AVVIO SISTEMA")
    print("=" * 80)

    # --------------------------------------------------------------------------------------------
    # A. KNOWLEDGE BASE
    # --------------------------------------------------------------------------------------------

    if not os.path.exists(KNOWLEDGE_PATH):
        create_knowledge_database()

    knowledge = load_knowledge_database()

    # --------------------------------------------------------------------------------------------
    # B. COMPUTER VISION MODEL
    # --------------------------------------------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        """
        Se abbiamo già addestrato ChefBot,
        NON ripetiamo il training.
        Carichiamo direttamente il modello.
        """

        print("\n[MODEL] Caricamento modello già addestrato...")

        vision_model = tf.keras.models.load_model( MODEL_PATH)

    else:

        # Caricamento dataset.
        ds_train, ds_validation, ds_info = load_food101()

        # Selezione delle 10 classi.
        train_subset, validation_subset = prepare_subset(
            ds_train,
            ds_validation,
            ds_info
        )

        # Costruzione MobileNet.
        vision_model = build_vision_model()

        print("\nARCHITETTURA MODELLO:\n")

        vision_model.summary()

        # Training.
        history = train_model(
            vision_model,
            train_subset,
            validation_subset
        )

        # Grafici.
        plot_training(history)

        # Valutazione.
        evaluate_model(vision_model,validation_subset)

    # --------------------------------------------------------------------------------------------
    # C. MOTORE NLP
    # --------------------------------------------------------------------------------------------

    (semantic_model,dish_labels,dish_embeddings) = create_semantic_engine(knowledge)

    # --------------------------------------------------------------------------------------------
    # D. AVVIO CHEFBOT
    # --------------------------------------------------------------------------------------------

    run_chefbot(
        vision_model,
        knowledge,
        semantic_model,
        dish_labels,
        dish_embeddings
    )