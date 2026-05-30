# Progetto Finale Modulo 4 EPICODE

# Obiettivo: Sviluppare un'IA basata su una CNN capace di 
# classificare le immagini in tre categorie: Sasso, Carta o Forbice. 

# ROADMAP DI SVILUPPO: 
# 1) Utilizzare la classe ImageDataGeneratore di Keras per gestire il flusso 
# delle immaginir ed evitare sovraccarichi della GPU

# 2) DataAugmentatoin (rotation, width_shift, shear, horizontal_flip, fill_mode = "nearest")
# 3) Normalizzazione (rescale con /255.0)

# ARCHITETTURA RETE CNN
# 1) Input: 150x150, RGB
# 2) Conv2D e MaxPooling2D alternati ( almeno 3 blocchi) con numero di filitri crescente.
# 3) Flatten + Dense (512 neuroni e activation "relu")
# 4) Output: Dense con 3 neuroni e activation = "softmax"

# COMPILAZIONE E TRAINING
# Optimizer = Adam
# Loss = "categorical_crossentropy"
# Metriche = "accuracy"
# EPOCHE = 20
# callbacks: Earlystopping se accuracy suera 98%

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import pandas as pd

# 1. CARICAMENTO DATI (MANCANTE CAUSA: DATI NON FORNITI)

# 2. Model Architecture CNN

def build_model():

    inputs = keras.Input(shape = (128,128,3))

    x = keras.layers.Conv2D(32, (3,3), padding = "same", activation = "relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Dropout(0.2)(x)
    
    x = keras.layers.Conv2D(64, (3,3), padding = "same", activation = "relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Conv2D(128, (3,3), padding = "same", activation = "relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(512, activation = "relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(3, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_model()

# 3. COMPILAZIONE

model.compile(
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3),
    loss = "sparse_categorical_crossentropy", # Usiamo SCCE per non dover fare one-hot encoding sulle labels. 
    metrics = ["accuracy"]
)

# 4. CALLBACKS

class StopAt98Accuracy(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        accuracy = logs.get("val_accuracy")

        if accuracy is not None and accuracy >= 0.98:
            print(f"\nAccuracy raggiunta: {accuracy:.4f}")
            print("Interruzione dell'addestramento.")
            self.model.stop_training = True

callbacks = [
    StopAt98Accuracy()
]

# 5. TRAINING

history = model.fit(
    x_train, y_train, 
    epochs = 20, 
    batch_size = 128, 
    validation_data=(x_test, y_test), 
    callbacks = callbacks, 
    verbose = True
)

loss, acc = model.evaluate(x_test, y_test, verbose = False)
print(f"ACCURACY on TEST DS: {(acc*100):.2f}%")
print(f"LOSS on TEST DS: {loss:.2f}")