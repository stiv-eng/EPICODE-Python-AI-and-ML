# Rete GRU per imparare testo Dante
# Versione Keras + PyTorch backend + GPU + dataset più leggero

import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np
import requests
import urllib3

keras.mixed_precision.set_global_policy("mixed_float16")

# --------------------------------------------------
# CHECK GPU
# --------------------------------------------------

print("Keras backend:", keras.backend.backend())
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected. Training will use CPU.")


# --------------------------------------------------
# 1) DOWNLOAD TESTO
# --------------------------------------------------

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scarica_testo_dante():
    url = "https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt"
    print(f"Scaricamento testo da: {url}...")

    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        response.encoding = "utf-8"
        text = response.text
        print(f"Download completato. Lunghezza testo: {len(text)} caratteri")
        return text

    except Exception as e:
        print(f"Errore: {e}")
        print("Tentativo con link di riserva...")

        url_backup = "https://raw.githubusercontent.com/wpm/t-snetext-vis/master/data/divina_commedia.txt"
        r2 = requests.get(url_backup, verify=False)
        r2.encoding = "utf-8"
        return r2.text


text = scarica_testo_dante()


# --------------------------------------------------
# 2) PRE-PROCESSING
# --------------------------------------------------

vocab = sorted(set(text))
char2idx = {char: i for i, char in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text], dtype=np.int64)

vocab_size = len(vocab)
print("Vocabolario:", vocab_size, "caratteri")


# --------------------------------------------------
# 3) CREAZIONE DATASET LEGGERO
# --------------------------------------------------

seq_length = 60
step = seq_length + 1

inputs = []
targets = []

for i in range(0, len(text_as_int) - seq_length, step):
    chunk = text_as_int[i:i + seq_length + 1]

    inputs.append(chunk[:-1])
    targets.append(chunk[1:])

X = np.array(inputs, dtype=np.int64)
y = np.array(targets, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# 4) MODELLO GRU
# --------------------------------------------------

embedding_dim = 128
rnn_units = 256

model = keras.Sequential([
    keras.layers.Input(shape=(seq_length,)),

    keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    ),

    keras.layers.GRU(
        rnn_units,
        return_sequences=True,
        recurrent_initializer="glorot_uniform"
    ),

    keras.layers.GRU(
        rnn_units,
        return_sequences=True
    ),

    keras.layers.Dense(vocab_size)
])

model.summary()


# --------------------------------------------------
# 5) TRAINING
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

history = model.fit(
    X,
    y,
    batch_size=32,
    epochs=10,
    validation_split=0.1,
    shuffle=True
)

if torch.cuda.is_available():
    print("GPU memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("GPU memory reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")


# --------------------------------------------------
# 6) GENERAZIONE TESTO
# --------------------------------------------------

def generate_text(model, start_string, temperature=0.7, num_generate=500):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = np.array([input_eval], dtype=np.int64)

    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)

        predictions = predictions[:, -1, :] / temperature

        predicted_id_tensor = keras.random.categorical(
            predictions,
            num_samples=1
        )

        predicted_id = int(
            keras.ops.convert_to_numpy(predicted_id_tensor)[0, 0]
        )

        new_char_tensor = np.array([[predicted_id]], dtype=np.int64)

        input_eval = np.concatenate(
            [input_eval, new_char_tensor],
            axis=1
        )

        if input_eval.shape[1] > seq_length:
            input_eval = input_eval[:, 1:]

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)


print(generate_text(
    model,
    start_string="Nel mezzo del",
    temperature=0.7,
    num_generate=500
))