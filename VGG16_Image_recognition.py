# VGG!& Image Recognition - EPICODE Project 
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Import VGG16 Model
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image

# Image to process: 
IMMAGINE_URL = (
    "https://images.pexels.com/photos/36853192/pexels-photo-36853192.jpeg?cs=srgb&dl=pexels-albima-2149657628-36853192.jpg&fm=jpg"
)

# Define function to build the model VGG16. 
def carica_modello_vgg():

    print("Caricamento del modello VGG16 con pesi ImageNet...")

    model = VGG16(weights="imagenet", include_top=True)

    print("Caricamento VGG16 completato!")
    return model


def ottieni_e_processa_immagine(url):
   
    print(f"Download e processing da: {url}")

    response = requests.get(url)
    response.raise_for_status()

    # Extract the image
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # VGG16 standard input size
    img_resized = img.resize((224, 224))

    x = keras_image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)

    # VGG16-specific preprocessing
    x = preprocess_input(x)

    return x, img_resized


def classifica_immagine(model, processed_image):

    print("Classificazione in corso...")

    predictions = model.predict(processed_image)

    results = decode_predictions(predictions, top=5)[0]

    return results


def stampa_risultati(results):

    print("Risultati:\n")
 

    for rank, (_, label, probability) in enumerate(results, start=1):
        print(f"{rank}. {label}: {probability * 100:.2f}%")


# MAIN FUCNTION

def main():
    print("Librerie importate. Pronto per VGG16.\n")
    print(f"URL impostato: {IMMAGINE_URL}\n")

    model = carica_modello_vgg()
    processed_image, original_image = ottieni_e_processa_immagine(IMMAGINE_URL)
    results = classifica_immagine(model, processed_image)
    stampa_risultati(results)


if __name__ == "__main__":
    main()
