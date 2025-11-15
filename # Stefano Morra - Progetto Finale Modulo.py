# Stefano Morra - Progetto Finale Modulo 1

# Parte 1: Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parte 2: Importazione con Pandas
data = pd.read_csv("PYTHON 1\Database.csv")
print(f"Prime 5 righe del database:\n {data.head()}")
print(f"Dimensioni del database: {data.shape}")
print(f"Informazioni sul database: {data.info()}")

# Parte 3: Elaborazione Dati con Pandas
data["Incasso"] = data["Quantita"] * data["Prezzo_Unitario"]

# Incasso totale catena
print(f"Incasso totale catena: {data['Incasso'].sum()} €")

# Incasso medio per negozio
incasso_medio = data.groupby("Negozio")["Incasso"].mean()
print(incasso_medio)

# Tre prodotti più venduti
prod_più_venduti = data.groupby("Prodotto")["Quantita"].sum()
prod_più_venduti = prod_più_venduti.sort_values(ascending = False)
print("Prodotti più venduti",prod_più_venduti.head(3))

# Raggrupa i dati per Negozio e Prodotto e mostra l'incasso medio:
# Incasso medio per negozio
incasso_medio = data.groupby("Negozio")["Incasso"].mean()
print("Incasso medio per negozio", incasso_medio)

#Incasso medio per prodotto
incasso_medio = data.groupby("Prodotto")["Incasso"].mean()
print("Incasso medio per prodotto: ", incasso_medio)


# Parte 4: Uso di Numpy

quantità = np.array(data["Quantita"])

print(f"Quantià media di prodotti comprati: {np.mean(quantità)}")
print(f"Quantià massima di prodotti comprati in un ordine: {np.max(quantità)}")
print(f"Quantià minima di prodotti comprati in un ordine: {np.min(quantità)}")
print(f"Deviazione standad di pordotti comprati in un ordine: {np.std(quantità)}")

media_quantità = np.mean(quantità)
sopra_media = quantità[quantità > media_quantità]
percent_sopra_media = (sopra_media.size / quantità.size)*100
print(f"Percentuale di vendite sopra la media: {percent_sopra_media} %")

# Crea un array 2D con Quantità e Prezzo_Unitario. Calcola per ogni riga l'incasso: 

arr = np.array([data["Quantita"],data["Prezzo_Unitario"]])
print(arr.shape)
prezzi_totali = arr[0,:] * arr[1,:]

# Confronta se la colonna prezzo totale corrisponde: 

if list(prezzi_totali) == list(data["Incasso"]): 
    print("I risultati coincidono :)")
else:
    print("I risultati non coincidono :(")


# Parte 5: Visualizzazione con Matplotlib

plt.figure(figsize=(8,7))
data_grpby_Negozio = data.groupby("Negozio")["Incasso"].sum()
plt.bar(data_grpby_Negozio.index, data_grpby_Negozio.values)
plt.xlabel("Negozio")
plt.ylabel("Incasso Totale")
#plt.show()

plt.figure(figsize=(8,7))
data_grpby_Prodotto = data.groupby("Prodotto")["Incasso"].sum()
plt.pie(data_grpby_Prodotto.values, labels =  data_grpby_Prodotto.index, autopct="%1.1f%%")
plt.title("Vendite dei vari prodotti percentuale")
#plt.show()

plt.figure(figsize=(8,7))
plt.plot(data["Data"], data["Incasso"])
plt.ylabel("Incasso giornaliero")
plt.title("Volume incasso giorneliero")
#plt.show()

# Parte 6 : Analisi Avanzata
data["Categoria"] = ["Cibo", "Accessori", "Cibo", "Cibo", "Pulizie", 
                     "Vestiti", "Accessori", "Cibo", "Cibo", "Pulizie", 
                     "Vestiti", "Accessori", "Cibo", "Cibo", "Pulizie", 
                     "Vestiti", "Accessori", "Cibo", "Cibo", "Casa", 
                     "Vestiti", "Accessori", "Cibo", "Cibo", "Pulizie", 
                     "Vestiti", "Accessori", "Cibo", "Cibo", "Casa"]

data_grpby_Categoria = data.groupby("Categoria")["Incasso"].sum()
print("Incasso totale per categoria: ", data_grpby_Categoria)

data_grpby_Categoria = data.groupby("Categoria")["Quantita"].mean()
print("Quantità media per categoria", data_grpby_Categoria)

data.to_csv("vendite_analizzate.csv")

# Parte 7: Estensioni

# Grafico combinato: 
fig, ax = plt.subplots(1,2)

data_grpby_Categoria = data.groupby("Categoria")["Incasso"].mean()
ax[0].bar(data_grpby_Categoria.index, data_grpby_Categoria.values)
ax[0].set_title("Incasso medio per categoria di prodotti")

data_grpby_Categoria = data.groupby("Categoria")["Quantita"].mean()
ax[1].plot(data_grpby_Categoria.index, data_grpby_Categoria.values)
ax[1].set_title("Quantità media per categoria di prodotti")

# Definizione funzione
def top_n_prodotti(data,n): 
    data_grpby_Prodotto = data.groupby("Prodotto")["Incasso"].sum()
    data_grpby_Prodotto.sort_values(ascending = False)
    print(data_grpby_Prodotto.head(3))
    
# Test funzione: 
top_n_prodotti(data,3)

