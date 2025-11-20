# Project 1: Manipolazione dati con Pandas
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Parte 1: 
df = pd.read_csv("Python 2\database_vendite_Project1.csv")
print(df.head())
print(df.info())
print(df.describe())

# Parte 2: Pulizia
df["Vendite"] = df["Vendite"].fillna(0)
df["Vendite"] = df["Vendite"].astype(int)
df["Prezzo"] = df["Prezzo"].fillna(df["Prezzo"].mean())
df["Data"] = pd.to_datetime(df["Data"])
df["Prodotto"] = df["Prodotto"].astype("string")

# Rimuovi righe duplicate con df.drop_duplicates(). 
# La funziona cancella le righe duplicate, cioè righe in cui tutti i valori sono uguauli per tutte le feature,
# mantenendone solo una. 
df = df.drop_duplicates()

# Crea funzione per controllare i tipi di dati nel DF
tipi_attesi = {
    "Data" : "datetime64[ns]",
    "Prodotto" : "string",
    "Vendite" : "int64",
    "Prezzo" : "float64",
}

def controlla_tipi(df, tipi_attesi):
    for colonna, tipo_atteso in tipi_attesi.items():
        tipo_reale = str(df[colonna].dtype)

        if tipo_reale != tipo_atteso:
            print(f"Tipo ERRATO per '{colonna}': trovato {tipo_reale}, atteso {tipo_atteso}")
        else:
            print(f"Tipo OK per '{colonna}' ({tipo_reale})")

controlla_tipi(df, tipi_attesi)

# Parte 3: Analisi Esplorativa
# Calcola Vendite Totali per Prodotto: 
for prodotto, valore in df.groupby("Prodotto")["Vendite"].mean().items():
    print(f"Vendite medie per {prodotto}: {valore}")

# Individuare Prodotto più venduto e meno venduto
for prodotto,valore in df.groupby("Prodotto")["Vendite"].sum().items():
    print(f"Vendite totali per {prodotto}: {valore}")

vend_tot_per_prod = df.groupby("Prodotto")["Vendite"].sum()
print(f"Prodotto più venduto: {vend_tot_per_prod.idxmax()}, {vend_tot_per_prod.max()}")
print(f"Prodotto MENO venduto: {vend_tot_per_prod.idxmin()}, {vend_tot_per_prod.min()}")

# Calcolare vendite medie giornaliere
print(f"Vendite medie giornaliere: {df.groupby("Data")["Vendite"].sum().mean()} al giorno.")

