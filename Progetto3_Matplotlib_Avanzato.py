import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.widgets import CheckButtons

df = pd.read_csv("PYTHON 2/Database.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"] = pd.to_datetime(df["Ship Date"])

df["Category"] = df["Category"].astype("category")
df["Sub-Category"] = df["Sub-Category"].astype("category")
df["Region"] = df["Region"].astype("category")

df["Profit"] = df["Profit"].fillna(df["Profit"].mean())
df["Sales"] = df["Sales"].fillna(df["Profit"].mean())
df["Quantity"] = df["Quantity"].fillna(df["Profit"].mean())

df["Year"] = df["Order Date"].dt.year


# Analisi Esplorativa

# Calcola totale Vendite e profitti per anno: 

print(f"Totale vendite per anno: {df.groupby("Year")["Sales"].sum()}")

print(f"\nTotale profitto per anno: {df.groupby("Year")["Profit"].sum()}")

# Trova le top 5 sottocategorie più vendute

top_5 = df.groupby("Sub-Category")["Quantity"].sum().sort_values(ascending=False)
print(top_5.head(5))

# Mappa interrativa delle vendite: 
# Creiamo un plot con le date e le sales e profit uno sopra l'altro, in cui si può scegliere con un button quale delle due fare: 

fig, ax = plt.subplots(figsize=(10, 6))

# Plot delle due serie (inizialmente visibili)
line_sales, = ax.plot(df["Order Date"], df["Sales"], label="Sales")
line_profit, = ax.plot(df["Order Date"], df["Profit"], label="Profit")

ax.set_xlabel("Date")
ax.set_ylabel("Value (€)")
ax.set_title("Sales & Profit over Time")
ax.legend()

# Creiamo lo spazio per il checkbox, le labels da aggiungere e la visibilità iniziale
checkbox = plt.axes([0.02, 0.4, 0.12, 0.15])
labels = ["Sales", "Profit"]
visibility = [True, True]

# Creiamo l'oggetto checkbutton
check = CheckButtons(checkbox, labels, visibility)

# Creiamo la funzione update per il check-button
def update(label):
    if label == "Sales":
        line_sales.set_visible(not line_sales.get_visible())
    if label == "Profit":
        line_profit.set_visible(not line_profit.get_visible())
    plt.draw()

# Colleghiamo la checkbox alla funzione
check.on_clicked(update)

plt.show()



