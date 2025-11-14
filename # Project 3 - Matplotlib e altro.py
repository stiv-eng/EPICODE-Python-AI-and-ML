# Project 3 - Matplotlib e altro

#---------------------------------
# Parte 1: Variabili e Tipi di Dati
nome = "Stefano"
età = 24
saldo_conto = 2500.78
VIP = True
destinazioni = ["Whistler", "Torquay", "Portacomaro", "Bruxelles", "Auburn"]

prezzi = {
          destinazioni[0] : 50,
          destinazioni[1] : 500,
          destinazioni[2] : 700,
          destinazioni[3] : 1000,
          destinazioni[4] : 200
          }


#-------------------------------
# Parte 2: OOP

class Cliente:
    def __init__(self, nome,eta, VIP):
        self.eta = eta
        self.nome = nome
        self.VIP = VIP

    def info_clienti(self):
        if self.VIP: 
            str_VIP = "VIP"

        else: 
            str_VIP = ""

        print(f"Cliente {self.nome}. Età {self.eta} anni. {str_VIP}")


class Viaggio: 
    def __init__(self, destinazione, prezzo, giorni): 
        self.destinazione = destinazione
        self.prezzo = prezzo
        self.giorni = giorni


class Prenotazione: 
    def __init__(self, cliente, viaggio): 
        self.cliente = cliente
        self.viaggio = viaggio

    def dettagli(self):

        if self.cliente.VIP: 
            sconto = 0.1
        else: 
            sconto = 0

        prezzo = self.viaggio.prezzo * sconto
        print(f"Il viaggio verso {self.viaggio.destinazione} per il cliente {self.cliente.nome} \
costa {prezzo}")
        
# Test delle funzionalità delle classi
Stefano = Cliente("Stefano", 24, True)
V1 = Viaggio("Whistler", 10, 1)
P1 = Prenotazione(Stefano, V1)
P1.dettagli()

#---------------------------------
# Parte 3: NumPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prezzi_prenotazioni = np.random.rand(200,2000,100)
print(f"Prezzo medio: {np.mean(prezzi_prenotazioni)}")
print(f"Prezzo massimo: {np.max(prezzi_prenotazioni)}")
print(f"Prezzo minimo: {np.min(prezzi_prenotazioni)}")
print(f"Deviazione Standard Prezzi: {np.std(prezzi_prenotazioni)}")

above_average = 100 * np.sum(prezzi_prenotazioni > np.mean(prezzi_prenotazioni)) / (prezzi_prenotazioni.size)
print(f"Percentuale di prezzi sopra la media: {above_average}%")

#-------------------------------
# Parte 4: Pandas

data_dict = {"Nome": ["Anna","Stefano", "Olivia", "Olivia", "Anna", "Olivia"],
        "Destinazione": ["Whistler", "Auburn", "Auburn", "Torquay", "Torquay","Torquay"],
        "Prezzo": [10,500,400,700,800,1000],
        "Giorno_Partenza": ["01-01-2023", "25-08-2023", "10-11-2023","10-12-2023","5-10-2023","7-07-2023"],
        "Durata" : [5,10,20,10,3,14], 
        "Incasso" : [5,200,150,400,430,520]
        }
data = pd.DataFrame(data_dict)

# Calcola l'incasso totale dell'azienda:
print(f"Incasso Totale: {data['Incasso'].sum()}")

# Calcola l'incasso medio per destinazione
for destinazione in data["Destinazione"].unique(): 
    print(f"Prezzo medio per {destinazione}: {(data["Prezzo"][data["Destinazione"] == destinazione]).mean()}")

# Cerca le top 3 destinazioni più visitate
top_3_dest = data["Destinazione"].value_counts().head(3)
print("Top 3 destinazioni per numero di viaggi:")
for destinazione, number in top_3_dest.items(): 
    print(f"{destinazione} : {number} viaggi.")

#----------------------------------------
# Part 5: Matplotlib

# Crea un grafico a barre con l'incasso medio per ogni destinazione
mean_income = data.groupby("Destinazione")["Incasso"].mean()

plt.figure(figsize=(8,5))
plt.bar(mean_income.index, mean_income.values)
plt.xlabel("Destinazione")
plt.ylabel("Incasso medio")
plt.title("Incasso medio per destinazione")
#plt.show()

# Crea un grafico a linee che mostri l'andamento giornaliero degli incassi
grouped_db = data.groupby("Destinazione")
plt.figure(figsize=(8,5))
plt.plot(data["Giorno_Partenza"], data["Incasso"])
plt.ylabel("Incasso")
plt.title("Andamento giornaliero degli incassi")
#plt.show()

# Crea un grafico a tortqa che mostri la percentuale di vendite per ciascuna destinazione
income_per_dest = data.groupby("Destinazione")["Incasso"].sum()
plt.figure(figsize=(8,5))
plt.pie(income_per_dest.values,labels = income_per_dest.index)
plt.title("Porzione di incassi per destinazione")
#plt.show()

#-------------------------------
# Parte 6: Analisi Avanzata
Continenti = {"Europa" : ["Torquay"], 
              "Asia" : [], 
              "America": ["Auburn", "Whistler"],
              "Africa" : []}

# Calcola l'incasso totale per categoria
print("Calcola incasso totale per continente:")
for continente, dest in Continenti.items(): 
    incasso = data["Incasso"][data["Destinazione"].isin(dest)].sum()
    print(f"Incasso per continente {continente}: {incasso}")

# Another way of doing it using loc(): 
print("Calcola incasso totale per categoria:")
for continente,dest in Continenti.items():
    incasso = data.loc[data["Destinazione"].isin(dest), "Incasso"].sum()
    print(f"Incasso per continente {continente}: {incasso}")

# Calcola la durata media dei viaggi per continente: 
print("Calcola la durata media per continente:")
for continente, dest in Continenti.items():
    durata = data.loc[data["Destinazione"].isin(dest), "Durata"].mean()
    if np.isnan(durata):
        durata  = 0

    print(f"Durata media in giorno per il continente {continente}: {durata} giorni.")

# Salva il DF in un CSV
data.to_csv("prenotazioni_analizzate.csv")

#-----------------------------------
# Parte 7: Estensioni
def clienti_con_più_prenotazioni(data,n): 

    top_n_clienti = data["Nome"].value_counts().head(n)
    print(f"Top {n} clienti: {top_n_clienti}")

clienti_con_più_prenotazioni(data,3)

# Realizza un grafico che mostri: incasso medio per categoria e durata media per categoria

fig,ax = plt.subplots(1,2)
inc = dict([])
dur = dict([])
for continente, dest in Continenti.items():
    inc[continente] = data.loc[data["Destinazione"].isin(dest), "Incasso"].mean()
    dur[continente] = data.loc[data["Destinazione"].isin(dest), "Durata"].mean()

    if np.isnan(inc[continente]):
        inc[continente] = 0

    if np.isnan(dur[continente]):
        dur[continente] = 0

    

ax[0].bar(inc.keys(), inc.values())
ax[0].set_title("Incasso medio per continente")
ax[1].plot(dur.keys(), dur.values())
ax[1].set_title("Durata media per continente")
plt.show()