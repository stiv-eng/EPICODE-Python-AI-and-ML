# PROJECT 2: NumPy 

import numpy as np

nome = "Stefano"
cognome = "Morra"
eta = 24
peso = 85
analisi = ["emocromo", "emoglobina", "colesterolo"]

nome = "Olivia"
cognome = "Jackson"
eta = 21
peso = 70
analisi = ["emocromo", "emoglobina"]

nome = "Alessandro"
cognome = "Morra"
eta = 18
peso = 75
analisi = ["emocromo", "emoglobina", "colesterolo"]

class Paziente: 

    def __init__(self, nome, cognome, codice_fiscale, eta, peso, analisi_effettuate, risultati_analisi): 
        self.nome = nome
        self.cognome = cognome
        self.codice_fiscale = codice_fiscale
        self.eta = eta
        self.peso = peso
        self.analisi_effettuate = analisi_effettuate
        self.risultati_analisi = risultati_analisi

    def scheda_personale(self):
        print("\nIl paziente:", self.nome, self.cognome, ". Codice fiscale:", self.codice_fiscale)
        print("eta =", self.eta, "peso =", self.peso, "Kg", "\nAnalisi Effettuate:", self.analisi_effettuate)

    def statistiche_analisi(self):
        print("\nPaziente", self.nome , self.cognome)
        print("Valore medio:", np.mean(self.risultati_analisi))
        print("Valore massimo:", np.max(self.risultati_analisi))
        print("Valore minimo:", np.min(self.risultati_analisi))
        print("Deviazione standard valori:", np.std(self.risultati_analisi))


class Medico:
    def __init__(self, nome, cognome, specializzazione): 
        self.nome = nome
        self.cognome = cognome
        self.specializzazione = specializzazione

    def visita_paziente(self, paziente): 
        print(f"Il Medico {self.nome} {self.cognome} sta visitando {paziente.nome} {paziente.cognome}.")


class Analisi:

    valori_emocromo = (100,300)
    valori_colesterolo = (10, 70)
    valori_emoglobina = (100,1000)

    def __init__(self, tipo, valore): 
        self.tipo = tipo
        self.valore = valore

    def valuta(self):
        if self.tipo == "emocromo": 
            valori = Analisi.valori_emocromo

        elif self.tipo == "colesterolo":
            valori = Analisi.valori_colesterolo

        elif self.tipo == "emoglobina": 
            valori = Analisi.valori_emoglobina
        
        else:
            print("There is no values signed for this analysis: ", self.tipo)

        if self.valore in valori: 
            print("Risultato Valori nella norma")
        else: 
            print("Valori fuori norma !")

    
# Generate random float numbers between 0 and 100. These are the results of the exams done on patients. 
valori_arr = np.random.rand(10)*100

print("\nValori degli esami: ", valori_arr)
print("\nValore medio: ", np.mean(valori_arr))
print("\nValore massimo: ", np.max(valori_arr))
print("\nValore minimo: ", np.min(valori_arr))
print("\nDeviazione standard valori: ", np.std(valori_arr))


# MAIN:

Dott_Reed = Medico("Richard", "Reed", "Oncologia")
Dott_Torchio = Medico("Edoardo", "Torchio", "Neurologia")
Dott_Fernicola = Medico("Francesco", "Fernicola", "Cardiologia")

P1 = Paziente("Stefano", "Morra", "MRRSFN", 24, 85, ["emocromo", "emoglobina", "colesterolo"],[200,50,500])
P2 = Paziente("Alessandro", "Morra", "ALSSFN", 18, 75, ["emocromo", "emoglobina", "colesterolo"],[100,100,500])
P3 = Paziente("Johhny", "Storm", "JHNSTM", 30, 70, ["emocromo", "emoglobina", "colesterolo"],[200,2,1000])
P4 = Paziente("Susan", "Storm", "SSNSTM", 28, 60, ["emocromo", "emoglobina", "colesterolo"],[500,50,500])
P5 = Paziente("Victor", "Von Doom", "VCRVND", 24, 85, ["emocromo", "emoglobina", "colesterolo"],[500,500,1500])

P1.scheda_personale()
P2.scheda_personale()
P3.scheda_personale()
P4.scheda_personale()
P5.scheda_personale()

Dott_Fernicola.visita_paziente(P1)
Dott_Fernicola.visita_paziente(P2)
Dott_Reed.visita_paziente(P3)
Dott_Torchio.visita_paziente(P4)
Dott_Torchio.visita_paziente(P5)

P1.statistiche_analisi()
P2.statistiche_analisi()
P3.statistiche_analisi()
P4.statistiche_analisi()
P5.statistiche_analisi()