# Parte 1 - Variabili e Tipi di Dati

Titolo = "Odissea"
Numero_copie = 10
Prezzo_medio = 15
Stato = True
if Stato == True: 
    Disponibilità = "Disponibile"
else: 
    Disponibilità = "Non Disponibile"

print(f" Titolo: {Titolo} - Numero Copie: {Numero_copie} - Prezzo : {Prezzo_medio} € - Disponibilità: {Disponibilità}")

# Parte 2 - Strutture Dati
Libri = ["Odissea", "Divina Commedia", "I Promessi Sposi", "Harry Potter", "Il Signore degli Anelli"]
Libri_Dict = {"Odissea": 10, "Divina Commedia": 15, "I Promessi Sposi": 7, "Harry Potter" : 5, "Il Signore degli Anelli": 2}
Utenti_set = {"Marco", "Marco", "Giulia", "Luca", "Stefano", "Alessandro", "Stefano", "Antonio", "Marco"}

# Parte 3 - Classi e OOP
class Libro(): 
    def __init__(self, titolo : str, autore: str, anno: int, copie_disponibili: int): 
        self.titolo = titolo
        self.autore = autore
        self.anno = anno
        self.copie_disponibili = copie_disponibili
        
    def info(self): 
        print(f" Il Libro : {self.titolo} dell'autore : {self.autore} è stato pubblicato nel {self.anno}. In biblioteca sono presenti {self.copie} copie.")

class Utente: 
    def __init__(self, nome: str, eta: int, ID: int): 
        self.nome = nome
        self.eta = eta
        self.ID = ID

    def scheda(self): 
        print(f"Utente: {self.nome}, {self.eta} anni. ID: {self.ID}")


class Prestito: 
    def __init__(self, utente: str, libro: str, giorni_prestito:int): 
        self.utente = utente
        self.libro = libro
        self.giorni_prestito = giorni_prestito

    def dettagli(self): 
        print(f"Prestito: Utente {self.utente} - Libro {self.libro} - Giorni Prestito {self.giorni_prestito}")

# Parte 4 - Funzionalità
def presta_libro(utente, libro, giorni): 
    if libro.copie_disponibili > 0: 
        libro.copie_disponibili -= 1
        print(f" Presito di {libro.titolo} a {utente.nome} avvenuto con successo !")
        return Prestito(utente, libro, giorni)
        
    else: 
        print(f"La bibliotecsa non ha più copie del libro: {libro.titolo}")

Utenti = []
Utenti.append(Utente("Stefano", 24, 335771))
Utenti.append(Utente("Marco", 20, 390876))
Utenti.append(Utente("Luisa", 50, 189765))

Libri = []
Libri.append(Libro("Harry Potter", "J.K. Rowling", 1991, 5))
Libri.append(Libro("Odissea", "Omero", -300, 3))
Libri.append(Libro("Divina Commedia", "Dante Alighieri", 1200, 10))

for i in Libri: 
    print(f"Libro: {i.titolo} - Copie Disponibili {i.copie_disponibili}")

presta_libro(Utenti[1], Libri[2], 10)
presta_libro(Utenti[2], Libri[0], 2)
presta_libro(Utenti[0], Libri[1], 20)

for i in Libri: 
    print(f"Libro: {i.titolo} - Copie Disponibili {i.copie_disponibili}")



