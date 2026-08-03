#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import codecs
import math
import nltk
from nltk.util import bigrams
#Importo le librerie necessarie.


#La seconda parte di main() è composta dalle dichiarazione di tutte funzioni necessarie.
#Le funzioni non processano due testi in parallelo, ma sono ideate per processare ogni testo, più volte.
#Questa è la mente del programma, dove vengono svolte tutte le operazioni. 

def numeroFrasi(x): #Parametro: lista delle frasi
    count = len(x)  #Assegna a count l'intero pari al numero di elementi di x
    return count

def contaTokens(x): #Parametro: lista dei tokens
    count = len(x)  #Assegna a count l'intero pari al numero di elementi di x
    return count

def lungMediaFrasi(x,y):        #Parametro: numero tokens, numero frasi
    result = (x*1.0)/(y*1.0)
    return ("%.2f" % result)

def contaCaratteri(x):              #Parametro: numero tokens
    count = 0
    for token in x:                 #Per ogni carattere di ogni token...
        count = count + len(token)  #...aggiungi 1 a count
    return count

def lungMediaTokens(x,y):           #Parametro: numero caratteri, numero tokens
    result = (x*1.0)/(y*1.0)
    return ("%.2f" % result)

def vocabolarioParziale(x):                         #Parametro: lista tokens
    i = 1000                                        #variabile contatore
    listaVocabolario = []                           #lista di output
    while i < len(x):                               #fintanto che il contatore è minore di |corpus|...
        limit = x[0:i]                              #...limita il corpus dal primo al i-esimo token  
        listaVocabolario.append(len(set(limit)))    #...rimuovi le ripetizioni ed estrai la cardinalità
        i = i + 1000                                #...aumenta il contatore di 1000
    else:                                           #se i è maggiore di |corpus|
        listaVocabolario.append(len(set(x)))        #...rimuovi le ripetizioni del corpus
    return listaVocabolario

def contaHapax(x,y):                        #Parametro: lista token, contatore range-1000
    listaFrequenze = nltk.FreqDist(x[0:y])  #estrai le frequenze
    return len(listaFrequenze.hapaxes())    #trova gli hapax e converti in numero

def rapporto(x):                                            #Parametro: corpus di PoS. Rapporto tra verbi e sostantivi.
    counterVerbi = 0                                        
    counterSostantivi = 0
    for element in x:                                       #per ogni elemento PoS
        if (element[1][0:2] == "VB"):                       #se è un verbo (ogni tipo)
            counterVerbi = counterVerbi + 1                 #aggiungi 1 al contatore
        if (element[1][0:2] == "NN"):                       #se è un sostantivo (ogni tipo)
            counterSostantivi = counterSostantivi +1        #aggiungi 1 
    result = (counterVerbi*1.0) / (counterSostantivi*1.0)   
    return ("%.2f" % result)                                #riduci a 2 decimali

def posChart(x):                                            #Parametro: Corpus PoS. Trova i 10 PoS più frequenti.
    varControllo = nltk.FreqDist(tag for (word, tag) in x)  #estrai le frequenze per ogni tag in in Corpus PoS
    oggettoPos = varControllo.most_common(10)               #estrai solo i 10 più frequenti
    listaPos = []
    listaPosF = []
    for element in oggettoPos:                              #per ogni elemento dei 10
        listaPos.append(element[0])                         #assegna il PoS alla lista
        listaPosF.append(element[1])                        #assegna la frequenza alla lista
    return listaPos, listaPosF
    
def creaListaPOS(x):            #Parametro: corpus PoS. Crea un elemento list con soli PoS
    POS = []
    for element in x:
        POS.append(element[1])  
    return POS

def probCond(x, y):                                                 #Parametro: lista bigrammi PoS, lista PoS
    DicBigrammi = {}                                                #crea un elemento dizionario
    for element in set(x) :                                         #per ogni bigramma PoS...
        freqBigramma = x.count(element)                             #trova la sua frequenza
        freqPOS = y.count(element[0])                               #trova la frequenza del PoS
        probCond = "%.2f" %((freqBigramma * 1.0)/(freqPOS * 1.0))   #applica la formula
        DicBigrammi[element] = probCond                             #per ogni bigramma assegna un valore
    frequenze = nltk.FreqDist(DicBigrammi)                          #estrai le frequenze
    result = frequenze.most_common(10)                              #estrai le 10 più frequenti 
    listaProb = []
    listaBigPC = []
    for element in range(10):                                       #assegna
        listaBigPC.append(result[element][0])
        listaProb.append(result[element][1])
    return listaBigPC, listaProb

def LocalMutualInformation(x, y):                                           #Parametro: lista bigrammi PoS, lista token
    freqBigrammi = nltk.FreqDist(x)                                         #estrai le frequenze dei bigrammi
    listaFrequenze = freqBigrammi.most_common(len(x))                       #usa i risultati dell'oggetto FreqDist per farci una lista di frequenze           
    dic_big = dict((el[0],el[1]) for el in listaFrequenze)                  #crea un dizionario di bigrammi
    dic_lmi = dict((el[0],0) for el in listaFrequenze)                      #crea un dizionario di valori
    for bigramma in dic_big.keys():                                         #per ogni bigramma...
        freqBigramma = dic_big[bigramma]                                    #applica alla variabile freqBigramma il valore pari alla frequenza del bigramma dal dizionario
        probBig = freqBigramma*1.0/len(y)                                   #calcola la probabilità del bigramma
        frequenzaA = y.count(bigramma[0])                                   #trova la frequenza del primo token del bigramma
        probA = (frequenzaA*1.0)/len(y)                                     #calcolane la probabilità
        frequenzaB = y.count(bigramma[1])                                   #trova la frequenza del secondo token del bigramma
        probB = (frequenzaB*1.0)/len(y)                                     #calcolane la probabilità
        LMI = freqBigramma*math.log(probBig/(probA*probB),2)                #applica la formula della Local Mutual Information
        dic_lmi[bigramma] = LMI                                             #applica al dizionario dedicato alla LMI, per ogni bigramma, la propria LMI
    lista_Ordinata = (sorted(dic_lmi.items(), key=lambda item: -item[1]))   #ordina le chiavi del dizionario in base al valore delle chiavi
    return lista_Ordinata
    

#Dichiaro la funzione madre main().
def main():

    #La funzione main() è composta unicamente di invocazioni e print(), in modo da avere una struttura output già pronta.
    #Oltre alle chiamate ci sono semplici cicli per la formattazione delle stringhe. Il codice è abbastanza autoesplicativo.
    #Rappresente il corpo fisico del programma.
    punto = "### Punto 1: ###"
    o = punto.center(100)
    print(o)
    print("Numero di frasi in " + file1.fileName + ": " + str(file1.numeroFrasi) + "\nNumero frasi in " + file2.fileName + ": " +str(file2.numeroFrasi))
    if file1.numeroFrasi > file2.numeroFrasi:
        print(file1.fileName + " ha più frasi di " + file2.fileName+".")
    elif file1.numeroFrasi < file2.numeroFrasi:
        print(file1.fileName + " ha meno frasi di " + file2.fileName+".")
    else:
        print(file1.fileName + " e " + file2.fileName +" hanno lo stesso numero di frasi.")

    print("\nNumero di tokens in " + file1.fileName + ": " + str(file1.numeroTokens) + "\nNumero tokens in " + file2.fileName + ": " +str(file2.numeroTokens))
    if file1.numeroTokens > file2.numeroTokens:
        print(file1.fileName + " ha più tokens di "+ file2.fileName+".")
    elif file1.numeroTokens < file2.numeroTokens:
        print(file1.fileName +" ha meno tokens di " + file2.fileName+".")
    else:
        print(file1.fileName + " e " + file2.fileName +" hanno lo stesso numero di tokens.\n")

    punto = "### Punto 2: ###"
    o = punto.center(100)
    print("\n" + o)
    print("Lunghezza media nelle frasi in " + file1.fileName + ": "  + str(file1.lunghezzaMediaFrasi) + "\nLunghezza media delle frasi in " + file2.fileName + ": " + str(file2.lunghezzaMediaFrasi))
    if file1.lunghezzaMediaFrasi > file2.lunghezzaMediaFrasi:
        print(file1.fileName + " ha frasi più lunghe di " + file2.fileName+".\n")
    elif file1.lunghezzaMediaFrasi < file2.lunghezzaMediaFrasi:
        print(file1.fileName + " ha frasi meno lunghe di " + file2.fileName+".\n")
    else:
        print(file1.fileName + " e " + file2.fileName +" hanno frasi di pari lunghezza.\n")
    
    print("\nLunghezza media dei tokens in " + file1.fileName + ": "  + str(file1.lunghezzaMediaTokens) + "\nLunghezza media dei tokens in " + file2.fileName + ": " + str(file2.lunghezzaMediaTokens))
    if file1.lunghezzaMediaTokens > file2.lunghezzaMediaTokens:
        print(file1.fileName + " ha tokens più lunghi di " + file2.fileName+".\n")
    elif file1.lunghezzaMediaTokens < file2.lunghezzaMediaTokens:
        print(file1.fileName + " ha tokens meno lunghi di " + file2.fileName +".\n")
    else:
        print(file1.fileName + " e " + file2.fileName +" hanno tokens di pari lunghezza.\n")

    punto = "### Punto 3: ###"
    o = punto.center(100)
    print(o)
    i  = 1
    print("Per " + file1.fileName + ":")
    while i < len(file1.vocabolario):
        print("Per " + str(i*1000) + " tokens:\tVocabolario: " + str(file1.vocabolario[i-1]) + " - Distribuzione hapax: " + "%.3f" %float(contaHapax(file1.listaTokens, i*1000)/float(file1.numeroTokens)))
        i = i + 1
    else:
        print("Per " + str(file1.numeroTokens) + " tokens:\tVocabolario: " + str(file1.vocabolario[i-1]) + " - Distribuzione hapax: " + "%.3f" %float(contaHapax(file1.listaTokens, len(file1.listaTokens))/float(file1.numeroTokens))+"\n")
    print("Per " + file2.fileName + ":")
    i  = 1
    while i < len(file2.vocabolario):
        print("Per " + str(i*1000) + " tokens:\tVocabolario: " + str(file2.vocabolario[i-1]) + " - Distribuzione hapax: " + "%.3f" %float(contaHapax(file2.listaTokens, i*1000)/float(file2.numeroTokens)))
        i = i + 1
    else:
        print("Per " + str(file2.numeroTokens) + " tokens:\tVocabolario: " + str(file2.vocabolario[i-1]) + " - Distribuzione hapax: " + "%.3f" %float(contaHapax(file2.listaTokens, len(file2.listaTokens))/float(file2.numeroTokens))+"\n")

    punto = "### Punto 4: ###"
    o = punto.center(100)
    print(o)
    print("Il rapporto tra verbi e sostantivi di " + file1.fileName + " è pari a " + str(rapporto(file1.corpusPOS)))
    print("Il rapporto tra verbi e sostantivi di " + file2.fileName + " è pari a " + str(rapporto(file2.corpusPOS))+ "\n")

    punto = "### Punto 5: ###"
    o = punto.center(100)
    print(o)
    print("Le PoS più frequenti in " + file1.fileName + " sono: ")
    for element in range(10):
        print(str(element+1) + ".\t"+str(file1.posFrequenti[element]) + "\tcon frequenza: " + str(file1.frequenzaPos[element]))
    
    print("\nLe PoS più frequenti in " + file2.fileName + " sono: ")
    for element in range(10):
        print(str(element+1) + ".\t"+str(file2.posFrequenti[element]) + "\tcon frequenza: " + str(file2.frequenzaPos[element]))
    
    punto = "### Punto 6: ###"
    o = punto.center(100)
    print(o)
    print("I 10 bigrammi PoS con la maggiore probabilità condizionata di " + file1.fileName + " sono: ")
    for element in range(10):
        print(str(element+1) + ".\t"+str(file1.listaPOSpc[element]) + "\tcon frequenza: " + str(file1.probCond[element]))
    print("\nI 10 bigrammi di PoS con la maggiore probabilità condizionata di " + file2.fileName + " sono: ")
    for element in range(10):
        print(str(element+1) + ".\t"+str(file2.listaPOSpc[element]) + "\tcon frequenza: " + str(file2.probCond[element]))

    print("\nI 10 bigrammi di PoS con la maggiore forza associativa di " + file1.fileName + " sono: ")
    for element in range(10):
        print(str(element+1) + ".\t"+str(file1.dicLMI[element][0]) + "\tcon forza associativa pari a: " +str(file1.dicLMI[element][1]))
    print("\nI 10 bigrammi di PoS con la maggiore forza associativa di " + file2.fileName + " sono: ")   
    for element in range(10):
        print(str(element+1) + ".\t"+str(file2.dicLMI[element][0]) + "\tcon forza associativa pari a: " + str(file2.dicLMI[element][1]))


#la terza parte del programma è la dichiarazione delle variabili. 
#questa è l'anima del programma, che lo mette in moto nella sua interezza.

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

#all'interno delle classi vi sono storati tutti i dati necessari alla stampa e alla invocazione.
#ogni classe rappresenta un file, e ogni classe ha lo stesso set di variabili, se non per la variabile fileName.
#questa struttura è utile per confrontare più di 2 file (in caso di modifica postuma), è compatta ed è semplice da richiamare,
#poichè file1 e file2 hanno gli stessi identificatori. Al richiamo si può facilemente chiamare la stessa variabile, sotto una classe diversa.
#la scelta privilegia la semplicità strutturale e la lettura.

#I nomi delle variabili sono abbastanza auto esplicativi
class file1:                            
    fileName = sys.argv[1]                                                      #stringa contentente il nome del file 
    fileInput = codecs.open(fileName, "r", "utf-8")                             #apertura del file
    fileStringa = fileInput.read()                                              #lettura
    listaFrasi = sent_tokenizer.tokenize(fileStringa)                           #separazione in frasi
    listaTokens = nltk.word_tokenize(fileStringa)                               #separazione in tokens
    numeroFrasi = numeroFrasi(listaFrasi)
    numeroTokens = contaTokens(listaTokens)
    lunghezzaMediaFrasi = lungMediaFrasi(numeroTokens, numeroFrasi)             
    numeroCaratteri = contaCaratteri(listaTokens)
    lunghezzaMediaTokens = lungMediaTokens(numeroCaratteri, numeroTokens)
    vocabolario = vocabolarioParziale(listaTokens)
    corpusPOS = nltk.pos_tag(listaTokens)
    posFrequenti, frequenzaPos = posChart(corpusPOS)
    listaBigrammi = list(bigrams(listaTokens))
    listaPos = creaListaPOS(corpusPOS)
    listaBigrammiPOS = list(bigrams(listaPos))
    listaPOSpc, probCond = probCond(listaBigrammiPOS, listaPos)
    dicLMI = LocalMutualInformation(listaBigrammiPOS, listaPos) 

#la classe file2 contiene gli stessi valori di file1 eccetto fileName

class file2:
    fileName = sys.argv[2]
    fileInput = codecs.open(fileName, "r", "utf-8")
    fileStringa = fileInput.read()
    listaFrasi = sent_tokenizer.tokenize(fileStringa)
    listaTokens = nltk.word_tokenize(fileStringa)
    numeroFrasi = numeroFrasi(listaFrasi)
    numeroTokens = contaTokens(listaTokens) 
    lunghezzaMediaFrasi = lungMediaFrasi(numeroTokens, numeroFrasi) 
    numeroCaratteri = contaCaratteri(listaTokens)
    lunghezzaMediaTokens = lungMediaTokens(numeroCaratteri, numeroTokens)
    vocabolario = vocabolarioParziale(listaTokens)
    corpusPOS = nltk.pos_tag(listaTokens)
    posFrequenti, frequenzaPos = posChart(corpusPOS)    
    listaBigrammi = list(bigrams(listaTokens))
    listaPos = creaListaPOS(corpusPOS)
    listaBigrammiPOS = list(bigrams(listaPos))
    listaPOSpc, probCond = probCond(listaBigrammiPOS, listaPos)
    dicLMI = LocalMutualInformation(listaBigrammiPOS, listaPos)

main()  #invoca main

# ~ Christian Attanasio