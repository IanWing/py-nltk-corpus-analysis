#!/usr/bin/python
#-*- coding: utf-8 -*-
import nltk
import codecs
import sys
from collections import Counter
import re
#Importo le librerie necessarie.

#questo programma tende a mantenere una struttura di esecuzione simile a quella del primo (definizione di funzioni, definizione di main con print, definizione di classi)

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle") 

def ie_preprocess(x):      #Parametro: stringa testo                #processo il documento per ottenere FRASI PoS taggate
    sentences = nltk.sent_tokenize(x)                               #converti in frasi
    sentences = [nltk.word_tokenize(sent) for sent in sentences]    #converti in token
    sentences = [nltk.pos_tag(sent) for sent in sentences]          #PoS tagga
    return sentences

def extract_names(x):    #Parametro: frasi PoS taggate              #Estrae i nomi d'interesse
    names = []                                                      #variabile temporanea
    sentences = x                                                   #variabile contenente lo stesso input di funzione
    for tagged_sentence in sentences:                               #per ogni frase taggata...
        for chunk in nltk.ne_chunk(tagged_sentence):                #per ogni chunck (entità grammaticale collettiva che costituisce un tag)...
            if type(chunk) == nltk.tree.Tree:                       #se il chunk è composto da più elementi in struttura alberata...
                if chunk.label() == 'PERSON':                       #se il label (etichetta) del chunk è PERSON...
                    names.append(' '.join([c[0] for c in chunk]))   #aggiungi a names ogni elemento del chunk nella stessa stringa (["abc"],["xyz"],...)
    freq = Counter(names)                                           #conta la frequenza degli elementi in names
    top10 = freq.most_common(10)                                    #seleziona i 10 più frequenti
    result = []                                                     #variabile temporanea di ritorno
    for element in top10:                                           #per ogni elemento in top10...
        result.append(element[0])                                   #aggiungi elemento a risultato
    return result

def trovaFrasi(frasi,nomi):                 #trova le frasi con all'interno i nomi di rilevanza
    dic = dict((el,[]) for el in nomi)      #crea un dizionario con x chiavi, dove x è pari al numero di elementi in nomi, e il valore di x è pari al valore del nome
    tmpmax = 0                              #variabile temporanea
    tmpmin = 99999                          #variabile temporanea
    for nome in nomi:                       #per ogni nome della lista dei nomi...
        for frase in frasi:                 # *E* per ogni frase della lista delle frasi...
            if nome in frase:               #se il nome è presente nella frase...
                dic[nome].append(frase)     #aggiungilo al dizionario dic, sotto la chiave indicata dal nome
        for value in dic[nome]:             #per ogni frase in dic con indice [nome]
            if len(value) < tmpmin:         #se la lunghezza è inferiore al valore della variabile temporanea..
                dic[nome][0] = value        #il primo valore di dic con indice [nome] diventa pari alla frase
                tmpmin = len(value)         #la variabile temporanea si aggiorna col valore della lunghezza della frase in questione
            if len(value) > tmpmax:         #questo controllo verifica l'inverso del precedente
                dic[nome][1] = value        #il secondo valore di dic con indice [nome] diventa pari alla frase
                tmpmax = len(value)
        tmpmax = 0                          #alla fine del ciclo per ogni chiave, reimposta il valore delle variabili temp
        tmpmin = 99999

    return dic                              #torna il dizionario di valori
        
def ricerca_dati(dic, Lcorp, tokens):   #parametri: dizionario con frasi, lista token   #questa funzione ricerca tutti i dati richiesti usando meno algoritmi diversi possibili
    dic_names = dict((el,[]) for el in dic.keys())                          #creo 6 dizionari, uno per ogni elemento richiesto
    dic_locations = dict((el,[]) for el in dic.keys())
    dic_verbs = dict((el,[]) for el in dic.keys())
    dic_date = dict((el,[]) for el in dic.keys())
    dic_substantive = dict((el,[]) for el in dic.keys())
    dic_markov = dict((el,"") for el in dic.keys())
    
    #definisco il pattern per l'espressione regolare che cerca date, mesi e giorni, aggiungengo più formati possibili
    pattern = r"\b(?:Janu|Febru)ary\b|\bMarch\b|\bApril\b|\bMay\b|\bJune\b|\bJuly\b|\bAugust\b|\b(?:Sept|Nov|Dec)ember\b|\bOctober\b|(?:Mon|Tue|Wednes|Thurs|Fri|Satur|Sun)day|(?:\d+/\d+/\d+)|(?:\d+-\d+-\d+)|(?:\b\d{1,2}\D{0,3})?\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(?:Nov|Dec)(?:ember)?)\D?(?:\d{1,2}\D?)?\D?(?:(?:19[7-9]\d|20\d{2})|\d{2})"

    for nome in dic.keys():             #da qui in poi, ripeti le seguenti operazioni per OGNI NOME importante...
        for value in set(dic[nome]):    # ... E PER OGNI FRASE associata a esso
            match = re.findall(pattern, value)                              #trova tutte le date con l'espressione regolare apposita, che sfrutta il pattern precedentemente definito
            for element in match:                                           #per ogni elemento in lista match...
                dic_date[nome].append(element)                              #assegna al dizionario delle date, all'indice del corrispettivo nome, ogni data trovata
                                                                            #NB: tale processo si ripete per ogni singolo nome, in modo da associare ad ogni gruppo di frasi la corrispettiva data contanuta in esse
            sentences = nltk.sent_tokenize(value)                           #dividi in frasi
            sentences = [nltk.word_tokenize(sent) for sent in sentences]    #tokenizza le parole, ma all'interno delle frasi, per mantenerne la struttura
            
            for sentence in sentences:                                      #per ogni frase...
                probMax = 0                                                 #variabile temporanea
                arrayProbs = []                                             #array temporaneo
                result = 1                                                  #variabile temporanea
                if len(sentence) >= 8 and len(sentence) <= 12:              #se una frase è composta da almeno 8 e massimo 12 tokens...
                    for parola in sentence:                                 #per ogni parola nella frase...
                        freq_parola = tokens.count(parola)                  #conta la frequenza di ogni parola
                        prob_parola = freq_parola*1.0/Lcorp                 #calcola la probabilità
                        arrayProbs.append(prob_parola)                      #aggiungi probabilità alla variabile temporanea
                        for element in arrayProbs:                          #moltiplica tutti i valori della lista
                            result = result * element
                        if result > probMax:                                #se il risultato della moltiplicazione supera la probabilità massima rilevata fin ora...
                            dic_markov[nome] = sentence                     #assegna la frase al dizionario
                            probMax = result                                #aggiorna la probabilità massima
            
            sentences = [nltk.pos_tag(sent) for sent in sentences]          #PoS-tagga ogni parola nella frase, mantenendone la struttura
            for sentence in sentences:                                      #per ogni frase...                             
                for parola in sentence:                                     #... E per ogni parola
                    if parola[1][0:2] == "VB":                              #se tale parola è taggata come un qualsiasi verbo...
                        dic_verbs[nome].append(parola[0])                   #aggiungi al corrispettivo dizionario di valori, in chiave [nome]
                    if parola[1][0:2] == "NN":                              #se tale parole è taggata come un qualsiasi sostantivo...
                        dic_substantive[nome].append(parola[0])             #aggiungi al corrispettivo dizionario di valori, in chiave [nome]
            for tagged_sentence in sentences:                               #per ogni frase PoS taggata...
                for chunk in nltk.ne_chunk(tagged_sentence):                #e quindi, per ogni chunk, della lista di tali frasi taggate...
                    if type(chunk) == nltk.tree.Tree:                       #se il chunk è composto da più elementi in struttura alberata...
                        if chunk.label() == 'PERSON':                       #se il label del chunk è PERSON...
                            dic_names[nome].append(' '.join([c[0] for c in chunk]))     #aggiungi al corrispettivo dizionario la concatenazione degli elementi del chunk
                        if chunk.label() == 'GPE':                                      #se il label del chunk è GPE (Geo-Political Entity)...
                            dic_locations[nome].append(' '.join([c[0] for c in chunk])) #aggiungi al corrispettivo dizionario la concatenazione degli elementi del chunk
    #di seguito, PER OGNI VOCABOLARIO, e PER OGNI CHIAVE DI ESSO, verranno estratte le frequenze degli elementi e quindi selezionati i 10 valori più frequenti, in sequenza ordinata
    for nome in dic_verbs.keys():
        freq = Counter(dic_verbs[nome])
        dic_verbs[nome] = freq.most_common(10)

    for nome in dic_locations.keys():
        freq = Counter(dic_locations[nome])
        dic_locations[nome] = freq.most_common(10)

    for nome in dic_substantive.keys():
        freq = Counter(dic_substantive[nome])
        dic_substantive[nome] = freq.most_common(10)

    for nome in dic_names.keys():
        freq = Counter(dic_names[nome])
        dic_names[nome] = freq.most_common(10)

    for nome in dic_names.keys():
        freq = Counter(dic_date[nome])
        dic_date[nome] = freq.most_common(10)
    #nel caso della probabilità della frase, viene semplicemente creata un'unica stringa per la frase più probabile
    for nome in dic_markov.keys():
        frase = ""
        i = 0
        for element in dic_markov[nome]:
            frase =  frase + dic_markov[nome][i] + " "
            i = i + 1
        dic_markov[nome] = frase

    return dic_names, dic_locations, dic_verbs, dic_substantive, dic_date, dic_markov   #ritorna tutte le strutture dati

class file1:        #definizione di variabili in classe file1
    file_name = sys.argv[1]                                         #stringa contentente il nome del file
    file_input = codecs.open(file_name, "r", "utf-8")               #apertura del file
    file_string = file_input.read()                                 #lettura
    phrases_list = sent_tokenizer.tokenize(file_string)             #lista delle frasi
    tokens_list = nltk.word_tokenize(file_string)                   #lista dei tokens
    phrases_tagged_list= ie_preprocess(file_string)                 #lista di frasi con tag
    important_names_list = extract_names(phrases_tagged_list)       #lista di nomi di rilevanza
    dic_frasi = trovaFrasi(phrases_list, important_names_list)      #dizionario con frasi ordinate per nome
    lunghezza_corpus = len(tokens_list)                             #int di lunghezza corpus
    #dizionari con vari valori ordinati
    dic_nomi, dic_luoghi, dic_verbi, dic_sostantivi, dic_date, dic_markov = ricerca_dati(dic_frasi, lunghezza_corpus, tokens_list)

class file2:    #stessi valori della classe file1, eccetto il file di input
    file_name = sys.argv[2]
    file_input = codecs.open(file_name, "r", "utf-8")
    file_string = file_input.read()
    phrases_list = sent_tokenizer.tokenize(file_string)
    tokens_list = nltk.word_tokenize(file_string)
    phrases_tagged_list= ie_preprocess(file_string)
    important_names_list = extract_names(phrases_tagged_list)
    dic_frasi = trovaFrasi(phrases_list, important_names_list)
    lunghezza_corpus = len(tokens_list)
    dic_nomi, dic_luoghi, dic_verbi, dic_sostantivi, dic_date, dic_markov = ricerca_dati(dic_frasi, lunghezza_corpus, tokens_list)

def main(): #la funzione main stampa semplicemente i risultati in maniera ordinata. Non ci sono calcoli inediti, solo semplice formattazione dell'output
    punto = "### Punto 1: ###"
    o = punto.center(100)
    print("\n" + o)
    print ("Nel file "+file1.file_name + " i 10 nomi di persona più frequenti rilevati sono:")
    i = 1
    for key in sorted(file1.dic_frasi.keys()):
        print (str(i)+ "\t"+ str(key))
        i = i+1
    else: print ("")

    for key in sorted(file1.dic_frasi.keys()):
        print("Frase più corta per " + key.encode('utf-8') + ":\n" + "<## " + file1.dic_frasi[key][0].encode('utf-8')+" ##>\n")

    for key in sorted(file1.dic_frasi.keys()):
        print("Frase più lunga per " + key.encode('utf-8') + ":\n" + "<## " + file1.dic_frasi[key][1].encode('utf-8')+" ##>\n")

    punto = "### Punto 2: ###"
    o = punto.center(100)
    print("\n" + o)
    print ( "______________________________________________________________________________________________________________________")
    print ("\t\tTutti i parametri richiesti per " + file1.file_name + " visualizzati in blocchi per:")
    print ( "______________________________________________________________________________________________________________________")
    punto = "###### I (10) luoghi più frequnti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_luoghi.keys():
        if len(file1.dic_luoghi[key])<10 and len(file1.dic_luoghi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file1.dic_luoghi[key])) + " luoghi:")
        elif len(file1.dic_luoghi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo luogo:")
        elif len(file1.dic_luoghi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati luoghi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 luoghi rilevati più frequenti sono:")
        i = 0
        for value in file1.dic_luoghi[key]:
            print(str(i+1) + "\t" + str(file1.dic_luoghi[key][i][0]) + "\t con frequenza pari a: " + str(file1.dic_luoghi[key][i][1]))
            i = i + 1
        else:
            print("")
            
    punto = "###### I (10) nomi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_nomi.keys():
        if len(file1.dic_nomi[key])<10 and len(file1.dic_nomi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file1.dic_nomi[key])) + " nomi:")
        elif len(file1.dic_nomi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo nome:")
        elif len(file1.dic_nomi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati nomi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 nomi rilevati più frequenti sono:")
        i = 0
        for value in file1.dic_nomi[key]:
            print(str(i+1) + "\t" + str(file1.dic_nomi[key][i][0]) + "\t con frequenza pari a: " + str(file1.dic_nomi[key][i][1]))
            i = i + 1
        else: print("")

    punto = "###### I (10) verbi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_verbi.keys():
        if len(file1.dic_verbi[key])<10 and len(file1.dic_verbi[key])>0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file1.dic_verbi[key])) + " verbi:")
        elif len(file1.dic_verbi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati verbi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 verbi più frequenti sono:")
        i = 0
        for value in file1.dic_verbi[key]:
            print(str(i+1) + "\t" + str(file1.dic_verbi[key][i][0]) + "\t con frequenza pari a: " + str(file1.dic_verbi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### I (10) sostantivi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_sostantivi.keys():
        if len(file1.dic_sostantivi[key])<10 and len(file1.dic_sostantivi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file1.dic_sostantivi[key])) + " sostantivi:")
        elif len(file1.dic_sostantivi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo sostantivi:")
        elif len(file1.dic_sostantivi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati sostantivi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 sostantivi rilevati più frequenti sono:")
        i = 0
        for value in file1.dic_sostantivi[key]:
            print(str(i+1) + "\t" + str(file1.dic_sostantivi[key][i][0]) + "\t con frequenza pari a: " + str(file1.dic_sostantivi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### Tutte le date, mesi e giorni della settimana ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_date.keys():
        if len(file1.dic_date[key]) == 0:
                print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non ci sono date!\n")
        else:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > tutte le date trovate sono:")
            i = 0
            for value in file1.dic_date[key]:
                print(str(i+1) + "\t" + str(file1.dic_date[key][i][0]) + "\t con frequenza pari a: " + str(file1.dic_date[key][i][1]))
                i = i + 1
            else: print("")

    
    punto = "###### La frase più probabile secondo la catena markoviana di ordine 0 ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file1.dic_markov.keys():
        print ("Per la parola: " + key.encode("utf-8") + " la frase con probabilità maggiore è :")
        print (">\t<## " + file1.dic_markov[key] + " ##>\n")

    ######################################################################################################################################################################################

    print ( "______________________________________________________________________________________________________________________")
    print ("\t\tTutti i parametri richiesti per " + file2.file_name + " visualizzati in blocchi per:")
    print ( "______________________________________________________________________________________________________________________")
    punto = "###### I (10) luoghi più frequnti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_luoghi.keys():
        if len(file2.dic_luoghi[key])<10 and len(file2.dic_luoghi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file2.dic_luoghi[key])) + " luoghi:")
        elif len(file2.dic_luoghi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo luogo:")
        elif len(file2.dic_luoghi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati luoghi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 luoghi rilevati più frequenti sono:")
        i = 0
        for value in file2.dic_luoghi[key]:
            print(str(i+1) + "\t" + str(file2.dic_luoghi[key][i][0]) + "\t con frequenza pari a: " + str(file2.dic_luoghi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### I (10) nomi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_nomi.keys():
        if len(file2.dic_nomi[key])<10 and len(file2.dic_nomi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file2.dic_nomi[key])) + " nomi:")
        elif len(file2.dic_nomi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo nome:")
        elif len(file2.dic_nomi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati nomi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 nomi rilevati più frequenti sono:")
        i = 0
        for value in file2.dic_nomi[key]:
            print(str(i+1) + "\t" + str(file2.dic_nomi[key][i][0]) + "\t con frequenza pari a: " + str(file2.dic_nomi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### I (10) verbi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_verbi.keys():
        if len(file2.dic_verbi[key])<10 and len(file2.dic_verbi[key])>0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file2.dic_verbi[key])) + " verbi:")
        elif len(file2.dic_verbi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati verbi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 verbi più frequenti sono:")
        i = 0
        for value in file2.dic_verbi[key]:
            print(str(i+1) + "\t" + str(file2.dic_verbi[key][i][0]) + "\t con frequenza pari a: " + str(file2.dic_verbi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### I (10) sostantivi più frequenti ######"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_sostantivi.keys():
        if len(file2.dic_sostantivi[key])<10 and len(file2.dic_sostantivi[key])>1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > sono stati rilevati solo " + str(len(file2.dic_sostantivi[key])) + " sostantivi:")
        elif len(file2.dic_sostantivi[key]) == 1:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > è stato rilevato un solo sostantivi:")
        elif len(file2.dic_sostantivi[key]) == 0:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non sono stati rilevati sostantivi!")
        else:
            print ("Per la parola: " + key.encode("utf-8") + " i 10 sostantivi rilevati più frequenti sono:")
        i = 0
        for value in file2.dic_sostantivi[key]:
            print(str(i+1) + "\t" + str(file2.dic_sostantivi[key][i][0]) + "\t con frequenza pari a: " + str(file2.dic_sostantivi[key][i][1]))
            i = i + 1
        else:
            print("")

    punto = "###### Tutte le date, mesi e giorni della settimana ######\n"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_date.keys():
        if len(file2.dic_date[key]) == 0:
                print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > non ci sono date!\n")
        else:
            print ("Nelle frasi in cui compare la parola < " + key.encode("utf-8") + " > tutte le date trovate sono:")
            i = 0
            for value in file2.dic_date[key]:
                print(str(i+1) + "\t" + str(file2.dic_date[key][i][0]) + "\t con frequenza pari a: " + str(file2.dic_date[key][i][1]))
                i = i + 1
            else: print ("")

    punto = "###### La frase più probabile secondo la catena markoviana di ordine 0 ######\n"
    o = punto.center(100)
    print("\n" + o)
    for key in file2.dic_markov.keys():
        print ("Per la parola: " + key.encode("utf-8") + " la frase con probabilità maggiore è :")
        print (">\t<## " + file2.dic_markov[key] + " ##>\n")

main() #invoca main

# ~ Christian Attanasio