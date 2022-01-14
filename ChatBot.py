import random
import json
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def Token(sentence):
    # luam propozitia si o impartim in cuvinte, si semne de punctuatie,
    # toate astea sunt puse intr un vector de cuvinte
    # practic asta face nltk.word_tokenize(<fraza pe care vrem sa o spargem in tokeni>)

    return nltk.word_tokenize(sentence)


def Lower(word):
    # pentru a nu lua in calcul toate variatiile unui cuvant, ii luam doar radacina cuvantului, partea comuna
    # pentru a reduce si mai mult cazurile, daca un cuvant contine litere mari sau e scris cu caps,
    # facem ca toate cuvintele din vectorul rezultant sa fie returnate cu litera mica,
    # pentru a evita niste variatii redundante

    return stemmer.stem(word.lower())


def Words(sentence, words):
     # dupa ce avem cuvintele trunchiate la radacina, creem un np_array
    # umplut cu atatia de 0 cate cuvinte unice sunt in total in fisierul json
    # si parcurgem propozitia primita sa vedem care cuvinte din propozitie sunt parte din fisierul json
    # daca fac parte, in array ul nostru punem 1 pe pozitia cuvantului regasit.
    # la final vom avea un np_array de 0 si 1, in care 1 reprezinta ca unul din cuvintele
    # din propozitia data ca input se afla pe pozitia i din cuvintele din json

    # facem cu litere mici fiecare cuvant
    sentence_w = [Lower(word) for word in sentence]
    # initializam vectorul cu 0 pentru fiecare cuvant
    new_words = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_w:
            new_words[idx] = 1

    return new_words

# feedforward neural network cu 2 hidden layere
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # apelam supermetoda
        super(NeuralNet, self).__init__()
        # 3 liniar layere
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        #o functie de activare pentru atunci cand sunt intre layere
        self.relu = nn.ReLU()

    def forward(self, x):
        #functia relu returneaza inputul daca acesta >= 0 si 0 daca e negativ
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#deschidem fila de json pt a citi din ea
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
#extragem informatiile create in CreateFile.py
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
model_state = data["model_state"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']


#antrenam reteaua neuronala pe dispozitiv
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BOT"
print()
print("                ChatBot")
print()
print("Chatbot-ul este gata de utilizare")
print("Tasteaza 'Exit' pentru a inchide programul.")
print()

while True:
    sentence = input("You: ")
    if sentence == "Exit" or sentence=="exit":
        break
    # impartim propozitia in cuvinte separate
    sentence = Token(sentence)
    # toate cuvintele pe care le avem din fisierul creat
    X = Words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    # deoarece bow returneaza un numpy array avem nevoie de torch
    X = torch.from_numpy(X).to(device)

    output = model(X)

    # predictia maxima
    var, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # calculeaza exponențialul fiecărui element împărțit la suma exponențialelor tuturor elementelor
    probs = torch.softmax(output, dim=1)
    # calculam probabilitatea pentru fiecare raspuns
    prob = probs[0][predicted.item()]

    if 0.75 <prob.item() :
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                print()
    else:
        print(f"{bot_name}: I do not understand.")