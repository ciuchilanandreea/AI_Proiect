import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#deschidem fila de json pt a citi din ea
with open('intents.json', 'r') as f:
    intents = json.load(f)

# datele pentru antrenare
data_1 = []
data_2 = []

#va tine pattern urile
patt = []
#va tine tag urie
tags = []
#va tine toate pattern urile si tag urile lor
pt = []

# semnele de punctatie pe care le ignoram
ignore = ['?', '.', '!']

def Token(sentence):
    # luam propozitia si o impartim in cuvinte, si semne de punctuatie,
    # toate astea sunt puse intr un vector de cuvinte
    # practic asta face nltk.word_tokenize(<fraza pe care vrem sa o spargem in tokeni>)

    return nltk.word_tokenize(sentence)


def Lower(word):
    # pentru a nu lua in calcul toate variatiile unui cuvant, ii luam doar radacina cuvantului, partea comuna
    # pentru a reduce si mai mult cazurile, daca un cuvant contine litere mari sau e scris cu caps,
    # facem ca toate cuvintele din vectorul rezultantsa fie returnate cu litera mica,
    # pentru a evita niste variatii redundante

    return stemmer.stem(word.lower())


def Words(tokenized_sentence, words):
     # dupa ce avem cuvintele trunchiate la radacina, creem un np_array
    # umplut cu atatia de 0 cate cuvinte unice sunt in total in fisierul json
    # si parcurgem propozitia primita sa vedem care cuvinte din propozitie sunt parte din fisierul json
    # daca fac parte, in array ul nostru punem 1 pe pozitia cuvantului regasit.
    # la final vom avea un np_array de 0 si 1, in care 1 reprezinta ca unul din cuvintele
    # din propozitia data ca input se afla pe pozitia i din cuvintele din json

    # face cuvintele sa fie cu litere mici
    sentence_words = [Lower(word) for word in tokenized_sentence]
    # initializam cu 0 pentru fiecare cuvant
    new_words = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
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
        ret = self.l1(x)
        ret = self.relu(ret)
        ret = self.l2(ret)
        ret = self.relu(ret)
        ret = self.l3(ret)
        return ret


for cuv in intents['intents']:
    tag = cuv['tag']
    # adauga la lista de tag-uri
    tags.append(tag)
    for pattern in cuv['patterns']:
        # transforma fraza intr o lista de token-uri
        w = Token(pattern)
        # adauga toate pattern urile
        patt.extend(w)
        # toata lista de pattern uri cu tag uri
        pt.append((w, tag))


patt = [Lower(w) for w in patt if w not in ignore]
# stergem duplicatele si sortam
patt = sorted(set(patt))
tags = sorted(set(tags))



for (pattern_sentence, tag) in pt:
    # X: bag of words pentru fiecare propozitie
    all_worlds = Words(pattern_sentence, patt)
    data_1.append(all_worlds)
    label = tags.index(tag)
    data_2.append(label)

#le schimbam in np.array pentru a fi mai usor de lucrat cu ele
data_1 = np.array(data_1)
data_2 = np.array(data_2)

#nr de straturi ascunse
hidd_size = 8
#nr de outputuri
out_size = len(tags)
#rata de invatare
learning_rate = 0.001
#nr de input-uri
in_size = len(data_1[0])
#numarul de epoci
epochs = 1000
batch_size = 8


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(data_1)
        self.x_data = data_1
        self.y_data = data_2

    # returneaza dataset[i]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # len(data_1)
    def __len__(self):
        return self.n_samples


dataset = ChatDataset() # apelam clasa
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

# verificam daca utilizam pe acest dispozitiv cpu sau gpu(cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#daca il putem utiliza atunci dam push la acest model device-ului
model = NeuralNet(in_size, hidd_size, out_size).to(device)

# pierderile si optimizarile
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# antrenam modelul
for epoch in range(epochs):
    for (words, labels) in train_loader:
        # aplicam un "push in device"
        # modelul este incarcat pe dispozitivul specificat

        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward pass
        outputs = model(words)

        # calculam pierderile din aceasta epoca
        loss = criterion(outputs, labels)

        # backward si optimizari
        # se reduc gradientii pentru ca torch-ul le acumuleaza
        optimizer.zero_grad()

        # calculeaza backpropagation
        loss.backward()
        optimizer.step()


# introducem informatiile
data = {"model_state": model.state_dict(),"input_size": in_size, "hidden_size": hidd_size, "output_size": out_size, "all_words": patt, "tags": tags}

FILE = "data.pth"

# salvam informatiile
torch.save(data, FILE)

print(f'File saved to {FILE}')