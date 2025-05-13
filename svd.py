import csv
import re
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import torch
def load_data(f):
    train = []
    with open(f, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        size = 10000
        for row in csv_reader:
            train.append(row)
            size -= 1
            if size == 0:
                break
    return train
def cleaner(doc):
    doc = doc.lower()
    doc = re.sub(r'&lt;.*&gt;', '', doc)
    doc = re.sub("[-;—:&%$()/*^\[\]\{\}]"," ",doc)
    doc = re.sub(r'\\', ' ', doc) 
    doc = re.sub("[#]","",doc) 
    doc = re.sub(r'[.][.][.]',"",doc) 
    doc = re.sub(r"[0-9]+|[0-9]+.[0-9]+","<NUM>",doc)
    doc = re.sub("[\"]"," \" ",doc)
    doc = re.sub("mr\.","mr",doc)
    doc = re.sub("mrs\.","mrs",doc)
    doc = re.sub("dr\.","dr",doc)
    doc = re.sub("a\.m\.","am",doc)
    doc = re.sub("p\.m\.","pm",doc)
    doc = re.sub("u\.s","us",doc)
    doc = re.sub("inc\.","inc",doc)
    doc = re.sub("corp\.","corp",doc)
    doc = re.sub(r"[.]"," . ", doc)
    doc = re.sub(r"[,]"," , ", doc)
    doc = re.sub(r"[?]"," ? ", doc)
    doc = re.sub(r"[!]"," ! ", doc)
    return doc
def Tokenizer(doc):
    words = doc.split(' ')
    real_words = []
    for word in words:
        if word:
            real_words.append(word)
    return real_words
def preprocess_data(data):
    for i in range(len(data)):
        data[i][1] = cleaner(data[i][1]) 
        data[i][1] = Tokenizer(data[i][1])
    sentences = []
    for i in range(len(data)):
        sentences.append(data[i][1])
    return sentences 
def create_cooccurrence_matrix(sentences, p, s):
    unique_words = {}
    for sentence in sentences:
        for word in sentence:
            if word not in unique_words:
                unique_words[word] = len(unique_words)
    unique_words["<unk>"] = len(unique_words)
    cooccur_mat = []
    for i in range(len(unique_words)):
        cooccur_mat.append([0] * len(unique_words))
    for sentence in sentences:
        for i in range(len(sentence)):
            start = max(0, i-p)
            end = min(len(sentence), i+s+1)
            for j in range(start, end):
                if i != j:
                    cooccur_mat[unique_words[sentence[i]]][unique_words[sentence[j]]] += 1
                    cooccur_mat[unique_words[sentence[j]]][unique_words[sentence[i]]] += 1
    return cooccur_mat, unique_words

train = load_data('train.csv')
test = load_data('test.csv')
train_sentences = preprocess_data(train)
test_sentences = preprocess_data(test) 
cooccur_mat, unique_words = create_cooccurrence_matrix(train_sentences, 2, 2)
sparse_mat = csc_matrix(cooccur_mat, dtype=float)
U, s, VT = svds(sparse_mat, k=200)
svd_word_embeddings = U.copy()
# print(len(train[0]), len(train))
# print(len(svd_word_embeddings[0]), len(svd_word_embeddings))
# print(len(unique_words))
# print(len(train_sentences))

tensor_word_vectors_svd = torch.tensor(svd_word_embeddings).float()
torch.save(tensor_word_vectors_svd, 'svd-word-vectors.pt')


