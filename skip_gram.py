import csv
import re
import numpy as np
import random
import torch
def load_data(f):
    train = []
    with open(f, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        size = 15000
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
    freq = {}
    for sentence in sentences:
        for word in sentence:
            if word in freq:
                freq[word] += 1
            else:    
                freq[word] = 1
    rare_words = []
    for word in freq:
        if freq[word] < 2:
            rare_words.append(word)
    for i in range (len(sentences)):
        for j in range (len(sentences[i])):
            if sentences[i][j] in rare_words:
                sentences[i][j] = "<unk>"
    return sentences
def generate_positive_context(sentences, p, s):
    unique_words = {}
    for sentence in sentences:
        for word in sentence:
            if word not in unique_words:
                unique_words[word] = [len(unique_words), 1]
            else:
                unique_words[word][1] += 1
    s = 0
    cdf = []
    for word in unique_words:
        unique_words[word][1] = pow(unique_words[word][1], 0.75)
        s += unique_words[word][1]
    curr_s = 0
    for word in unique_words:
        curr_s += unique_words[word][1]/s 
        cdf.append(curr_s)
    positive_context = {}
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i] not in positive_context:
                positive_context[unique_words[sentence[i]][0]] = []
            start = max(0, i-p)
            end = min(len(sentence), i+s+1)
            for j in range(start, end):
                if i != j:
                    if unique_words[sentence[j]][0] not in positive_context[unique_words[sentence[i]][0]]:
                        positive_context[unique_words[sentence[i]][0]].append(unique_words[sentence[j]][0])
    return unique_words, positive_context, cdf

def generate_negative_samples(positive_context, cdf, num_neg_samples):
    negative_samples = {}
    for i_d in positive_context:
        negative_samples[i_d] = []
        c = num_neg_samples
        while c > 0:
            r = random.random()
            idx = np.searchsorted(cdf, r)
            if idx not in negative_samples[i_d] and idx not in positive_context[i_d] and idx != i_d:
                negative_samples[i_d].append(idx)
                c -= 1
    return negative_samples
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))
def generate_word_embeddings(positive_context, negative_samples, vocab_size, embedding_dim, lr):
    W = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
    C = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
    # print(W.shape, C.shape)
    for i in range(vocab_size):
        for pos_id in positive_context[i]:
            error = sigmoid(np.dot(C[pos_id], W[i])) - 1
            C[pos_id] -= lr * error * W[i]
            W[i] -= lr * error * C[pos_id]

        for neg_id in negative_samples[i]:
            error = sigmoid(np.dot(C[neg_id], W[i]))
            C[neg_id] -= lr * error * W[i]
            W[i] -= lr * error * C[neg_id]
    return W + C
train = load_data('train.csv')
test = load_data('test.csv')
train_sentences = preprocess_data(train)
test_sentences = preprocess_data(test)
unique_words, positive_context, cdf = generate_positive_context(train_sentences, 2, 2)
negative_samples = generate_negative_samples(positive_context, cdf, 5)
skip_gram_word_embeddings = generate_word_embeddings(positive_context, negative_samples, len(unique_words) , 200, 0.025)
skip_gram_word_embeddings = skip_gram_word_embeddings.copy()
tensor_word_vectors_skipgram = torch.tensor(skip_gram_word_embeddings).float()
torch.save(tensor_word_vectors_skipgram, 'skip-gram-word-vectors.pt')
# print(skip_gram_word_embeddings)
