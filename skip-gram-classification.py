import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from skip_gram import train, train_sentences, test, test_sentences, unique_words
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [torch.tensor([word_to_idx["<unk>"] if word not in word_to_idx else word_to_idx[word] for word in text], dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [text for text in texts if len(text) > 0]
    lengths = [len(text) for text in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    
    lengths, sorted_idx = torch.tensor(lengths).sort(descending=True)
    texts_padded = texts_padded[sorted_idx]
    labels = labels[sorted_idx]
    
    return texts_padded, labels, lengths

labels_train = [int(train[i][0]) - 1 for i in range (len(train))]
labels_test = [int(test[i][0]) - 1 for i in range (len(test))]
# print(unique_words)
for word in unique_words:
    # print(type(unique_words[word]))
    unique_words[word] = unique_words[word][0] 
train_dataset = NewsDataset(train_sentences, labels_train, unique_words)
# test_dataset = train_dataset
test_dataset = NewsDataset(test_sentences, labels_test, unique_words)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
# for batch in test_loader:
#     print(batch)

class RNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.gru = nn.GRU(embedding_matrix.size(1), hidden_dim, batch_first=True) 
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.gru(packed_embedded)
        # packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = hidden[-1]
        out = self.fc(hidden)

        return out

def Train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0
    for texts, labels, lengths in iterator:
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)
    
def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    
  
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels, lengths in iterator:
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            total_loss += loss.item() 
            # print(predictions)
            preds = torch.argmax(predictions, axis=1) 
            # print(preds)
            

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics using scikit-learn
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1score = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    
  
    return total_loss / len(iterator), accuracy, precision, recall, f1score, conf_matrix


# print(type(skip_gram_word_embeddings), skip_gram_word_embeddings.shape)
embedding_matrix_skip_gram = torch.load('skip-gram-word-vectors.pt')
model = RNNClassifier(embedding_matrix_skip_gram, hidden_dim=24, output_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
criterion = nn.CrossEntropyLoss()
Epochs = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_loop(model, train_loader, test_loader, optimizer, loss_function, epochs, device):
    test_loss, test_accuracy, test_precision, test_recall, test_f1, conf_matrix = evaluate(model, test_loader, loss_function)
    print(f'Epoch {0}/{epochs}:')
    print(f'test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.4f}, test precision: {test_precision:.4f}, test Recall: {test_recall:.4f}, test F1: {test_f1:.4f}')

    for epoch in range(epochs):
        train_loss = Train(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy, test_precision, test_recall, test_f1, conf_matrix = evaluate(model, test_loader, criterion)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
        print(f'Confusion Matrix:') 
        print(f'{conf_matrix}')
train_loop(model, train_loader, test_loader, optimizer, criterion, Epochs, device)
torch.save(model.state_dict(), 'skip-gram-classification-model.pt')