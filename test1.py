import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.data import Field, TabularDataset, BucketIterator

tokenize = lambda x: list(x)

def build_data(train_file, test_file):
    TEXT = Field(sequential=True, tokenize = tokenize, lower=False)
    LABELS = Field(sequential=False, use_vocab=True)

    datafields = [('word', TEXT), ('label', LABELS), ('left', TEXT), ('right', TEXT)]
    train_data, valid_data = TabularDataset.splits(path='data', train='sample_train.txt',
                                                   validation='sample_train.txt', format='tsv',
                                                   skip_header=False, fields=datafields)
    TEXT.build_vocab(train_data)
    LABELS.build_vocab(train_data)

    train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                                   batch_size=(64, 64), device=-1,
                                                   sort_key=lambda x: len(x.word),
                                                   sort_within_batch=False, repeat=False)

    # print(next(train_iter.__iter__()))
    return train_iter, valid_iter


class NGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def train_ngram(vocab, embedding_dim, context_size, word_to_ix, trigrams):
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModel(len(vocab), embedding_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:
            context_idxs = torch.LongTensor([word_to_ix[w] for w in context])
            target_idxs  = torch.LongTensor([word_to_ix[target]])
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, target_idxs)

            # autograd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    # print(losses)
    return losses


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
        self.states = []

    def init_hidden(self):
        return (torch.randn(1, 1, self.hidden_dim),
                torch.randn(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        self.states.append((self.hidden, lstm_out))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_space, dim=1)


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def train_tagger(word_to_ix, tag_to_ix, training_data):
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for sent, tags in training_data:
            model.zero_grad()
            model.hidden = model.init_hidden()
            sent_in = prepare_sequence(sent, word_to_ix)
            target  = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sent_in)
            loss = loss_function(tag_scores, target)
            loss.backward()
            optimizer.step()
            print('epoch: %d score: %f' % (epoch, loss))
    return model


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers)

    def forward(self, inputs, labels):
        # embedding
        emb = nn.Embedding()
        logits, hidden = self.encoder(inputs, labels)
        return logits

