import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torchtext.data import Field, TabularDataset, BucketIterator


class Classifier(nn.Module):
    '''
    Basic LSTM classifier with two way inputs
    '''
    def __init__(self, emb_dim, hidden_dim, vocab_size, tagset_size, batch_size=64):
        supoert(Classifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.left_encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.right_encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.left_hidden = self.init_hidden()
        self.right_hidden = self.init_hidden()
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)


    def init_hidden(self):
        # hidden = (h_0, c_0)
        # h_0: hidden state for each element, (num_layers * num_directions, batch, hidden_size)
        # c_0: cell state, same dimension;
        return (torch.randn(2 * 2, self.batch_size, self.hidden_dim),
                torch.randn(2 * 2, self.batch_size, self.hidden_dim))


    def forward(self, inputs):
        left_input, right_input = inputs
        # from (seq_len, batch_size) to (seq_len, batch_size, emb_dim)
        left_embeds = self.embedding(left_input)
        right_embeds = self.embedding(right_input)
        left_out, self.left_hidden = self.left_encoder(left_embeds, self.left_hidden)
        right_out, self.right_hdden = self.right_encoder(right_embeds, self.right_hidden)

        # concate output from each direction
        # (seq_len, batch_size, hidden_dim * directions) => (batch_size, hidden_dim * directions)
        all_out = torch.cat([left_out[-1], right_out[-1]], dim=1)
        tags = self.hidden2tag(all_out)
        return F.log_softmax(tags, dim=1)


# simplest tokenizer
tokenize = lambda x : list(x)

class DataReader():
    '''
    Load data from files
    '''

    def __init__(self):
        super().__init__()
        self.text_field = Field(sequential=True, tokenize = tokenize, lower=False)
        self.label_field = Field(sequential=False, use_vocab=True)
        self.datafields = [('word', self.text_field), ('label', self.label_field),
                           ('left': self.text_field), ('right', self.text_field)]

    def load(self, train_file, val_file):
        train_data, val_data = TabularDataset.splits(path="data", train=train_file,
                                                     validation=val_file, format="tsv",
                                                     skip_header=False,
                                                     fields=self.datafields)
        # build vocab on train data only
        self.text_field.build_vocab(train_data)
        self.label_field.build_vocab(train_data)
        return (train_data, val_data)


class Trainer():
    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, training_data):
        for epoch in self.epochs:
            # split into batches, returns a tuple
            train_iter = BucketIterator.splits((training_data,),
                                               batch_size=64, device=-1,
                                               sort_key=lambda x: len(x.word),
                                               repeat=False)[0]
            for batch in train_iter:
                self.model.zero_grad()
                preds = self.model((batch.left, batch.right))
                loss  = self.loss_function(preds, batch.label)
                loss.backward()
                self.optimizer.step()
                print('epoch: %d, score: %f' % (epoch, loss))


    def eval(self):
        pass

    def predict(self):
        pass


