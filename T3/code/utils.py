# Adapted from https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
import os, glob, gzip, random
import torch
from torch.autograd import Variable
from torchtext import data, datasets, vocab
from scipy.stats import norm
import numpy as np
random.seed(1234)
from collections import Counter
# Load jester dataset, tested on Python 2.7
#### Arguments
# -load_text: Load text or not. In problem 2, text is unnecessary hence this flag should be
#             set to False to save memory. Default True.
# -batch_size: batch size. Default 1.
# -subsample_rate: Change this to 0.2 in problem 3 and use default 1.0 in problem 2. Default: 1.0
# -repeat: Whether to repeat the iterator for multiple epochs. If set to False, 
#          then .init_epoch() needs to be called before starting next epoch. Default False.
# -shuffle: Whether to shuffle examples between epochs.
# -ratings_path: The path to user, joke, rating file.
# -jokes_path: The path to jokes file 
# -max_vocab_size: Only the most max_vocab_size frequent words would be kept. We use 
#                  this to reduce memory footprint and the number of model parameters. Default: 150.
# -gpu: Use GPU or not. Default False.
#
#### Returns
# -train_iter: An iterator for training examples. You can call "for batch in train_iter"
#  to get the training batches. Note that if repeat=False, then
#  train_iter.init_epoch() needs to be called before starting next epoch
# -val_iter: An iterator for validation examples.
# -test_iter: An iterator for test examples.
# -text_field (when load_text is True): A field object, text_field.vocab is the vocabulary. 
#
#### Note:
# batch.ratings are ratings, can be 1, 2, 3, 4 or 5.
# batch.users are user ids, ranging from 1 to 63978.
# batch.jokes are joke ids, ranging from 1 to 150.
#
#### Example 1:
# train_iter, val_iter, test_iter, text_field = load_jester(batch_size=100, subsample_rate=1.0, load_text=True)
# V = len(text_field.vocab) # vocab size
# for epoch in range(num_epochs):
#     train_iter.init_epoch()
#     for batch in train_iter:
#         text = batch.text[0] # x is a tensor of size batch_size x max_len, where max_len
#                           # is the maximum joke length in the batch. The other jokes with
#                           # length < max_len are padded with text_field.vocab.stoi['<pad>']
#         ratings = batch.ratings-1 # batch.rating is a tensor containing actual ratings 1/2/3/4/5,
#                                # and we want that to be 0/1/2/3/4.
#         users = batch.users-1 
#         jokes = batch.jokes-1 
#### Example 2 (word id to word str, word str to word id):
# word_id = 5
# word_str = text_field.vocab.itos[word_id]
# word_id = text_field.vocab.stoi[word_id]

# Ignore this, irrelevant to homework
class Example(data.Example):
    @classmethod
    def fromlist(cls, data, fields):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                setattr(ex, name, field.preprocess(val))
        return ex

def load_jester(load_text=True, batch_size=1, subsample_rate=1.0, repeat=False, shuffle=True,
        ratings_path='jester_ratings.dat.gz', jokes_path='jester_items.clean.dat.gz', max_vocab_size=150, gpu=False):
    DEV = 0 if gpu else -1
    assert os.path.exists(jokes_path), "jokes file %s does not exist!"%jokes_path
    assert os.path.exists(ratings_path), "ratings file %s does not exist!"%ratings_path
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True)
    rating_field = data.Field(sequential=False, use_vocab=False)
    user_field = data.Field(sequential=False, use_vocab=False)
    joke_field = data.Field(sequential=False, use_vocab=False)
    if load_text:
        fields = [('text', text_field), ('ratings', rating_field), ('users', user_field), ('jokes', joke_field)]
    else:
        fields = [('ratings', rating_field), ('users', user_field), ('jokes', joke_field)]
    jokes_text = {}
    joke = -1
    all_tokens = []
    with gzip.open(jokes_path) as f:
        for i, line in enumerate(f):
            l = line.decode('utf-8')
            if len(l.strip()) == 0:
                continue
            if l.strip()[-1] == ':':
                joke = int(l.strip().strip(':'))
            else:
                joke_text = l.strip()
                tokens = l.strip().split()
                all_tokens.extend(tokens)
                jokes_text[joke] = joke_text
    counts = Counter(all_tokens)
    most_common = counts.most_common(max_vocab_size)
    most_common = set([item[0] for item in most_common])


    print ('Loading Data, this might take several minutes')
    if subsample_rate < 1.0:
        print ('Subsampling rate set to %f'%subsample_rate)

    train, val, test = [], [], []
    with gzip.open(ratings_path) as f:
        for i, l in enumerate(f):
            if i % 100000 == 0:
                print ('%d lines read'%i)
            user, joke, rating = l.split()
            user = int(user)
            joke = int(joke)
            rating = int(rating)
            if load_text:
                assert joke in jokes_text
                example = Example.fromlist([' '.join([item for item in jokes_text[joke].split() if item in most_common]), rating, user, joke], fields)
            else:
                example = Example.fromlist([rating, user, joke], fields)
            p = random.random()
            q = random.random()
            if p < 0.98:
                if q < subsample_rate:
                    train.append(example)
            elif p < 0.99:
                val.append(example)
            elif p < 1.0:
                test.append(example)
        train = data.Dataset(train, fields)
        val = data.Dataset(val, fields)
        test = data.Dataset(test, fields)
        train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), 
            batch_size=batch_size, device=DEV, repeat=repeat,
            shuffle=shuffle)
        train_iter.sort_key = lambda p: len(p.text) if hasattr(p, 'text') else 0
        val_iter.sort_key = lambda p: len(p.text) if hasattr(p, 'text') else 0
        test_iter.sort_key = lambda p: len(p.text) if hasattr(p, 'text') else 0

    print ('Data Loaded')

    if load_text:
        text_field.build_vocab(train)
        return train_iter, val_iter, test_iter, text_field
    else:
        return train_iter, val_iter, test_iter,


# PyTorch function for calcuating log \phi(x)
# example usage: normlogcdf1 = NormLogCDF()((h-b_r)/sigma)
class NormLogCDF(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        input_numpy = input.numpy()
        output = torch.Tensor(norm.logcdf(input_numpy))
        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        input_numpy = input.numpy()
        grad_input = grad_output.clone()
        grad_input = grad_input * torch.Tensor(np.exp(norm.logpdf(input_numpy) - norm.logcdf(input_numpy)))
        # clip infinities to 1000
        grad_input[grad_input==float('inf')] = 1000
        # clip -infinities to -1000
        grad_input[grad_input==float('-inf')] = -1000
        # set nans to 0
        grad_input[grad_input!=grad_input] = 0
        return grad_input

# PyTorch function for calculating log (\phi(x) - \phi(y)) where \phi is the normal distribution cdf
#### Arguments
# -x: a PyTorch Variable of size (batch_size).
# -y: a PyTorch Variable of size (batch_size). x[i] should be always greater than y[i].
#### Returns
# log (phi (x) - \phi(y))
def log_difference(x, y):
    # calculate by using p1 and p2
    logp1 = NormLogCDF()(x)
    logp2 = NormLogCDF()(y)
    logp = logp1 + torch.log(1 - torch.exp(logp2-logp1))
    # calculate by using 1-p1 and 1-p2
    log1_p1 = NormLogCDF()(-x)
    log1_p2 = NormLogCDF()(-y)
    logp_ = log1_p2 + torch.log(1 - torch.exp(log1_p1-log1_p2))
    return torch.max(logp, logp_)

