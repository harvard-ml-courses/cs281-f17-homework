# Adapted from https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
import os, glob, zipfile
import torch
from torch.autograd import Variable
from torchtext import data, datasets, vocab

# Load imdb dataset
#### Arguments
# -batch_size: batch size. Default 1.
# -repeat: Whether to repeat the iterator for multiple epochs. If set to False, 
#  then .init_epoch() needs to be called before starting next epoch. Default False.
# -shuffle: Whether to shuffle examples between epochs.
# -imdb_path: The path to imdb zip file.
# -imdb_dir: The directory storing unzipped imdb files
# -reuse: If True and imdb_dir exists, imdb zip file will not be unzipped. Default True.
#
#### Returns
# -train_iter: An iterator for training examples. You can call "for batch in train_iter"
#  to get the training batches. Note that if repeat=False, then
#  train_iter.init_epoch() needs to be called before starting next epoch
# -val_iter: An iterator for validation examples.
# -test_iter: An iterator for test examples.
# -text_field: A field object, text_field.vocab is the vocabulary
#
#### Note:
# batch.label == 2 for positive examples, batch.label == 1 for negative examples
#
#### Example 1:
# train_iter, val_iter, test_iter, text_field = load_imdb(batch_size=100)
# V = len(text_field.vocab) # vocab size
# for epoch in range(num_epochs):
#     for batch in train_iter:
#         x = bag_of_words(batch, text_field)
#         y = batch.label - 1 # batch.label is 1/2, while we want 0/1.
#### Example 2 (word id to word str, word str to word id):
# word_id = 5
# word_str = text_field.vocab.itos[word_id]
# word_id = text_field.vocab.stoi[word_id]

def load_imdb(imdb_path='imdb.zip', imdb_dir='imdb', batch_size=1, gpu=False,
        reuse=False, repeat=False, shuffle=True):
    print "Loading Data"
    if (not reuse) or (not os.path.exists(imdb_dir)):
        f = zipfile.ZipFile(imdb_path, 'r')
        f.extractall('.')
        f.close()
    DEV = 0 if gpu else -1

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False)
    train = datasets.IMDB(os.path.join(imdb_dir, 'train'), text_field, label_field)
    val = datasets.IMDB(os.path.join(imdb_dir, 'val'), text_field, label_field)
    test = datasets.IMDB(os.path.join(imdb_dir, 'test'), text_field, label_field)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=DEV, repeat=repeat,
            shuffle=shuffle)

    text_field.build_vocab(train, vectors=vocab.GloVe(name='6B', dim=300))
    label_field.build_vocab(train)
    print (label_field.vocab.stoi['pos'])
    print (label_field.vocab.stoi['neg'])

    return train_iter, val_iter, test_iter, text_field

# Returns bag of words representation given a Batch object
#### Arguments
# -batch: A Batch object from the returned iterator from load_imdb.
# -text_field: text_field returned from load_imdb.
#### Returns
# -x: A Variable of size (batch_size, V) storing the word counts. x[b][word_id]
#  stores the number of occurrences of word_id in the b-th example in the batch
def bag_of_words(batch, text_field):
    V = len(text_field.vocab)
    x = torch.zeros(batch.text[0].size(0), V)
    ones = torch.ones(batch.text[0].size(1))
    for b in range(batch.text[0].size(0)):
        x[b].index_add_(0, batch.text[0].data[b], ones)
        x[b][text_field.vocab.stoi['<pad>']] = 0
    x = Variable(x, requires_grad=False)
    return x
