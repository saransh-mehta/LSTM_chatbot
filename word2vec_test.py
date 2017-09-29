# we will create word vectors from Game of thrones book
from __future__ import absolute_import, print_function, division
# we did the above to handle compatibility issues in py 2 and 3
import codecs #for encodings
import glob #regex
import multiprocessing #for running multiple threads to speed up
import os
import pprint  #pretty printing
import re
import nltk
import matplotlib.pyplot as plt
# below we will googlez word2vec from gensim
import gensim.models.word2vec as word2vec
#import seaborn as sb  #to plot nicely
#we will also remove stopwords using nltk and
#punctuations using string module
from nltk.corpus import stopwords
import string
from sklearn.manifold import TSNE
import pandas as pd


print('imported all dependencies.')
fileName = []
path = os.getcwd() + '/data'
for name in os.listdir(path):
    fileName.append(name)
print(fileName)

#we will one by one read all the above files and put data in one utf-8 string, u is used for that only
all_corpus = u""
for file in fileName:
    # we will use codec model to read file in utf-8 string
    print('reading file {}'.format(file))
    f = codecs.open(path + '/' + file, 'r', 'utf-8')
    all_corpus += f.read()
    print('file added ', file)
    print('corpus length = ',all_corpus.__len__())

'''''''''
print('going into tokenizing, stopword and punctuation removing and stuffs. May take time...')
preTokens = nltk.word_tokenize(all_corpus)
tokens = []
for word in preTokens:
    if (word not in stopwords.words('english')) and (word not in string.punctuation):
        tokens.append(word)
print('saving tokens to file')
w = open('tokens.txt', 'w')
w.write(tokens)
print('Done tokenize and above, now moving to building model')
#print(tokens)
'''
# we will also keep a list of all sentence if we want
# here what we have to do is not simply tokenize words, but sentence wise split the complete corpus
# ie, first make a list of sentences, and then make a list where tokenized each sentence will be a list
#for breaking text into sentences either use punkt tokenizer or loop method
'''''''''
sentences = []
for sentence in all_corpus.splitlines():
    sentences.append(sentence)
print(sentences)
'''

#nltk.download('punkt')
#download once if not available punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentencesList = tokenizer.tokenize(all_corpus)

#now we will individually tokenize each sentence and remove stopword, punctuations
'''''''''
for sentence in sentencesList:
    tokens = []
    preTokenize = nltk.word_tokenize(sentence)
    for tok in preTokenize:
        if tok not in stopwords.words('english') and tok not in string.punctuation:
            tok.append(tokens)
'''


#convert into a list of words
#rtemove unnnecessary,, split into words, no hyphens
#list of words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in sentencesList:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

# now we will build a word2vec model in built fro gensim

dimensions = 300
# here dienssions is the dimension of the vector that we want. The more the dimensions
#the more accurate the model is, more general
#but computational cost will increase

minWordCount = 3
#this is the minimum threshold that a word should cross to get registered

workers = multiprocessing.cpu_count()
# this is to use multiprocessing

contextSize = 7
# this is the length of the sentence that would b considered in one context

downsample = 1e-5
#for frequent word we use downsampling
print('building wor2vec from gensim')
gotVector = word2vec.Word2Vec(sg=1, seed=1, size = dimensions, workers=workers, min_count=minWordCount,
                              window=contextSize, sample=downsample)
#here we have just made the model, not feed data into it

# now we will build vocabulary to model and then print how many word got there in vocab
#all wont come due to mincount parameteres and all

gotVector.build_vocab(sentences)
print('Done with building vocab for model.')
print('vocab that got into length : ',gotVector.wv.vocab.__len__())
#print('vocab :', gotVector.wv.vocab)
# now we ll train the model with tokens
print('going into training. May take time...')

gotVector.train(sentences, total_examples= gotVector.corpus_count, epochs=gotVector.iter)

print('Successfully done training !')
print('Now moving to saving model in the name trained')

if not os._exists('trained'):
    os.makedirs('trained')

gotVector.save(os.path.join('trained', 'gotVector.w2v'))

print('Saved the trained model')

# using t-sne (t-distributed stochastic neighbor embedding) we will compress this 300 dimension vector/tensor
#into 2D to be plot
print('moving to compressing the {} dimensional vector to 2D for plot'.format(dimensions))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
#here we have just made the model, not feed data
#tsne is also a trained model, which compresses the vectors n generate co ordinates for them in 2D

allVectorMatrix = gotVector.syn0
# this is bring all the vectors to a vector matrix which can be fed to tsne

print('Training Tsne for compressing the vectors.May take time...')

allVectorMatrix_2d = tsne.fit_transform(allVectorMatrix)

print('Successfully compressed !')

#now we will go for plotting,
print('visualizing in dataframe')

df = pd.DataFrame(

    [ (word, coords[0], coords[1]) for word, coords in [(word, allVectorMatrix_2d[gotVector.vocab[word].index])
            for word in gotVector.vocab ]
    ],
    columns=["word", "x", "y"]
)

print(df.head(10))

print('now going for plotting using seaborn')

sb.set_context('poster')

plt.scatter(df['x'], df['y'], s=10)
print('the overall plot must be ready..,Thanks for patience ! Close the plot for next ')
plt.show()

def zoomInTo(xStart, xEnd, yStart, yEnd):

    axes = plt.axes()
    axes.set_xlim(xStart, xEnd)
    axes.set_ylim(yStart, yEnd)
    plt.scatter(df['x'], df['y'], s=35)
    plt.show()
    return
zoomInTo(xStart=4.0, xEnd=4.15, yStart= -0.5, yEnd= -0.1)


#axes = plt.axes()
#axes.
