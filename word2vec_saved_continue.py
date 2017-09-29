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
import seaborn as sb  #to plot nicely
#we will also remove stopwords using nltk and
#punctuations using string module
from nltk.corpus import stopwords
import string
from sklearn.manifold import TSNE
import pandas as pd

print('loading the saved trained model')
#modelName = 'gotVector.w2v'

# NOTE : the direct methods word2vec.vocab and syn0 and other methods have been moved to
# KeyedVectored class.
gotVector = word2vec.Word2Vec.load(os.path.join('trained', 'gotVector.w2v'))

print('moving to compressing the {} dimensional vector to 2D for plot')

tsne = TSNE(n_components=2, random_state=0)
#here we have just made the model, not feed data
#tsne is also a trained model, which compresses the vectors n generate co ordinates for them in 2D

allVectorMatrix = gotVector.wv.syn0
# this is bring all the vectors to a vector matrix which can be fed to tsne
# the direct method syn0 has been moved to KeyedIndex class, hence
#we have to use wv

print('Training Tsne for compressing the vectors.May take time...')

allVectorMatrix_2d = tsne.fit_transform(allVectorMatrix)

print('Successfully compressed !')

#now we will go for plotting,
print('visualizing in dataframe')

df = pd.DataFrame(

    [ (word, coords[0], coords[1]) for word, coords in [(word, allVectorMatrix_2d[gotVector.wv.vocab[word].index])
            for word in gotVector.wv.vocab ]
    ],
    columns=["word", "x", "y"]
)

print(df.head(10))

print('now going for plotting using seaborn')

sb.set_context('poster')

#this is different from plt.scatter.
#this function is for plotting dataframe. Look documentation
df.plot.scatter('x', 'y', s=10, figsize = (20, 12))

#

print('the overall plot must be ready..,Thanks for patience ! Close the plot for next ')
plt.show()
'''''''''''
def zoomInTo(xStart, xEnd, yStart, yEnd):

    axes = plt.axes()
    axes.set_xlim(xStart, xEnd)
    axes.set_ylim(yStart, yEnd)
    plt.scatter(df['x'], df['y'], s=35)
    
    #, figsize=(20, 12)
    plt.show()
    return
zoomInTo(xStart=-50.0, xEnd=0, yStart= -50.0, yEnd= 0)

'''


def plot_region(xStart, xEnd, yStart, yEnd ):

    region  =df[ (df.x >= xStart) & (df.x <= xEnd) & (df.y >= yStart) & (df.y <= yEnd) ]

    design = region.plot.scatter("x", "y", s=35, figsize=(10, 8))

    # iterrows() is used to iterate over dataFrame rows, returns both index anf=d values
    for index, point in region.iterrows():

        design.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
    plt.show()

#x = input("enter x co - ordinates in tuple format")
#y = input("enter y co ordinate in tuple format")
plot_region(xStart= 4.0, xEnd = 4.2,yStart= -3.0, yEnd = -2.7)