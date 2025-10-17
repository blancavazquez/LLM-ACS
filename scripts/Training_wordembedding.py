"""
Blanca Vázquez <blancavazquez2013@gmail.com>
IIMAS, UNAM
-------------------------------------------------------------------------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------------
Script for training word embeddings
"""

import time
from utils import *
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

database = "sinACS"
w2v_model = "SkipGram_bigram"
database_path = "../data/sinACS_notas/"
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
save_wordEmbedding = "../embeddings/"+ database+"_"+w2v_model
algorithm = 1 ## 1 for skip-gram; otherwise CBOW.
vector_size = 300

print("***Loading data")
xtrain, ytrain, label = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)
print("***Split each word for each clinical note")
xtrain = [row.split() for row in xtrain] #'captopril', 'tolerated', 'switch', 'amio',
xtest = [row.split() for row in xtest]

print("***Bigram")
from gensim.models.phrases import Phrases, Phraser
phrases = Phrases(xtrain, 
                  min_count=30,
                  progress_per=10000)
bigram = Phraser(phrases)
xtrain = bigram[xtrain]

print("***Pipeline")
start = time.time()
modelo = Word2Vec(
                vector_size=vector_size,
                alpha=0.03, 
                window=2,
                min_count=20,
                sample=6e-5,
                min_alpha=0.0007,
                negative=20,
                sg=algorithm,
                )
modelo.build_vocab(xtrain, progress_per=10000)
words_in_vocab = modelo.wv.key_to_index.keys()
modelo.train(xtrain, total_examples=modelo.corpus_count, epochs=10, report_delay=10)
elapse = (time.time() - start) / 60
print('Time in minutes: ', elapse)

print("***Saving wordEmbedding")
modelo.save(save_wordEmbedding)
word_vectors = modelo.wv
word_vectors.save(save_wordEmbedding)