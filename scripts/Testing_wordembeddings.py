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
Script for testing model in STEMI and NSTEMI patients
"""
from utils import *
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')

database = "STEMI"
database_path = "../data/STEMI_notas/"
w2v_model = "CBOW_unigram"
model_type = "lr"
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
save_metrics_training = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type +"_metrics_training"
save_metrics_testing = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_metrics_testing"
save_ypred_test = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_ypred"
save_ypred_proba_test = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_ypred_proba"
save_wordEmbedding = "../best_models/"+ "sinACS" +"_"+ w2v_model
saving_modelML = "../best_models/"+w2v_model+"_"+model_type

print("\n\n-------------------------------")
print("***Loading data")
xtrain, ytrain, labels = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)
pd_xreal = pd.concat([xtrain,xtest], axis = 0)
pd_yreal = pd.concat([ytrain,ytest], axis = 0)
print("***Split each word for each clinical note")
xreal = [row.split() for row in pd_xreal]
print("\n\n---- Loading model ----")
modelo = opening_model(saving_modelML)
print("\n\n-------------------------------")
print("***Loading embedding")
word2vec_model = KeyedVectors.load(save_wordEmbedding, mmap='r') #SkipGram, CBOW, FastText
#word2vec_model = KeyedVectors.load_word2vec_format(save_wordEmbedding+".txt") #Glove

print("\n\n---- Testing set ----")
testing_w2v(xreal, pd_yreal, modelo, labels, word2vec_model, save_ypred_test, save_ypred_proba_test, save_metrics_testing)#SkipGram, CBOW, FastText
#testing_glove(xreal, pd_yreal, modelo, labels, word2vec_model, save_ypred_test, save_ypred_proba_test, save_metrics_testing)#Glove
get_shapvalues(pd_xreal, pd_yreal, modelo, labels, word2vec_model)