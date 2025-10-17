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
Script for training machine learning models from trained word embeddings models.
"""

from utils_w2v import *
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')

database = "sinACS"
w2v_model = "Fast_text"
database_path = "../data/sinACS_notas/"
model_type = "xgb"
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
save_metrics_training = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type +"_metrics_training"
save_metrics_testing = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_metrics_testing"
save_ypred_test = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_ypred"
save_ypred_proba_test = "../logs_ML/"+ database +"_"+ w2v_model + "_"+ model_type + "_ypred_proba"
save_wordEmbedding = "../embeddings/"+ database +"_"+ w2v_model
saving_modelML = "../models_ML/"+w2v_model+"_"+model_type

print("\n\n-------------------------------")
print("***Loading data")
xtrain, ytrain, labels = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)

print("\n\n-------------------------------")
print("***Split each word for each clinical note")
xtrain = [row.split() for row in xtrain]
xtest = [row.split() for row in xtest]

print("\n\n-------------------------------")
print("***Loading embedding")
word2vec_model = KeyedVectors.load(save_wordEmbedding, mmap='r')

print("\n\n---- GridSearch ----")
if model_type == "xgb":
    print("XGB")
    modelo = grid_xgboost_w2v(xtrain, ytrain, labels, word2vec_model, save_metrics_training, saving_modelML)
if model_type == "svm":
    print("SVM")
    modelo = grid_svm_w2v(xtrain, ytrain, labels, word2vec_model, save_metrics_training, saving_modelML)
if model_type == "knn":
    print("KNN")
    modelo = grid_knn_w2v(xtrain, ytrain, labels, word2vec_model, save_metrics_training, saving_modelML)
if model_type == "lr":
    print("LR")
    modelo = grid_lr_w2v(xtrain, ytrain, labels, word2vec_model, save_metrics_training, saving_modelML)

print("\n\n---- Testing set ----")
testing_w2v(xtest, ytest, modelo, labels, word2vec_model, save_ypred_test, save_ypred_proba_test, save_metrics_testing)