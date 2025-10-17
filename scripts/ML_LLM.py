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
Script for training machine learning models from pretrained LLM.
"""

import torch
from utils import *

#------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\n Device: ", device)
database = "sinACS"
database_path = "../data/sinACS_notas/"
llm_model = "BioClinicalBert_pretrained"
model_type = "xgb" #svm, #lr, #knn
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
save_metrics_training = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type +"_metrics_training"
save_metrics_testing = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_metrics_testing"
save_ypred_test = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_ypred"
save_ypred_proba_test = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_ypred_proba"
save_bert_model = "../models_LLM/"+llm_model
save_wordEmbedding = "../embeddings/"+ database +"_Embedding_"+ llm_model
saving_modelML = "../models_ML/"+llm_model+"_"+model_type

print("***********************************")
print("***Loading data")
xtrain, ytrain, labels = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)
xtrain = xtrain.to_frame()
xtest = xtest.to_frame()
xtrain=xtrain.reset_index(drop=True)
ytrain=ytrain.reset_index(drop=True)
xtest=xtest.reset_index(drop=True)
ytest=ytest.reset_index(drop=True)
print("\n\n---- GridSearch ----")
modelo = grid_xgboost(xtrain, ytrain, labels, save_bert_model, save_metrics_training, saving_modelML)
print("\n\n---- Testing set ----")
testing_xgb(xtest, ytest, modelo, labels, save_bert_model, save_ypred_test, save_ypred_proba_test, save_metrics_testing)