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
import torch
from utils import *

#------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\n Device: ", device)
database = "NSTEMI"
database_path = "../data/NSTEMI_notas/"
llm_model = "BioClinicalBert_pretrained"
model_type = "xgb"
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
save_metrics_training = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type +"_metrics_training"
save_metrics_testing = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_metrics_testing"
save_ypred_test = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_ypred"
save_ypred_proba_test = "../logs_ML/"+ database +"_"+ llm_model + "_"+ model_type + "_ypred_proba"
save_bert_model = "../best_models/"+llm_model
save_wordEmbedding = "../embeddings/"+ database +"_Embedding_"+ llm_model
saving_modelML = "../best_models/"+llm_model+"_"+model_type
save_shap_values = "../shap_values/"+llm_model+"_"+model_type

print("\n\n-------------------------------")
print("***Loading data")
xtrain, ytrain, labels = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)
pd_xreal = pd.concat([xtrain,xtest], axis = 0)
pd_yreal = pd.concat([ytrain,ytest], axis = 0)
xreal = pd_xreal.to_frame()
yreal = pd_yreal
xreal=xreal.reset_index(drop=True)
yreal=yreal.reset_index(drop=True)
print("\n\n---- Loading model ----")
modelo = opening_model(saving_modelML)

print("\n\n---- Testing set ----")
testing_xgb(xreal, yreal, modelo, labels, save_bert_model, save_ypred_test, save_ypred_proba_test, save_metrics_testing)
get_shap_values_LLM(xreal, yreal, modelo, labels, save_bert_model)

