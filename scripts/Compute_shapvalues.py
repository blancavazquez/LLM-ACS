"""
Scripts to compute shap values.
This code is based from the LLMs_in_perioperative_care repository.
https://github.com/cja5553/LLMs_in_perioperative_care/blob/main/codes/model%20eXplainability/SHAP_implementation.ipynb
"""
import torch
from utils import *
import pickle
import shap
import xgboost as xgb
import cupy as cp
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModel)

#------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\n Device: ", device)
database = "NSTEMI"
database_path = "../data/NSTEMI_notas/"
llm_model = "BioGPT_pretrained"
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
print("Concat: Train + Test")
pd_xreal = pd.concat([xtrain,xtest], axis = 0)
pd_yreal = pd.concat([ytrain,ytest], axis = 0)
#Convert from series to dataframe pandas
xreal = pd_xreal.to_frame()
yreal = pd_yreal
xreal=xreal.reset_index(drop=True)
yreal=yreal.reset_index(drop=True)

max_len= 256
tokenizer = AutoTokenizer.from_pretrained(save_bert_model, model_max_length=max_len)
model = AutoModel.from_pretrained(save_bert_model, output_hidden_states=True).to(device)

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        outputs = self.model(input)
        #embeddings=outputs.hidden_states[-1]
        embeddings = outputs.last_hidden_state
        pooled_embeddings = embeddings.mean(dim=1)
        return pooled_embeddings

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def batched_process(model, data, batch_size=64):
    dataset = TextDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    output = []
    for batch in dataloader:
        batch = [item for item in batch]
        with torch.no_grad():
            batch_out = model(
                torch.tensor([tokenizer.encode(v, padding='max_length', max_length=batch_size, truncation=True) for v in batch]).to('cuda'))
            batch_out_np = batch_out.cpu().numpy()  # Convert tensor to numpy array
            output.append(batch_out_np)
    return np.concatenate(output, axis=0)

def save_shap_values(filename, shap_values):
    # Create a tuple of the values, base_values, data, and feature names
    data_to_save = (shap_values.values, shap_values.base_values, shap_values.data, shap_values.feature_names)
    # Save the tuple using pickle
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

wrapped_model = WrapperModel(model)# Wrap the model
X_embed = batched_process(wrapped_model, list(xreal.CLINICAL_NOTE))# Processing data in batches
y = yreal.MORTALITY_INHOSPITAL.values#[:1]
modelo=xgb.XGBClassifier(device = "cuda",learning_rate=0.1, max_depth=4, base_score = 0.5)
modelo.fit(cp.array(X_embed),cp.array(y))

def fn(texts):
    embeds=batched_process(wrapped_model, texts)
    model_predict_proba = modelo.predict_proba(cp.array(embeds))
    return model_predict_proba

print("\n--------------")
masker = shap.maskers.Text(tokenizer, mask_token = "<mask>", collapse_mask_token=True)
explainer = shap.Explainer(fn, masker)
shap_values=explainer(list(xreal.CLINICAL_NOTE))
save_shap_values("shapvalues_nstemi.pickle", shap_values)#Save the shap_values