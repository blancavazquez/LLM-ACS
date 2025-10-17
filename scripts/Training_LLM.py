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
Script for fine-tunning LLMs
"""

import torch
from utils import *
import pandas as pd
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,IntervalStrategy,TrainingArguments, Trainer,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\n Device: ", device)

cv = 5
epochs = 100
batch_size = 8
database = "sinACS"
database_path = "../data/sinACS_notas/"
path_xtrain = database_path + "xtrain_labels_notes_cleaned.csv"
path_xtest = database_path +"xtest_labels_notes_cleaned.csv"
llm_model = "BioClinicalBert_pretrained"
model_type = "multitrainer"
model_ckpt = "microsoft/biogpt"
save_metrics_training = "../logs_LLM/"+ database +"_"+ llm_model + "_"+ model_type +"_bs_"+str(batch_size)+"_metrics_training"
save_metrics_testing = "../logs_LLM/"+ database +"_"+ llm_model + "_"+ model_type +"_bs_"+str(batch_size)+"_metrics_testing"
save_ypred_test = "../logs_LLM/"+ database +"_"+ llm_model + "_"+ model_type +"_bs_"+str(batch_size)+"_ypred"
save_ypred_proba_test = "../logs_LLM/"+ database +"_"+ llm_model + "_"+ model_type +"_bs_"+str(batch_size)+"_ypred_proba"
save_bert_model = "../models/"+llm_model+"_bs_"+str(batch_size)

def tokenize_and_encode(examples):
  return tokenizer(examples["CLINICAL_NOTE"], padding='max_length', truncation=True, max_length=164)

class MultilabelTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
          loss_fct = torch.nn.BCEWithLogitsLoss(weight=self.class_weights)
          loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                          labels.float().view(-1, self.model.config.num_labels))
        else:
          loss_fct = torch.nn.BCEWithLogitsLoss()
          loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                          labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

print("\n\n---- Loading data ----")
xtrain, ytrain, label = loading_data(path_xtrain)
xtest, ytest, _ = loading_data(path_xtest)
xtrain = xtrain.to_frame()
xtest = xtest.to_frame()
print("\n\n---- Loading LLM model ----")
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(label)).to('cuda')
print("\n\n---- Train test split ----")
xtrain=xtrain.reset_index(drop=True)
ytrain=ytrain.reset_index(drop=True)
xtest=xtest.reset_index(drop=True)
ytest=ytest.reset_index(drop=True)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
pd_yval = yval.reset_index(drop=True)
pd_ytest = ytest.reset_index(drop=True)

print("\n\n---- Concatenate CLINICAL_NOTE + LABELS")
xtrain = pd.concat([xtrain,ytrain], axis = 1).reset_index(drop=True)
xval = pd.concat([xval,yval], axis = 1).reset_index(drop=True)
xtest = pd.concat([xtest,ytest], axis = 1).reset_index(drop=True)

print("\n\n---- Counting per class")
class_mortalidad = xtrain['MORTALITY_INHOSPITAL'].value_counts()[1]
class_reinfarction= xtrain['Acute myocardial Reinfarction'].value_counts()[1]
class_heart_failure = xtrain['Heart failure'].value_counts()[1]
class_cardiogenic_shock = xtrain['Cardiogenic shock'].value_counts()[1]
class_mace = xtrain['MACE'].value_counts()[1]

print("\n\n---- Calculate inverse frequency weights")
num_samples_per_class = [class_mortalidad, class_reinfarction, class_heart_failure, class_cardiogenic_shock, class_mace]
total_samples = sum(num_samples_per_class)
class_weights = torch.tensor([1 - (count / total_samples) for count in num_samples_per_class], dtype=torch.float32)
class_weights = class_weights.to("cuda")

print("\n\n---- Datasets ----")
from datasets import Dataset
train_dataset = Dataset.from_pandas(xtrain)
xval_dataset = Dataset.from_pandas(xval)
xtest_dataset = Dataset.from_pandas(xtest)

print("\n\n---- Preprocess data ----")
cols = train_dataset.column_names
xtrain_dataset = train_dataset.map(lambda x : {"labels": [x[c] for c in cols if c != "CLINICAL_NOTE"]})
xval_dataset = xval_dataset.map(lambda x : {"labels": [x[c] for c in cols if c != "CLINICAL_NOTE"]})
xtest_dataset = xtest_dataset.map(lambda x : {"labels": [x[c] for c in cols if c != "CLINICAL_NOTE"]})
print(xtest_dataset)

print("\n\n---- Tokenize and encode ----")
cols = xtrain_dataset.column_names
cols.remove("labels")
xtrain_dataset_enc = xtrain_dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)
xval_dataset_enc = xval_dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)
xtest_dataset_enc = xtest_dataset.map(tokenize_and_encode, batched=True, remove_columns=cols)

print("\n\n---- TrainingArguments ----")
training_args = TrainingArguments(output_dir="../models/", # Directory for saving model checkpoints
                         learning_rate=5e-5, # Start with a small learning rate
                         per_device_train_batch_size=batch_size, ## Batch size per GPU
                         per_device_eval_batch_size=batch_size,  ## Batch size per GPU
                         num_train_epochs=epochs,
                         logging_dir="../events/",  # Directory for logs
                         logging_steps=1000,
                         warmup_steps=1500, 
                         weight_decay=0.001,  # Regularization
                         gradient_accumulation_steps = 4,
                         load_best_model_at_end=True,
                         eval_strategy=IntervalStrategy.EPOCH,
                         save_strategy=IntervalStrategy.EPOCH,
                         metric_for_best_model="eval_loss",
                         )
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
multi_trainer = MultilabelTrainer(model = model,  # Pre-trained BERT model
                                  args = training_args,
                                  train_dataset=xtrain_dataset_enc,
                                  eval_dataset=xval_dataset_enc,
                                  class_weights=class_weights,
                                  compute_metrics=compute_metrics, ## Custom metric
                                  data_collator = data_collator,
                                  tokenizer=tokenizer,
                                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                                  )
print("\n\n************Trainer*************************")
print(multi_trainer.train())

print("\n\n************Trainer evaluate****************")
print(multi_trainer.evaluate())

print("\n\n************Saving model*******************")
multi_trainer.save_model(save_bert_model)
tokenizer = AutoTokenizer.from_pretrained(save_bert_model)
print(tokenizer("hello world", truncation=True))

print("\n\n************Saving metrics during training**")
metrics_training = pd.DataFrame(multi_trainer.state.log_history)
metrics_training.to_csv(save_metrics_training+".csv")

print("\n\n************Compute metrics on test set****")
ypred_proba, ypred_per_sample, metrics = multi_trainer.predict(xtest_dataset_enc)
ypred_test = processing_ypred(ypred_proba, label, save_ypred_test)
ypred_proba = pd.DataFrame(ypred_proba, columns = label)
ypred_proba.to_csv(save_ypred_proba_test+".csv")
get_metrics_oneVSRest_bert(pd_ytest, ypred_test, ypred_proba,label,save_metrics_testing)
