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
Script for preprocessing clinical notes
"""
import os
import nltk
import pandas as pd
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print("\n****** Loading data ******")
path = "../data/NSTEMI_notas/"
xtrain_path = path + "train/"
path_saving_xtrain = path + "xtrain.csv"

xtest_path = path + "test/"
path_saving_xtest = path + "xtest.csv"

"""Extending to stopwords with the most common words found in the clinical notes"""
adding_most_common_words = ['reason','year','plan','report',
                             'patient','medical','noted',
                             'examination','diagnosis','given',
                             'status','admitting','history',
                             'cont','impression','continue',
                             'family','please','today','findings',
                            'comparison','note','indication',
                             'time','action','since','current',
                             'hours','total','started',
                             'obtained','commands','hour',
                             'data','made','comments','following',
                             'view','sent','ordered',
                             'identified','times','compared',
                             'clip','portable','received',
                             'nursing','chest']
stop_words.extend(adding_most_common_words)
"""------------------------------------------------------"""


def preprocessing(note):
    data = {'original_note': note}
    pd_preprocessing = pd.DataFrame(data, index=[0])
    pd_preprocessing['clean_note'] = pd_preprocessing['original_note'].str.replace("[^a-zA-Z#]", " ",regex=True)
    pd_preprocessing['clean_note'] = pd_preprocessing['clean_note'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    pd_preprocessing['clean_note'] = pd_preprocessing['clean_note'].apply(lambda x: x.lower())
    tokenized_doc = pd_preprocessing['clean_note'].apply(lambda x: x.split())
    tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
    detokenized_doc = []
    for i in range(len(pd_preprocessing)):
         t = ' '.join(tokenized_doc[i])
         detokenized_doc.append(t)
    pd_preprocessing['clean_note'] = detokenized_doc
    return pd_preprocessing['clean_note'].values[0]

def get_text(path_data):
    pd_results = pd.DataFrame(columns = ['SUBJECT_ID', 'HADM_ID','CLINICAL_NOTE','FILENAME'])
    print("\nIterate over files")
    for root, _, files in os.walk(path_data): 
        for filename in files: 
            filepath = os.path.join(root, filename)
            print("Reading filepath: ", filepath)
            subject_id = filename.split('_')[0]
            hadm_id = filename.split('_')[3]
            with open(filepath, "r", encoding="utf-8") as f:
                clinical_note = f.read()
                pd_clinical_note = preprocessing(clinical_note)
                notas = {'SUBJECT_ID': subject_id, 
                         'HADM_ID': hadm_id,
                         'CLINICAL_NOTE':pd_clinical_note,
                         'FILENAME':filename}
                pd_results.loc[len(pd_results)] = notas
    return pd_results

print("Extracting: SUBJECT_ID, CLINICAL_NOTE, FILENAME")
xtrain_clinicalnotes = get_text(xtrain_path)
xtrain_clinicalnotes.to_csv(path_saving_xtrain)
print("Extracting: SUBJECT_ID, CLINICAL_NOTE, FILENAME")
xtest_clinicalnotes = get_text(xtest_path)
xtest_clinicalnotes.to_csv(path_saving_xtest)

