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
Script for extracting labels for train and test datasets
"""
import os
import pandas as pd

path = "../data/NSTEMI_root/"
all_labels_path = path+"labels.csv"

def extract_hadmIDS(path):
    all_subject_ids = os.listdir(path) #extract all filenames from path
    print("All subject_ids:", type(all_subject_ids), len(all_subject_ids))
    subject_ids = [line.split('_')[3] for line in all_subject_ids]
    subject_ids= list(map(int, subject_ids))
    print("subject_ids:", type(subject_ids), len(subject_ids))

    with open("subject_ids.txt", "w") as f:
        for item in subject_ids:
            f.write(f"{item}\n")
    return subject_ids

def emerge_with_labels(list_hadmids):
    pd_hadmids = pd.DataFrame({'HADM_ID': list_hadmids})
    pd_hadmids['HADM_ID'] = pd_hadmids['HADM_ID'].astype(str).astype(int)
    print("pd_hadmids:", pd_hadmids.shape)

    print("\n\nLoading all labels")
    all_labels = pd.read_csv(all_labels_path)
    print("all_labels:", all_labels.shape)
    labels_dataset = pd.merge(all_labels.reset_index(), pd_hadmids.reset_index(), 
                              on="HADM_ID", 
                              how="inner")
    #dropping duplicates
    labels_dataset.drop_duplicates(subset=['SUBJECT_ID','HADM_ID'], 
                                            keep='first', inplace=True, 
                                            ignore_index=True)
    print("labels_dataset: ", labels_dataset.shape)
    unique_count_SUBJECT_ID = labels_dataset['SUBJECT_ID'].nunique()
    unique_count_HADM_ID= labels_dataset['HADM_ID'].nunique()
    unique_count_ICUSTAY_ID = labels_dataset['ICUSTAY_ID'].nunique()
    print(f"Unique 'SUBJECT_ID': {unique_count_SUBJECT_ID}")
    print(f"Unique 'HADM_ID': {unique_count_HADM_ID}")
    print(f"Unique 'ICUSTAY_ID': {unique_count_ICUSTAY_ID}")

    return labels_dataset   

print("\n\nExtract labels for train dataset")
trainset_path = path + "train_text_fixed"
trainset_hadm_ids = extract_hadmIDS(trainset_path)
trainset_labels = emerge_with_labels(trainset_hadm_ids)
trainset_labels.to_csv(path + "labels_train.csv")

print("\n\nExtract labels for test dataset")
testset_path = path + "test_text_fixed"
testset_hadm_ids = extract_hadmIDS(testset_path)
testset_labels = emerge_with_labels(testset_hadm_ids)
testset_labels.to_csv(path + "labels_test.csv")