# Steps
To perform all steps, you need to have access to the MIMIC-III database and download all files locally.

## Selection of STEMI patients
Run the script "1selection_subject_ids.sql" available in the [repository](https://github.com/blancavazquez/Riskmarkers_ACS/blob/master/sql_extraction/stemi/1selection_subject_ids.sql)

## Selection of NSTEMI patients
Run the script "1_selection_patNSTEMI.sql" available in the [repository](https://github.com/blancavazquez/Riskmarkers_ACS/blob/master/sql_extraction/nstemi/1_selection_patNSTEMI.sql)

## Selection of patients without ACS
Run the script:"selection_patients_without_ACS.py" 

* Then, you need create three separate folders with exclusive data of patients with STEMI, NSTEMI, and without ACS. Each folder should contain the same original structure of MIMIC-III. That is, each folder should included: admissions.csv, chartevents.csv, diagnoses_icd.csv, and so on.

* Clone the [repository](https://github.com/YerevaNN/mimic3-benchmarks) and run all scripts until the step 4 for each folder of STEMI, NSTEMI, and patients without ACS. The target is to obtain the train and test datasets for each population.

## Extraction of clinical notes
Run all files under "text scripts" for each folder of STEMI, NSTEMI, and without ACS patients.

## Training word embeddings models
Run the script "Training_wordembedding.py". In our study, we trained: SkipGram, CBOW, Glove, and FasText.

## Fine-tuning LLM
Run the script "Training_LLM.py". In our study, we trained: BERT, BioClinicalBert, BioBert, and BioGPT.

## Building multi-label classifiers using trained word embeddings models
Run the script "ML_wordembeddings.py". In our study, we trained: XGBoost, KNN, SVM, and LR.

## Building multi-label classifiers using pretrained LLM models
Run the script "ML_LLM.py". In our study, we trained: XGBoost, KNN, SVM, and LR.

# Testing on external validation datasets (Word embeddings)
Run the script "Testing_wordembeddings.py".

# Testing on external validation datasets (LLM)
Run the script "Testing_LLM.py".
