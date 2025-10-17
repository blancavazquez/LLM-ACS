# Steps
To perform all steps, you need to have access to the MIMIC-III database and download all files locally.

## Selection of STEMI patients
Run the script "1selection_subject_ids.sql" available in the [repository](https://github.com/blancavazquez/Riskmarkers_ACS/blob/master/sql_extraction/stemi/1selection_subject_ids.sql)

## Selection of NSTEMI patients
Run the script "1_selection_patNSTEMI.sql" available in the [repository]([https://github.com/blancavazquez/Riskmarkers_ACS/blob/master/sql_extraction/stemi/1selection_subject_ids.sql)

## Selection of patients without ACS
Run the script:"selection_patients_without_ACS.py" 


First, you should create three separate folders with exclusive data of patients with STEMI, NSTEMI, and without ACS. To do this, you can should used the ICD-9 code. 

Clone the [repositorio](https://github.com/YerevaNN/mimic3-benchmarks) and run all scripts until the step 4. The target is to obtain the train and test datasets. Previously identified the patients with STEMI, NSTEMI, without ACS using the 

# Text scripts
Run all files under "text scripts" folder.
