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

## Extract clinical notes
Run all files under "text scripts" for each folder of STEMI, NSTEMI, and without ACS patients.


