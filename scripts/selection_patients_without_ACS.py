import pandas as pd
import numpy as np


print("** Loading STEMI")
data_stemi = pd.read_csv("../data/data_STEMI/PATIENTS.csv")
stemi_patients = data_stemi['SUBJECT_ID'].unique()
print("stemi_patients: ", len(stemi_patients))

print("** Loading NSTEMI")
data_nstemi = pd.read_csv("../data/data_NSTEMI/PATIENTS.csv")
nstemi_patients = data_nstemi['SUBJECT_ID'].unique()
print("nstemi_patients: ", len(nstemi_patients))

print("** Concat unique patients")
conACS_patients = np.concatenate((stemi_patients, nstemi_patients))
print("conACS_patients: ", len(conACS_patients))
conACS_patients = list(set(conACS_patients))
print("conACS_patients: ", len(conACS_patients))

print("** Loading MIMIC III patients")
data_mimic = pd.read_csv("/home/blanca/Documents/Databases/DB_MimicIII/PATIENTS.csv")
mimic_patients = data_mimic['SUBJECT_ID'].unique()
print("mimic_patients: ", len(mimic_patients))

print("** Select patients from MIMIC III sin ACS")
patients_sinACS = np.setdiff1d(mimic_patients, conACS_patients)
print("patients_sinACS: ", len(patients_sinACS))

print("*** Saving data ***")
patients_sinACS = pd.DataFrame(patients_sinACS, columns = ['SUBJECT_ID'])
patients_sinACS.to_csv("../data/data_sinACS/unique_SUBJECT_ID.csv")