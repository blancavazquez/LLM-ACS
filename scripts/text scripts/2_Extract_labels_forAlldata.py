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
Script for getting labels from all dataset (MERGE)
Mortality, LOS, MICE events
"""
import pandas as pd

print("*** Loading data ***")
path = "../data/NSTEMI_root/"
phenotyping = pd.read_csv(path + "phenotypes_with_IcustayID.csv")
all_stays = pd.read_csv(path+"all_stays.csv")

print("phenotyping: ", phenotyping.shape)
print("all_stays: ", all_stays.shape)

print("phenotyping: ", phenotyping.columns.tolist())
print("all_stays: ", all_stays.columns.tolist())

print("*** Merge labels ***")
result = pd.merge(phenotyping, all_stays, on="ICUSTAY_ID")
result.to_csv(path+"all_columns.csv")
print("result: ", result.shape)

print("*** Selected of variables ***")
columns = [
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "GENDER",
    "AGE",
    "LOS",
    "Acute and unspecified renal failure",
    "Acute myocardial infarction",
	"Acute myocardial Reinfarction",
    "Atrial fibrillation",
    "Cardiac arrest",
    "Cardiogenic shock",
    "Heart failure",
    # "Cerebrovascular accident",
    "Left bundle branch block (LBBB)",
    "Stroke",
    "Right bundle branch block",
    "Ventricular fibrillation",
    "Ventricular tachycardia",
    "MORTALITY_INUNIT",
    "MORTALITY",
    "MORTALITY_INHOSPITAL"]


result = result[columns]
result.to_csv(path+"labels.csv")
print("result: ", result.shape)

unique_count_SUBJECT_ID = result['SUBJECT_ID'].nunique()
unique_count_HADM_ID= result['HADM_ID'].nunique()
unique_count_ICUSTAY_ID = result['ICUSTAY_ID'].nunique()
print(f"Unique 'SUBJECT_ID': {unique_count_SUBJECT_ID}")
print(f"Unique 'HADM_ID': {unique_count_HADM_ID}")
print(f"Unique 'ICUSTAY_ID': {unique_count_ICUSTAY_ID}")