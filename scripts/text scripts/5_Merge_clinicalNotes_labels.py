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
Script for merging Labels + Clinical notes cleaned
"""
import pandas as pd

path = "../data/NSTEMI_notas/"
xtrain_notes = path + "xtrain.csv"
ytrain_labels = path + "labels_train.csv"
xtrain_labels_notes = path + "xtrain_labels_notes_cleaned.csv"

xtest_notes = path + "xtest.csv"
ytest_labels = path + "labels_test.csv"
xtest_labels_notes = path + "xtest_labels_notes_cleaned.csv"

def merge_data(path_labels, path_clinical_notes):
    labels = pd.read_csv(path_labels)
    clinical_notes = pd.read_csv(path_clinical_notes)
    print("labels: ", labels.shape, "clinical_notes: ", clinical_notes.shape)
    merge_clinical_notes = pd.merge(clinical_notes.reset_index(), labels.reset_index(), 
                            on=["SUBJECT_ID","HADM_ID"],
                            how="inner")
    print("merge_clinical_notes: ", merge_clinical_notes.shape)
    columns = ['SUBJECT_ID','HADM_ID','ICUSTAY_ID',
            'GENDER','AGE','LOS',
            'Acute and unspecified renal failure',
            'Acute myocardial infarction',
            "Acute myocardial Reinfarction",
            'Atrial fibrillation',
            'Cardiac arrest',
            'Cardiogenic shock',
            'Heart failure',
            'Left bundle branch block (LBBB)',
            "Stroke",
            'Right bundle branch block',
            'Ventricular fibrillation',
            'Ventricular tachycardia',
            'MORTALITY_INUNIT',
            'MORTALITY',
            'MORTALITY_INHOSPITAL',
            'CLINICAL_NOTE',
            'FILENAME']
    merge_clinical_notes = merge_clinical_notes[columns]
    print("\n\nNan values")
    count_nan = merge_clinical_notes['CLINICAL_NOTE'].isnull().sum()
    print('Number of NaN values present: ' + str(count_nan))
    print("merge_clinical_notes: ", merge_clinical_notes.shape)
    return merge_clinical_notes

print("****************************************************")
print("\n\nTraining")
xtrain_label_notes = merge_data(ytrain_labels, xtrain_notes)
xtrain_label_notes.to_csv(xtrain_labels_notes)
print("\n\nTesting")
xtest_label_notes = merge_data(ytest_labels, xtest_notes)
xtest_label_notes.to_csv(xtest_labels_notes)