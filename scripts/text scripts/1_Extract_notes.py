"""
This code is based from the Aggarwal repository.
https://github.com/kaggarwal/ClinicalNotesICU

"""

from scipy import stats
import os
import re
import json
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE|HISTORY OF PRESENT ILLNESS|BRIEF HISTORY'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION|MEDICATIONS ON DISCHARGE|ADMISSION LABS|ADMISSION EKG|ADMISSION ECHO'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION|CAUSE OF DEATH'
    r'|TECHNIQUE|ON ADMISSION|ON DISCHARGE|IMPRESSION|INDICATIONS FOR CATHETERIZATION|OTHER HEMODYNAMIC DATA'
    r'):|FINAL REPORT|FINAL DIAGNOSIS|GENERAL COMMENTS|CLINICAL IMPLICATIONS|CARDIAC HISTORY',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends) #*** ends:  [6407]


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()


def getSentences(t):
    return list(preprocess_mimic(t))


#******************************************************************#
#******************************************************************#
#******************************************************************#

df = pd.read_csv('../Databases/DB_MimicIII/NOTEEVENTS.csv')
df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
df.STORETIME = pd.to_datetime(df.STORETIME)

df2 = df[df.SUBJECT_ID.notnull()]
df2 = df2[df2.HADM_ID.notnull()]
df2 = df2[df2.CHARTTIME.notnull()] #Original
df2 = df2[df2.TEXT.notnull()]

df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]
#df2.to_csv("df2_lista_noteevents.csv")

del df
print(df2.groupby('HADM_ID').count().describe())
'''
count  55926.000000  55926.000000  55926.000000
mean      28.957283     28.957283     28.957283
std       59.891679     59.891679     59.891679
min        1.000000      1.000000      1.000000
25%        5.000000      5.000000      5.000000
50%       11.000000     11.000000     11.000000
75%       27.000000     27.000000     27.000000
max     1214.000000   1214.000000   1214.000000
'''

dataset_path = '../mimic_multitask/mimic3-benchmarks/data/NSTEMI_root/test/' #train/
all_files = os.listdir(dataset_path)
all_folders = list(filter(lambda x: x.isdigit(), all_files))
output_folder = '../mimic3-benchmarks/data/NSTEMI_root/test_text_fixed/'

suceed = 0
failed = 0
number_note = 0
failed_exception = 0

all_folders = all_folders

sentence_lens = []
hadm_id2index = {}
conteo_notas = 0

for folder in all_folders:
    try:
        print("**********************************************")
        print("SubjectID: ", folder)
        patient_id = int(folder)
        noteEvents = df2[df2.SUBJECT_ID == patient_id]
        
        if noteEvents.shape[0] == 0:
            print("No notes for PATIENT_ID : {}".format(patient_id))
            failed += 1
            continue
        noteEvents.sort_values(by='CHARTTIME')
        stays_path = os.path.join(dataset_path, folder, 'stays.csv')
        stays_df = pd.read_csv(stays_path)
        hadm_ids = list(stays_df.HADM_ID.values)
        for conteo_hadmid, hadmid in enumerate(hadm_ids):
            hadm_id2index[str(hadmid)] = str(hadmid)
            noteEvents_hadmid = noteEvents[noteEvents.HADM_ID == hadmid]
            data_json = {}
            for index, df_nota_per_patient in noteEvents_hadmid.iterrows():
                data_json["{}".format(df_nota_per_patient['CHARTTIME'])] = getSentences(df_nota_per_patient['TEXT'])
                with open(os.path.join(output_folder, folder + '_episode' + str(conteo_hadmid+1)+ '_hadmid_' + str(hadmid)), 'w') as f:
                    json.dump(data_json, f)
                conteo_notas +=1
        suceed += 1
    except:
        import traceback
        traceback.print_exc()
        print("Failed with Exception FOR Patient ID: %s", folder)
        failed_exception += 1

print("Sucessfully Completed: %d/%d" % (suceed, len(all_folders)))
print("No Notes for Patients: %d/%d" % (failed, len(all_folders)))
print("Failed with Exception: %d/%d" % (failed_exception, len(all_folders)))
print("conteo_notas: ", conteo_notas)
with open(os.path.join(output_folder, 'train_hadm_id2index'), 'w') as f:
    json.dump(hadm_id2index, f)
