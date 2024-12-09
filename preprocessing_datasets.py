# Importing essential libraries for data manipulation, visualization, and natural language processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# Load the admissions data from a CSV file
admissions_data= pd.read_csv('ADMISSIONS_sorted.csv')

# Convert admission, discharge, and death times to datetime format
admissions_data.ADMITTIME = pd.to_datetime(admissions_data.ADMITTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
admissions_data.DISCHTIME = pd.to_datetime(admissions_data.DISCHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
admissions_data.DEATHTIME = pd.to_datetime(admissions_data.DEATHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Sort the data by SUBJECT_ID and ADMITTIME for further processing
admissions_data= admissions_data.sort_values(['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)

# Generate next admission details for each patient
admissions_data['NEXT_ADMITTIME'] = admissions_data.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
admissions_data['NEXT_ADMISSION_TYPE'] = admissions_data.groupby('SUBJECT_ID')['ADMISSION_TYPE'].shift(-1)

# Filter out rows with 'ELECTIVE' next admission types
admissions_data.loc[admissions_data.NEXT_ADMISSION_TYPE == 'ELECTIVE', ['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']] = pd.NaT

# Forward fill the missing next admission details within each patient group
admissions_data[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']] = admissions_data.groupby('SUBJECT_ID')[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']].fillna(method='bfill')

# Calculate the number of days until the next admission for each record
admissions_data['DAYS_NEXT_ADMIT'] = (admissions_data.NEXT_ADMITTIME - admissions_data.DISCHTIME).dt.total_seconds() / (24 * 60 * 60)

# Label the data as 1 if the next admission is within 30 days, else 0
admissions_data['OUTPUT_LABEL'] = (admissions_data.DAYS_NEXT_ADMIT < 30).astype('int')

# Filter out records for newborn admissions and patients who are deceased
admissions_data= admissions_data[admissions_data['ADMISSION_TYPE'] != 'NEWBORN']
admissions_data= admissions_data[admissions_data.DEATHTIME.isnull()]

# Compute the duration of hospital stay in days
admissions_data['DURATION'] = (admissions_data['DISCHTIME'] - admissions_data['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60)


# Load and sort the notes data by SUBJECT_ID, HADM_ID, and CHARTDATE
patient_notes = pd.read_csv('NOTEEVENTS_sorted.csv')
patient_notes = patient_notes.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])

# Merge the admissions and notes data on SUBJECT_ID and HADM_ID
merged_adm_notes = pd.merge(
    admissions_data[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DAYS_NEXT_ADMIT', 'NEXT_ADMITTIME', 
            'ADMISSION_TYPE', 'DEATHTIME', 'OUTPUT_LABEL', 'DURATION']],
    patient_notes[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT', 'CATEGORY']],
    on=['SUBJECT_ID', 'HADM_ID'],
    how='left'
)

# Extract the date part from ADMITTIME for further processing
merged_adm_notes['ADMITTIME_C'] = pd.to_datetime(merged_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0]), format='%Y-%m-%d', errors='coerce')

# Convert CHARTDATE to datetime
merged_adm_notes['CHARTDATE'] = pd.to_datetime(merged_adm_notes['CHARTDATE'], format='%Y-%m-%d', errors='coerce')


# Extract the last discharge summary for each patient admission
discharge_summaries = merged_adm_notes[merged_adm_notes['CATEGORY'] == 'Discharge summary']
discharge_summaries = discharge_summaries.groupby(['SUBJECT_ID', 'HADM_ID']).nth(-1).reset_index()
discharge_summaries = discharge_summaries[discharge_summaries['TEXT'].notnull()]


# Function to filter early notes based on the number of days from admission
def less_n_days_data(merged_adm_notes, n):
    df_less_n = merged_adm_notes[
        ((merged_adm_notes['CHARTDATE'] - merged_adm_notes['ADMITTIME_C']).dt.total_seconds() / (24 * 60 * 60)) < n
    ]
    df_less_n = df_less_n[df_less_n['TEXT'].notnull()]
    
    # Combine the text of early notes for each admission
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(
        lambda x: df_less_n[df_less_n['HADM_ID'] == x].OUTPUT_LABEL.values[0]
    )
    return df_concat

# Extract notes for admissions within 2 and 3 days
early_notes_2_days = less_n_days_data(merged_adm_notes, 2)
early_notes_3_days = less_n_days_data(merged_adm_notes, 3)


# Function to clean and preprocess the text data
def preprocess1(x):
    # Remove unnecessary details such as de-identified brackets and specific medical terms
    y = re.sub(r'\[\*\*(.*?)\*\*\]', '', x)
    y = re.sub(r'[0-9]+\.', '', y)
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'--|__|==', '', y)
    return y

# Preprocess the text data by removing newline characters, stripping extra spaces, and converting to lowercase
def preprocessing(df_less_n):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ').str.replace('\r', ' ').str.strip().str.lower()
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(preprocess1)
    
    # Split the text into smaller chunks
    chunks = []
    for i in tqdm(range(len(df_less_n))):
        x = df_less_n.TEXT.iloc[i].split()
        n = len(x) // 318
        for j in range(n):
            chunks.append({'TEXT': ' '.join(x[j * 318 : (j + 1) * 318]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i], 'ID': df_less_n.HADM_ID.iloc[i]})
        if len(x) % 318 > 10:
            chunks.append({'TEXT': ' '.join(x[-(len(x) % 318) :]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i], 'ID': df_less_n.HADM_ID.iloc[i]})
    
    return pd.DataFrame(chunks)

# Apply preprocessing to discharge, less than 2 days, and less than 3 days data
discharge_summaries = preprocessing(discharge_summaries)
early_notes_2_days = preprocessing(early_notes_2_days)
early_notes_3_days = preprocessing(early_notes_3_days)


# Split data into readmitted (label 1) and not readmitted (label 0)
readmitted_patient_ids = admissions_data[admissions_data.OUTPUT_LABEL == 1].HADM_ID
not_readmitted_patient_ids = admissions_data[admissions_data.OUTPUT_LABEL == 0].HADM_ID

# Sample equal number of non-readmitted cases
not_readmitted_patient_ids_use = not_readmitted_patient_ids.sample(n=len(readmitted_patient_ids), random_state=1)

# Split data into training, validation, and test sets
id_val_test_t = readmitted_patient_ids.sample(frac=0.2, random_state=1)
id_val_test_f = not_readmitted_patient_ids_use.sample(frac=0.2, random_state=1)

id_train_t = readmitted_patient_ids.drop(id_val_test_t.index)
id_train_f = not_readmitted_patient_ids_use.drop(id_val_test_f.index)

id_val_t = id_val_test_t.sample(frac=0.5, random_state=1)
id_test_t = id_val_test_t.drop(id_val_t.index)

id_val_f = id_val_test_f.sample(frac=0.5, random_state=1)
id_test_f = id_val_test_f.drop(id_val_f.index)

id_test = pd.concat([id_test_t, id_test_f])
id_val = pd.concat([id_val_t, id_val_f])
id_train = pd.concat([id_train_t, id_train_f])

# Prepare the final datasets for training, validation, and testing
discharge_train = discharge_summaries[discharge_summaries.ID.isin(id_train)]
discharge_val = discharge_summaries[discharge_summaries.ID.isin(id_val)]
discharge_test = discharge_summaries[discharge_summaries.ID.isin(id_test)]

# Save the datasets as CSV files
discharge_train.to_csv('./discharge/train.csv', index=False)
discharge_val.to_csv('./discharge/val.csv', index=False)
discharge_test.to_csv('./discharge/test.csv', index=False)
