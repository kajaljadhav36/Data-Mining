import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# Load and preprocess ADMISSIONS data
df_adm = pd.read_csv('ADMISSIONS_sorted.csv')
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format='%Y-%m-%d %H:%M:%S', errors='coerce')

df_adm = df_adm.sort_values(['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)
df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)
df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID')['ADMISSION_TYPE'].shift(-1)

# Remove 'ELECTIVE' next admission types
rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
df_adm.loc[rows, 'NEXT_ADMITTIME'] = pd.NaT
df_adm.loc[rows, 'NEXT_ADMISSION_TYPE'] = np.nan

df_adm[['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']] = df_adm.groupby('SUBJECT_ID')[
    ['NEXT_ADMITTIME', 'NEXT_ADMISSION_TYPE']
].fillna(method='bfill')
df_adm['DAYS_NEXT_ADMIT'] = (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds() / (24 * 60 * 60)
df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')

# Filter out newborns and deaths
df_adm = df_adm[df_adm['ADMISSION_TYPE'] != 'NEWBORN']
df_adm = df_adm[df_adm.DEATHTIME.isnull()]
df_adm['DURATION'] = (df_adm['DISCHTIME'] - df_adm['ADMITTIME']).dt.total_seconds() / (24 * 60 * 60)

# Load and preprocess NOTEEVENTS data
df_notes = pd.read_csv('NOTEEVENTS_sorted.csv')
df_notes = df_notes.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE'])
df_adm_notes = pd.merge(
    df_adm[
        [
            'SUBJECT_ID',
            'HADM_ID',
            'ADMITTIME',
            'DISCHTIME',
            'DAYS_NEXT_ADMIT',
            'NEXT_ADMITTIME',
            'ADMISSION_TYPE',
            'DEATHTIME',
            'OUTPUT_LABEL',
            'DURATION',
        ]
    ],
    df_notes[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'TEXT', 'CATEGORY']],
    on=['SUBJECT_ID', 'HADM_ID'],
    how='left',
)

df_adm_notes['ADMITTIME_C'] = pd.to_datetime(
    df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0]), format='%Y-%m-%d', errors='coerce'
)
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes['CHARTDATE'], format='%Y-%m-%d', errors='coerce')

# Extract discharge summaries
df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
df_discharge = df_discharge.groupby(['SUBJECT_ID', 'HADM_ID']).nth(-1).reset_index()
df_discharge = df_discharge[df_discharge['TEXT'].notnull()]

# Extract early notes (less than n days)
def less_n_days_data(df_adm_notes, n):
    df_less_n = df_adm_notes[
        ((df_adm_notes['CHARTDATE'] - df_adm_notes['ADMITTIME_C']).dt.total_seconds() / (24 * 60 * 60)) < n
    ]
    df_less_n = df_less_n[df_less_n['TEXT'].notnull()]
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(
        lambda x: df_less_n[df_less_n['HADM_ID'] == x].OUTPUT_LABEL.values[0]
    )
    return df_concat

df_less_2 = less_n_days_data(df_adm_notes, 2)
df_less_3 = less_n_days_data(df_adm_notes, 3)

# Preprocess text data
def preprocess1(x):
    y = re.sub(r'\[\*\*(.*?)\*\*\]', '', x)  # Remove de-identified brackets
    y = re.sub(r'[0-9]+\.', '', y)  # Remove 1.2. segments
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'--|__|==', '', y)
    return y

def preprocessing(df_less_n):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ').str.replace('\r', ' ').str.strip().str.lower()
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(preprocess1)

    chunks = []
    for i in tqdm(range(len(df_less_n))):
        x = df_less_n.TEXT.iloc[i].split()
        n = len(x) // 318
        for j in range(n):
            chunks.append({'TEXT': ' '.join(x[j * 318 : (j + 1) * 318]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i], 'ID': df_less_n.HADM_ID.iloc[i]})
        if len(x) % 318 > 10:
            chunks.append({'TEXT': ' '.join(x[-(len(x) % 318) :]), 'Label': df_less_n.OUTPUT_LABEL.iloc[i], 'ID': df_less_n.HADM_ID.iloc[i]})
    return pd.DataFrame(chunks)

df_discharge = preprocessing(df_discharge)
df_less_2 = preprocessing(df_less_2)
df_less_3 = preprocessing(df_less_3)

# Split train/test/validation
readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 1].HADM_ID
not_readmit_ID = df_adm[df_adm.OUTPUT_LABEL == 0].HADM_ID

not_readmit_ID_use = not_readmit_ID.sample(n=len(readmit_ID), random_state=1)
id_val_test_t = readmit_ID.sample(frac=0.2, random_state=1)
id_val_test_f = not_readmit_ID_use.sample(frac=0.2, random_state=1)

id_train_t = readmit_ID.drop(id_val_test_t.index)
id_train_f = not_readmit_ID_use.drop(id_val_test_f.index)

id_val_t = id_val_test_t.sample(frac=0.5, random_state=1)
id_test_t = id_val_test_t.drop(id_val_t.index)

id_val_f = id_val_test_f.sample(frac=0.5, random_state=1)
id_test_f = id_val_test_f.drop(id_val_f.index)

id_test = pd.concat([id_test_t, id_test_f])
id_val = pd.concat([id_val_t, id_val_f])
id_train = pd.concat([id_train_t, id_train_f])

# Final dataset preparation
discharge_train = df_discharge[df_discharge.ID.isin(id_train)]
discharge_val = df_discharge[df_discharge.ID.isin(id_val)]
discharge_test = df_discharge[df_discharge.ID.isin(id_test)]

discharge_train.to_csv('./discharge/train.csv', index=False)
discharge_val.to_csv('./discharge/val.csv', index=False)
discharge_test.to_csv('./discharge/test.csv')
