import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import seaborn as sns

sns.set()
import sys
import tensorflow as tf

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def physionet_data_loader(normalize='none', dataset='set-a'):
    feature_map = {'Albumin': 'Serum Albumin (g/dL)',
                   'ALP': 'Alkaline phosphatase (IU/L)',
                   'ALT': 'Alanine transaminase (IU/L)',
                   'AST': 'Aspartate transaminase (IU/L)',
                   'Bilirubin': 'Bilirubin (mg/dL)',
                   'BUN': 'Blood urea nitrogen (mg/dL)',
                   'Cholesterol': 'Cholesterol (mg/dL)',
                   'Creatinine': 'Serum creatinine (mg/dL)',
                   'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
                   'FiO2': 'Fractional inspired O2 (0-1)',
                   'GCS': 'Glasgow Coma Score (3-15)',
                   'Glucose': 'Serum glucose (mg/dL)',
                   'HCO3': 'Serum bicarbonate (mmol/L)',
                   'HCT': 'Hematocrit (%)',
                   'HR': 'Heart rate (bpm)',
                   'K': 'Serum potassium (mEq/L)',
                   'Lactate': 'Lactate (mmol/L)',
                   'Mg': 'Serum magnesium (mmol/L)',
                   'MAP': 'Invasive mean arterial blood pressure (mmHg)',
                   'Na': 'Serum sodium (mEq/L)',
                   'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
                   'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
                   'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
                   'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
                   'PaO2': 'Partial pressure of arterial O2 (mmHg)',
                   'pH': 'Arterial pH (0-14)',
                   'Platelets': 'Platelets (cells/nL)',
                   'RespRate': 'Respiration rate (bpm)',
                   'SaO2': 'O2 saturation in hemoglobin (%)',
                   'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
                   'Temp': 'Temperature (°C)',
                   'TroponinI': 'Troponin-I (μg/L)',
                   'TroponinT': 'Troponin-T (μg/L)',
                   'Urine': 'Urine output (mL)',
                   'WBC': 'White blood cell count (cells/nL)'
                   }
    feature_list = list(feature_map.keys())
    local_list = ['MechVent', 'Weight']
    data_dir = '/home/azencot_group/datasets/physionet'
    static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

    if os.path.exists(('/home/azencot_group/datasets/physionet/processed_df.csv')):
        df_full = pd.read_csv('/home/azencot_group/datasets/physionet/processed_df.csv')
        df_static = pd.read_csv('/home/azencot_group/datasets/physionet/processed_static_df.csv')
    else:
        txt_all = list()
        for f in os.listdir(os.path.join(data_dir, dataset)):
            with open(os.path.join(data_dir, dataset, f), 'r') as fp:
                txt = fp.readlines()
            # get recordid to add as a column
            recordid = txt[1].rstrip('\n').split(',')[-1]
            try:
                txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
                txt_all.extend(txt[1:])
            except:
                continue

        # convert to pandas dataframe
        df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])

        # extract static variables into a separate dataframe
        df_static = df.loc[df['time'] == '00:00', :].copy()

        df_static = df_static.loc[df['parameter'].isin(static_vars)]

        # remove these from original df
        idxDrop = df_static.index
        df = df.loc[~df.index.isin(idxDrop), :]

        # pivot on parameter so there is one column per parameter
        df_static = df_static.pivot(index='recordid', columns='parameter', values='value')

        # some conversions on columns for convenience
        df['value'] = pd.to_numeric(df['value'], errors='raise')
        df['time'] = df['time'].map(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

        df.head()
        # convert static into numeric
        for c in df_static.columns:
            df_static[c] = pd.to_numeric(df_static[c])

        df_full = pd.DataFrame(columns=['time', 'recordid'] + feature_list + local_list)

        df_full.to_csv('/home/azencot_group/datasets/physionet/processed_df.csv')
        df_static.to_csv('/home/azencot_group/datasets/physionet/processed_static_df.csv')

    selected_features = ['DiasABP', 'GCS', 'HCT', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp']

    # load in outcomes
    if dataset == 'set-a':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-a.txt'))
    elif dataset == 'set-b':
        y = pd.read_csv(os.path.join(data_dir, 'Outcomes-.txt'))
    label_list = ['SAPS-I', 'SOFA', 'In-hospital_death']

    df_sampled = df_full.groupby(['recordid'])
    max_len = 80
    signals, signal_maps, signal_lens = [], [], []
    z_ls, z_gs = [], []
    for i, sample in enumerate(df_sampled):
        id = sample[0]
        x = sample[1][selected_features]
        if np.array(x.isna()).mean() > 0.6 or len(x) < 0.5 * max_len:
            continue
        sample_map = x.isna().astype('float32')
        labels = y[y['RecordID'] == id][label_list]
        z_l = sample[1][['MechVent']]
        x = x.fillna(0.0)
        z_g = df_static[df_static['RecordID'] == id][['Age', 'Gender', 'Height', 'ICUType', 'Weight']]
        signals.append(np.array(x))
        signal_maps.append(np.array(sample_map))
        z_ls.append(np.array(z_l))
        z_gs.append(np.concatenate([np.array(z_g), np.array(labels)], axis=-1).reshape(-1, ))
        signal_lens.append(min(max_len, len(x)))
    signals = tf.keras.preprocessing.sequence.pad_sequences(signals, maxlen=max_len, padding='post', value=0.0,
                                                            dtype='float32')
    locals = tf.keras.preprocessing.sequence.pad_sequences(z_ls, maxlen=max_len, padding='post', value=0.0,
                                                           dtype='float32')
    maps = tf.keras.preprocessing.sequence.pad_sequences(signal_maps, maxlen=max_len, padding='post', value=1.0,
                                                         dtype='float32')
    z_gs = np.array(z_gs)
    signal_lens = np.array(signal_lens)

    test_inds = list(range(int(0.2 * len(signals))))
    inds = list(range(int(0.2 * len(signals)), len(signals)))
    random.shuffle(inds)
    train_inds = inds[:int(0.8 * len(inds))]
    valid_inds = inds[int(0.8 * len(inds)):]

    # plot a random sample
    ind = np.random.randint(0, len(train_inds))
    f, axs = plt.subplots(nrows=signals.shape[-1], ncols=1, figsize=(18, 14))
    for i, ax in enumerate(axs):
        ax.plot(signals[ind, :, i])
        ax.set_title(feature_list[i])
    plt.tight_layout()
    plt.savefig('/home/azencot_group/datasets/physionet/sample.pdf')

    train_signals, valid_signals, test_signals, normalization_specs = normalize_signals(signals, maps,
                                                                                        (train_inds, valid_inds,
                                                                                         test_inds),
                                                                                        normalize)
    trainset = tf.data.Dataset.from_tensor_slices((train_signals, maps[train_inds], signal_lens[train_inds],
                                                   locals[train_inds], z_gs[train_inds])).batch(30)
    validset = tf.data.Dataset.from_tensor_slices((valid_signals, maps[valid_inds], signal_lens[valid_inds],
                                                   locals[valid_inds], z_gs[valid_inds])).batch(10)
    testset = tf.data.Dataset.from_tensor_slices((test_signals, maps[test_inds], signal_lens[test_inds],
                                                  locals[test_inds], z_gs[test_inds])).batch(30)
    return trainset, validset, testset, normalization_specs