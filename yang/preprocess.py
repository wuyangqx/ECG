from scipy.signal import resample, butter, lfilter
import pandas as pd
import numpy as np
import tqdm

RESAMPLE_FREQUENCY = 200  # Hz
BAND_PASS = (0.5, 45)  # Hz

def frequency_resampling(records, resample_fs=RESAMPLE_FREQUENCY):
    for record in tqdm.tqdm(records):
        df_wave = record['df_wave']
        num_samples = int(len(df_wave) * resample_fs / record['sample_frequency'])
        df_wave1 = pd.DataFrame()
        df_wave1['Time'] = np.linspace(0, len(df_wave) / record['sample_frequency'], num_samples)
        df_wave1['MLII'] = resample(df_wave['MLII'], num_samples)
        
        record['df_wave'] = df_wave1
        record['sample_frequency'] = resample_fs
    return records

def bandpass_filtering(records, lowcut=BAND_PASS[0], highcut=BAND_PASS[1], order=5):
    for record in tqdm.tqdm(records):
        nyquist = 0.5 * record['sample_frequency']
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        df_wave = record['df_wave']
        df_wave['MLII'] = lfilter(b, a, df_wave['MLII'])
    return records

def merge_annotation_to_wave(records):
    for record in tqdm.tqdm(records):
        df_wave = record['df_wave']
        df_ann = record['df_ann']
        df1 = pd.merge(df_wave, df_ann, how='outer', on='Time')
        df1['AuxNote'] = df1['AuxNote'].ffill()
        df2 = pd.merge(df_wave, df1[['Time', 'AuxNote']], how='left', on='Time')
        df2 = df2.dropna(subset=['AuxNote'])
        record['df_wave'] = df2

def cut_by_moving_window(records, window_duration, stride_duration):
    samples = []
    for record in tqdm.tqdm(records):
        sample_size = int(record['sample_frequency'] * window_duration)
        stride_size = int(record['sample_frequency'] * stride_duration)
        df_wave = record['df_wave']
        total_size = df_wave.shape[0]
        end = total_size - sample_size + 1
        for i in range(0, end, stride_size):
            df_sample = df_wave.iloc[i:i + sample_size].copy()
            samples.append({
                "df_wave" : df_sample,
                "sample_frequency" : record['sample_frequency'],
                "data_name" : record['data_name'],
                "record_name" : record['record_name']})
    return samples

def map_aux_note_to_label(auxNote):
    if auxNote == '(AFIB':
        return 1
    elif auxNote == '(AFL':
        return 2
    else:
        return 0
    

### functions to convert class to one-hot encoding and vice versa
def class_to_onehot(Y_class, num_classes):
    Y_onehot = np.zeros((Y_class.shape[0], Y_class.shape[1] * num_classes))
    for i in range(Y_class.shape[0]):
        for j in range(Y_class.shape[1]):
            Y_onehot[i, j * num_classes + Y_class[i, j]] = 1
    return Y_onehot

def onehot_to_class(Y_onehot, num_classes):
    Y_class = np.zeros((Y_onehot.shape[0], Y_onehot.shape[1] // num_classes))
    for i in range(Y_onehot.shape[0]):
        for j in range(Y_onehot.shape[1] // num_classes):
            Y_class[i, j] = np.argmax(Y_onehot[i, j * num_classes:(j + 1) * num_classes])
    return Y_class
# onehot = class_to_onehot(np.array([[0, 2, 1, 2, 1, 0]]), 3)
# clazz = onehot_to_class(onehot, 3)
# print("Onehot:", onehot)
# print("Clazz:", clazz)



def populate_label(samples):
    for sample in tqdm.tqdm(samples):
        df_wave = sample['df_wave'].copy()
        time_min = df_wave['Time'].min()
        df_wave['Time1'] = df_wave['Time'] - time_min # zero shift
        df_wave['Second'] = df_wave['Time1'].apply(lambda x: int(x))
        dfL = df_wave[['Second', 'AuxNote']].groupby('Second').agg("first").reset_index() # labels
        dfL['label'] = dfL['AuxNote'].apply(lambda x: map_aux_note_to_label(x))
        sample['df_label'] = dfL[['label']]



def preprocess_records(records, frequency_resample, bandpass_filter):
    #analysis.plot_wave(records, plot_idx, time_interval=[0,1], title='Original')
    if frequency_resample:
        records = frequency_resampling(records)
        #analysis.plot_wave(records, plot_idx, time_interval=[0,1], title='Frequency Resampling')
    if bandpass_filter:
        records = bandpass_filtering(records)
        #analysis.plot_wave(records, plot_idx, time_interval=[0,1], title='Bandpass Filtering')   
    return records

def generate_samples_from_records(records, window_duration, stride_duration):
    merge_annotation_to_wave(records)
    samples = cut_by_moving_window(records, window_duration=window_duration, stride_duration=stride_duration)
    populate_label(samples)
    print(f"Generated {len(samples)} samples from {len(records)} records.")
    return samples

def standardize_wave(df_wave):
    df_wave['MLII'] = (df_wave['MLII'] - df_wave["MLII"].mean()) / df_wave["MLII"].std()
    return df_wave

def generate_inputs_from_samples(samples):
    X = []
    y = []
    for sample in tqdm.tqdm(samples):
        df_wave = standardize_wave(sample['df_wave'])
        df_label = sample['df_label']
    
        X.append(df_wave['MLII'].values)
        y.append(df_label['label'].values)
    X = np.array(X)
    y = np.array(y)
    print(f"Generated {X.shape[0]} samples")
    print("X shape:", X.shape, "y shape:", y.shape)
    return X, y