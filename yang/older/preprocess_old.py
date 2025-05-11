import numpy as np
import tqdm


# preporocess the data
def process_label(auxNote):
    if auxNote == '(AFIB':
        return 1
    elif auxNote == '(AFL':
        return 2
    else:
        return 0
    
def process_wave(df_wave):
    df_wave['MLII'] = (df_wave['MLII'] - df_wave["MLII"].mean()) / df_wave["MLII"].std()
    return df_wave

def process_data(data):
    X = []
    y = []
    for i in tqdm.tqdm(range(len(data))):
        df_wave = process_wave(data[i]['wave'])
        label = process_label(data[i]['label'])
    
        X.append(df_wave['MLII'].values)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y


# add the bandpass filtering
from scipy.signal import butter, lfilter
def bandpass_filtering(data, lowcut=0.5, highcut=45, fs=360, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    for i in range(len(data)):
        df_wave = data[i]['wave']
        df_wave['MLII'] = lfilter(b, a, df_wave['MLII'])
    return data

# add resampling function, resample data to 200hz by scipy.signal.resample
from scipy.signal import resample
import pandas as pd
from load import SAMPLE_FREQUENCY
def resample_data(data, new_fs=200):
    for i in range(len(data)):
        df_wave = data[i]['wave']
        num_samples = int(len(df_wave) * new_fs / SAMPLE_FREQUENCY)
        df_wave1 = pd.DataFrame()
        df_wave1['MLII'] = resample(df_wave['MLII'], num_samples)
        df_wave1['Time'] = np.linspace(0, len(df_wave) / SAMPLE_FREQUENCY, num_samples)
        data[i]['wave'] = df_wave1
    return data


# # resample the training data such that each class has the same number of samples
# from sklearn.utils import resample

# X_train_0 = X_train[y_train == 0]
# X_train_1 = X_train[y_train == 1]
# X_train_2 = X_train[y_train == 2]

# X_train_0_upsampled = resample(X_train_0, replace=True, n_samples=len(X_train_0), random_state=SEED)
# X_train_1_upsampled = resample(X_train_1, replace=True, n_samples=len(X_train_0), random_state=SEED)
# X_train_2_upsampled = resample(X_train_2, replace=True, n_samples=len(X_train_0), random_state=SEED)

# X_train_upsampled = np.concatenate([X_train_0_upsampled, X_train_1_upsampled, X_train_2_upsampled])
# y_train_upsampled = np.concatenate([np.zeros(len(X_train_0_upsampled)), 
#                                      np.ones(len(X_train_1_upsampled)), 
#                                      2*np.ones(len(X_train_2_upsampled))])

# print(f"X_train_upsampled shape: {X_train_upsampled.shape}, y_train_upsampled shape: {y_train_upsampled.shape}")
# # check the distribution of y_train_upsampled
# print("y_train_upsampled value counts:\n", pd.Series(y_train_upsampled).value_counts())

# X_train, y_train = X_train_upsampled, y_train_upsampled

# convert y_train and y_test to one-hot encoding
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train, num_classes=3)
# y_test = to_categorical(y_test, num_classes=3)
# print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")