import numpy as np
import tqdm
from extract import extract_to_df

SAMPLE_DURATION = 15 # seconds
SAMPLE_FREQUENCY = 360  # per https://www.physionet.org/lightwave/?db=mitdb/1.0.0
SAMPLE_SIZE = SAMPLE_DURATION * SAMPLE_FREQUENCY
AF_MIN_DURATION = 30 # seconds
AF_MIN_SIZE = AF_MIN_DURATION * SAMPLE_FREQUENCY


def extract_event(df_ann):
    # define an evant as a dict from sample to label
    events = []
    for i in range(len(df_ann)):
        sample = df_ann.iloc[i]['Sample']
        sample_next = df_ann.iloc[i+1]['Sample'] if i < len(df_ann)-1 else df_ann.shape[0]
        auxNote = df_ann.iloc[i]['AuxNote']
        if auxNote != '':
            events.append({'Sample': sample, 'SampleNext' : sample_next, 'AuxNote': auxNote})
    return events

def cut_by_event(df_wave, events):
    data = []
    n = len(events)
    for i in range(n):
        sample = events[i]['Sample']
        auxNote = events[i]['AuxNote']
        sample_next = events[i+1]['Sample'] if i < n-1 else df_wave.shape[0]
        df_wave1 = df_wave.iloc[sample:sample_next].copy()
        data.append({"wave": df_wave1, "label": auxNote})
    return data

def cut_by_sample(data, stride):
    df_wave = data['wave']
    label = data['label']
    total_size = df_wave.shape[0]
    end = total_size - SAMPLE_SIZE + 1
    data1 = [{"wave" : df_wave.iloc[i:i + SAMPLE_SIZE], "label" : label} for i in range(0, end, stride)]
    return data1

def remove_short_AF(events):
    """
    keep only the long ones.
    """
    events_keep = []
    n = len(events)
    i = 0
    while i < n:
        auxNote = events[i]['AuxNote']
        if auxNote in ['(AFIB', '(AFL']:
            # for AF, we check the length of AF (AFIB or AFL) events, only keep the long ones
            events_temp = [events[i]]
            j = i + 1
            while j < n and events[j]['AuxNote'] in ['(AFIB', '(AFL']:
                events_temp.append(events[j])
                j += 1
            if events_temp[-1]['SampleNext'] - events_temp[0]['Sample'] >= AF_MIN_SIZE:
                events_keep.extend(events_temp)
            i = j
        else:
            # for other events, we just keep them
            events_keep.append(events[i])
            i += 1
    return events_keep

def load_dataset(indices, stride):
    result = []
    for idx in tqdm.tqdm(indices):
        try:
            df_wave, df_ann = extract_to_df(idx)
            events = extract_event(df_ann)
            events = remove_short_AF(events)
            data_by_event = cut_by_event(df_wave, events)
            for data in data_by_event:
                data1 = cut_by_sample(data, stride)
                result.extend(data1)
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
    return result

if __name__ == "__main__":
    # Example usage
    record_indices = range(100, 110)  # Adjust this range as needed
    data = load_dataset(record_indices, stride=SAMPLE_SIZE // 2)
    print(f"Loaded {len(data)} samples")
    # Save the data to a file or use it for training/testing