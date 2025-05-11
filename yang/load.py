import numpy as np
import tqdm
from extract import extract_to_df
import os

import meta

def load_record_names(dataset):
    # Load record names from the dataset directory
    atr_files = [f for f in os.listdir(dataset.data_path) if f.endswith('.atr')]
    record_names = [os.path.splitext(f)[0] for f in atr_files]
    return [name for name in record_names if name not in dataset.record_exclude] if dataset.record_exclude else record_names


def load_dataset(dataset):
    records = []
    record_names = load_record_names(dataset)
    for record_name in tqdm.tqdm(record_names):
        try:
            record_path = os.path.join(dataset.data_path, record_name)
            df_wave, df_ann = extract_to_df(record_path)
            df_wave = df_wave.rename(columns={dataset.wave_col: 'MLII'})
            records.append({'df_wave': df_wave[['Time', 'MLII']],
                            'df_ann': df_ann[['Time', 'AuxNote']], 
                            'sample_frequency': dataset.sample_frequency,
                            'data_name': dataset.data_name,
                            'record_name': record_name})
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")
    print(f"Loaded {len(records)} records from {dataset.data_name} dataset.")
    return records

if __name__ == "__main__":
    # Example usage
    dataset = meta.datasets[0]
    records = load_dataset(dataset)
    