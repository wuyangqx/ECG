import pandas as pd
import wfdb

def extract_record(record_path):
    return wfdb.rdrecord(record_path)

def extract_wave(record_path):
    record = wfdb.rdrecord(record_path)
    df = record.to_dataframe()
    df = df.reset_index(drop=False).rename(columns={'index': 'Time'})
    df['Time'] = df['Time'].dt.total_seconds()
    df['Sample'] = df.index
    return df

def extract_annotation(record_path):
    annotation = wfdb.rdann(record_path, 'atr')
    df = pd.DataFrame({
        'Sample': annotation.sample,
        # 'Symbol': annotation.symbol,
        # 'SubType': annotation.subtype,
        # 'Channel': annotation.channel,
        # 'Num': annotation.num,
        'AuxNote': annotation.aux_note})
    
    # replace '\x00' with '' and '(AFL ' with 'AFL'
    df['AuxNote'] = df['AuxNote'].str.replace(r'\x00', '', regex=True)
    df['AuxNote'] = df['AuxNote'].str.replace(' ', '', regex=True)
    df = df[df['AuxNote'] != '']

    fs = wfdb.rdheader(record_path).fs
    df['Time'] = df['Sample'] / fs
    return df

def extract_to_df(record_path):
    df_wave = extract_wave(record_path)
    df_annotation = extract_annotation(record_path)
    # df = pd.merge(df_wave, df_annotation, on ='Sample', how='left')
    # df['AuxNote'] = df['AuxNote'].ffill()
    # df = df.dropna(subset=['AuxNote'])
    return df_wave, df_annotation

if __name__ == "__main__":
    # Example usage
    record_path = "c:/Users/wuyan/Projects/wfdb-python/mitdb/100"
    df_wave, df_ann = extract_to_df(record_path)
    print(df_wave.head(3))
    print(df_ann.head(5))