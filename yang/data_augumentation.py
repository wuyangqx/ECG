from load import SAMPLE_SIZE, cut_by_event, extract_event, cut_by_sample

# for index 222, 42 AFL appearance but we only get 13 samples from them, a lot of them have smaller sample size than 15s
idx = 222
df_wave, df_ann = extract_to_df(idx)
print(df_ann['AuxNote'].value_counts())
data1 = load_dataset(indices=[idx], stride=SAMPLE_SIZE)
labels = [d['label'] for d in data1]
print(pd.Series(labels).value_counts())

dfs_events = extract_event(df_ann)
data_by_event = cut_by_event(df_wave, dfs_events)
print(f"Number of AFL thrown away: {len([d['wave'].shape for d in data_by_event if d['label'] == '(AFL' and d['wave'].shape[0] < SAMPLE_SIZE])}")
[d['wave'].shape for d in data_by_event if d['label'] == '(AFL' and d['wave'].shape[0] < SAMPLE_SIZE]


def expand_by_dumplicate(data):
    df_wave = data['wave'].copy()
    label = data['label']
    current_size = df_wave.shape[0]
    number_of_duplicates = SAMPLE_SIZE // current_size + 1
    df_wave = pd.concat([df_wave] * number_of_duplicates, ignore_index=True)
    data1 = [{"wave" : df_wave.iloc[:SAMPLE_SIZE], "label" : label}]
    return data1

def load_dataset_extra(record_indices, label="(AFL"):
    result = []
    for idx in tqdm.tqdm(record_indices):
        try:
            df_wave, df_ann = extract_to_df(idx)
            events = extract_event(df_ann)

            data_by_event = cut_by_event(df_wave, events)
            data_too_short = [d['wave'] for d in data_by_event if d['label'] == label and d['wave'].shape[0] < SAMPLE_SIZE and d['wave'].shape[0] > SAMPLE_SIZE // 2]
            for data in data_too_short:
                data1 = expand_by_dumplicate(data, stride)
                result.extend(data1)
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
    return result


data_extra = load_dataset_extra(record_indices=range(100, 300), stride=SAMPLE_SIZE, events_filter=['(AFL'])
data_extra