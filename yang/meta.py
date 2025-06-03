class Dataset:
    """
    Meta class for data.
    """
    def __init__(self, parent_dir, data_name, 
                 sample_frequency, wave_col, record_exclude=None):
        self.parent_dir = parent_dir
        self.data_name = data_name
        self.data_path = f"{parent_dir}/{data_name}"
        self.sample_frequency = sample_frequency
        self.wave_col = wave_col
        self.record_exclude = record_exclude
    
    def __str__(self):
        return f"data(parent_dir={self.parent_dir}, data_name={self.data_name}, sample_frequency={self.sample_frequency})"
    
if __name__ == "__main__":
    # Example usage
    data = Dataset("c:/Users/wuyan/Projects/wfdb-python/", "mitdb", 
                   360, "MLII")
    
    print(data.sample_frequency)
    

