import pandas as pd
import os
import concurrent.futures

def process_file(remote_path, local_path, dates, instruments):
    try:
        remote_path = remote_path.strip()
        local_path = local_path.strip()
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        df = pd.read_csv(remote_path, index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        
        df = df.reindex(index=dates, columns=instruments)
        
        df.to_csv(local_path, header=False, index=False, compression='gzip')
        print(f"Successfully processed {remote_path} -> {local_path}")
    except Exception as e:
        print(f"Error processing {remote_path}: {e}")

def main():
    fields = pd.read_csv('fields.csv', skipinitialspace=True)
    dates = pd.read_csv('universe_dates.csv', header=None).iloc[:, 0].astype(str).tolist()
    instruments = pd.read_csv('universe_instruments.csv', header=None).iloc[:, 0].astype(str).tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for _, row in fields.iterrows():
            remote_path = row['remote_path']
            local_path = row['local_path']
            futures.append(executor.submit(process_file, remote_path, local_path, dates, instruments))
            
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    main()