import pandas as pd
import csv
import os
import logging

# Setup logging
logging.basicConfig(filename='data_loading.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def load_dataset1(file_path):
    full_path = os.path.join('data', file_path)
    data = []
    with open(full_path, mode='r', encoding='latin1', errors='replace') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            try:
                label = 1 if row[0] == 'HS' else 0
                tweet = row[1]
                data.append({'Label': label, 'Tweet': tweet})
            except IndexError:
                continue
    return pd.DataFrame(data)

def load_dataset2(file_path):
    full_path = os.path.join('data', file_path)
    data = []
    with open(full_path, mode='r', encoding='latin1', errors='replace') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip the header row
        for row in reader:
            try:
                tweet = row[0]
                label = int(row[1])
                data.append({'Tweet': tweet, 'Label': label})
            except (IndexError, ValueError):
                continue
    return pd.DataFrame(data)

def combine_datasets(dataset1, dataset2):
    return pd.concat([dataset1, dataset2], ignore_index=True)

if __name__ == "__main__":
    # Define paths relative to the 'data' directory
    dataset1_path = 'IDHSD_RIO_unbalanced_713_2017.txt'
    dataset2_path = 're_dataset.csv'
    
    # Load datasets
    dataset1 = load_dataset1(dataset1_path)
    dataset2 = load_dataset2(dataset2_path)
    
    # Combine datasets
    combined_dataset = combine_datasets(dataset1, dataset2)
    
    # Log basic information about the combined dataset
    logging.info("Combined dataset head:\n%s", combined_dataset.head())
    logging.info("Combined dataset label distribution:\n%s", combined_dataset['Label'].value_counts())
