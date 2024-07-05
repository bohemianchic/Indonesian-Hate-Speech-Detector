import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

def resample_data(df):
    # Separate majority and minority classes
    df_majority = df[df.Label == 0]
    df_minority = df[df.Label == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,     # Sample with replacement
                                     n_samples=len(df_majority),    # To match majority class
                                     random_state=123) # Reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled

def split_data(df, test_size=0.25, random_state=42):
    X = df['Tweet']  
    y = df['Label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Load combined dataset from load_data.py
    from load_data import load_dataset1, load_dataset2, combine_datasets

    # Define paths relative to the 'data' directory
    dataset1_path = 'IDHSD_RIO_unbalanced_713_2017.txt'
    dataset2_path = 're_dataset.csv'

    # Load datasets
    dataset1 = load_dataset1(dataset1_path)
    dataset2 = load_dataset2(dataset2_path)

    # Combine datasets
    combined_dataset = combine_datasets(dataset1, dataset2)
    
    # Resample data to address class imbalance
    resampled_dataset = resample_data(combined_dataset)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(resampled_dataset)

    # Save the processed data
    train_data = pd.DataFrame({'Tweet': X_train, 'Label': y_train})
    test_data = pd.DataFrame({'Tweet': X_test, 'Label': y_test})
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
