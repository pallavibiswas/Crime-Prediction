import pandas as pd
import numpy as np

csv_files = {
    "residencehall_arrest": "data/residencehall_arrest.csv",
    "residencehall_crime": "data/residencehall_crime.csv",
    "residencehall_discipline": "data/residencehall_discipline.csv",
    "residencehall_hate": "data/residencehall_hate.csv",
    "residencehall_vawa": "data/residencehall_vawa.csv",
    "residencehall_result": "data/residencehall_result.csv"
}

irrelevant_columns = ["OPEID","address","ZIP","sector_cd","Sector_desc"]

def preprocess(file, irrelvant):

    df = pd.read_csv(file)

    df.dropna(inplace=True)
    df.drop(columns=irrelvant, inplace=True, errors='ignore')

    df.reset_index(drop=True, inplace=True)

    return df

processed_data = {}
for key, file in csv_files.items():
    processed_df = preprocess(file, irrelevant_columns)

    processed_file_name = f'cleaned_data/processed_{key}.csv'
    processed_df.to_csv(processed_file_name, index=False)
    
    print(f"Processed data for {key} saved to {processed_file_name}")


print("Data preprocessing complete. Processed data and summary saved to CSV files.")