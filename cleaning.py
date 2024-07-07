import pandas as pd
import numpy as np

csv_files = {
    "noncampus_arrest": "data/noncampus_arrest.csv",
    "noncampus_crime": "data/noncampus_crime.csv",
    "noncampus_discipline": "data/noncampus_discipline.csv",
    "noncampus_vawa": "data/noncampus_vawa.csv",
    "noncampus_result": "data/noncampus_result.csv"
}

irrelevant_columns = ["OPEID","Address","ZIP","City","sector_cd","Sector_desc","men_total","women_total"]

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