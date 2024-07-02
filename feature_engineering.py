import pandas as pd

output_dir = 'transformed_data'

csv_files = {
    "residencehall_arrest": "cleaned_data/processed_residencehall_arrest.csv",
    "residencehall_crime": "cleaned_data/processed_residencehall_crime.csv",
    "residencehall_discipline": "cleaned_data/processed_residencehall_discipline.csv",
    "residencehall_hate": "cleaned_data/processed_residencehall_hate.csv",
    "residencehall_vawa": "cleaned_data/processed_residencehall_vawa.csv"
}

def feature_engineering(df, incident_columns, filter_columns):
    
    for year in ['20', '21', '22']:
        year_columns = [col for col in incident_columns if col.endswith(year)]
        df[f'sum_incidents_{year}'] = df[year_columns].sum(axis=1)
    
    incident_types = set(col[:-2] for col in incident_columns)
    for incident in incident_types:
        df[f'total_{incident}'] = df[[col for col in incident_columns if col.startswith(incident)]].sum(axis=1)

    for filter_col in filter_columns:
        for incident_col in incident_columns:
            df[f'{filter_col}_x_{incident_col}'] = df[filter_col] * df[incident_col]
    
    return df

incident_columns = {
    'residencehall_arrest': ['WEAPON', 'DRUG', 'LIQUOR'],
    'residencehall_crime': ['MURD', 'NEG_M', 'RAPE', 'FONDL', 'INCES', 'STATR', 'ROBBE', 'AGG_A', 'BURGLA', 'VEHIC', 'ARSON'],
    'residencehall_discipline': ['WEAPON', 'DRUG', 'LIQUOR'],
    'residencehall_hate': ['MURD', 'MURD_RAC', 'MURD_REL', 'MURD_SEX', 'MURD_GEN', 'MURD_GID', 'MURD_DIS', 'MURD_ET', 'MURD_NAT', 'RAPE', 'RAPE_RAC', 'RAPE_REL', 'RAPE_SEX', 'RAPE_GEN', 'RAPE_GID', 'RAPE_DIS', 'RAPE_ET', 'RAPE_NAT'],
    'residencehall_vawa': ['DOMEST', 'DATING', 'STALK']
}

filter_columns = ['FILTER20', 'FILTER21', 'FILTER22']

def feature_engineering(df, incident_columns, filter_columns):
    for incident in incident_columns:

        #Sum over 3 years
        df[f'{incident}_sum'] = df[f'{incident}20'] + df[f'{incident}21'] + df[f'{incident}22']
        
        #Average over 3 years
        df[f'{incident}_avg'] = (df[f'{incident}20'] + df[f'{incident}21'] + df[f'{incident}22']) / 3
        
        #Differences between years
        df[f'{incident}_diff_20_21'] = df[f'{incident}21'] - df[f'{incident}20']
        df[f'{incident}_diff_21_22'] = df[f'{incident}22'] - df[f'{incident}21']
        df[f'{incident}_diff_20_22'] = df[f'{incident}22'] - df[f'{incident}20']
        
        #Interaction terms
        for filter_column in filter_columns:
            df[f'{filter_column}_x_{incident}20'] = df[filter_column] * df[f'{incident}20']
            df[f'{filter_column}_x_{incident}21'] = df[filter_column] * df[f'{incident}21']
            df[f'{filter_column}_x_{incident}22'] = df[filter_column] * df[f'{incident}22']
    
    return df

for key, path in csv_files.items():
    df = pd.read_csv(path)
    incident_cols = incident_columns[key]
    df = feature_engineering(df, incident_cols, filter_columns)
    
    feature_engineered_file = f'{output_dir}/feature_engineered_{key}.csv'
    df.to_csv(feature_engineered_file, index=False)
    
    print(f"Feature engineered data for {key} saved to {feature_engineered_file}")