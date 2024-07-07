import pandas as pd
import os

input_file = 'cleaned_data/processed_oncampus_result.csv'
output_file = 'transformed_data/feature_engineered_oncampus_result.csv'

incident_columns = {
    'arrest': ['WEAPON', 'DRUG', 'LIQUOR'],
    'crime': ['MURD', 'NEG_M', 'RAPE', 'FONDL', 'INCES', 'STATR', 'ROBBE', 'AGG_A', 'BURGLA', 'VEHIC', 'ARSON'],
    'discipline': ['WEAPON', 'DRUG', 'LIQUOR'],
    'vawa': ['DOMEST', 'DATING', 'STALK']
}

filter_columns = {
    'arrest': ['arrest_FILTER20', 'arrest_FILTER21', 'arrest_FILTER22'],
    'crime': ['crime_FILTER20', 'crime_FILTER21', 'crime_FILTER22'],
    'discipline': ['discipline_FILTER20', 'discipline_FILTER21', 'discipline_FILTER22'],
    'vawa': ['vawa_FILTER20', 'vawa_FILTER21', 'vawa_FILTER22']
}

def feature_engineering(df, prefix, incident_types, filter_columns):
    for incident in incident_types:

        # Sum over 3 years
        df[f'{prefix}_{incident}_sum'] = df[f'{prefix}_{incident}20'] + df[f'{prefix}_{incident}21'] + df[f'{prefix}_{incident}22']
        
        # Average over 3 years
        df[f'{prefix}_{incident}_avg'] = (df[f'{prefix}_{incident}20'] + df[f'{prefix}_{incident}21'] + df[f'{prefix}_{incident}22']) / 3
        
        # Differences between years
        df[f'{prefix}_{incident}_diff_20_21'] = df[f'{prefix}_{incident}21'] - df[f'{prefix}_{incident}20']
        df[f'{prefix}_{incident}_diff_21_22'] = df[f'{prefix}_{incident}22'] - df[f'{prefix}_{incident}21']
        df[f'{prefix}_{incident}_diff_20_22'] = df[f'{prefix}_{incident}22'] - df[f'{prefix}_{incident}20']
        
        # Interaction terms
        for filter_column in filter_columns:
            df[f'{filter_column}_x_{prefix}_{incident}20'] = df[filter_column] * df[f'{prefix}_{incident}20']
            df[f'{filter_column}_x_{prefix}_{incident}21'] = df[filter_column] * df[f'{prefix}_{incident}21']
            df[f'{filter_column}_x_{prefix}_{incident}22'] = df[filter_column] * df[f'{prefix}_{incident}22']
    
    return df

# Load the input file
df = pd.read_csv(input_file)

# Apply feature engineering for each incident type
for prefix, incidents in incident_columns.items():
    filters = filter_columns[prefix]
    df = feature_engineering(df, prefix, incidents, filters)

# Save the updated DataFrame to the output file
df.to_csv(output_file, index=False)

print(f"Feature engineered data saved to {output_file}")