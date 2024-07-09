import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

output_dir = 'figures/oncampus_arrest'

df = pd.read_csv('transformed_data/feature_engineered_oncampus_arrest.csv')

incident_columns = ['WEAPON', 'DRUG', 'LIQUOR']
filter_columns = ['FILTER20', 'FILTER21', 'FILTER22']

# Trend Analysis: Line Plot for Each Incident Type
for incident in incident_columns:
    
    # Sum over years
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[f'{incident}20'], marker='o', label='2020')
    plt.plot(df.index, df[f'{incident}21'], marker='o', label='2021')
    plt.plot(df.index, df[f'{incident}22'], marker='o', label='2022')
    plt.xlabel('Index')
    plt.ylabel(f'{incident} Incidents')
    plt.title(f'Trend of {incident} Incidents Over the Years')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'trend_{incident}_incidents.png'))
    plt.close()

    # Average over years
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[f'{incident}_avg'], marker='o', label=f'{incident}_avg')
    plt.xlabel('Index')
    plt.ylabel(f'Average {incident} Incidents')
    plt.title(f'Average {incident} Incidents Over the Years')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'average_{incident}_incidents.png'))
    plt.close()

# Distribution Analysis: Plot for Each Incident Type 
for incident in incident_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}20'], kde=True)
    plt.xlabel(f'{incident} Incidents in 2020')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {incident} Incidents in 2020')
    plt.savefig(os.path.join(output_dir, f'distribution_{incident}_2020.png'))
    plt.close()

for incident in incident_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}21'], kde=True)
    plt.xlabel(f'{incident} Incidents in 2021')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {incident} Incidents in 2021')
    plt.savefig(os.path.join(output_dir, f'distribution_{incident}_2021.png'))
    plt.close()

for incident in incident_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}22'], kde=True)
    plt.xlabel(f'{incident} Incidents in 2022')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {incident} Incidents in 2022')
    plt.savefig(os.path.join(output_dir, f'distribution_{incident}_2022.png'))
    plt.close()

# Difference Analysis: Histograms for Differences Between Years
for incident in incident_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}_diff_20_21'], kde=True)
    plt.xlabel(f'Difference in {incident} Incidents (2021-2020)')
    plt.ylabel('Frequency')
    plt.title(f'Difference in {incident} Incidents Between 2021 and 2020')
    plt.savefig(os.path.join(output_dir, f'difference_{incident}_20_21.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}_diff_21_22'], kde=True)
    plt.xlabel(f'Difference in {incident} Incidents (2022-2021)')
    plt.ylabel('Frequency')
    plt.title(f'Difference in {incident} Incidents Between 2022 and 2021')
    plt.savefig(os.path.join(output_dir, f'difference_{incident}_21_22.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{incident}_diff_20_22'], kde=True)
    plt.xlabel(f'Difference in {incident} Incidents (2022-2020)')
    plt.ylabel('Frequency')
    plt.title(f'Difference in {incident} Incidents Between 2022 and 2020')
    plt.savefig(os.path.join(output_dir, f'difference_{incident}_20_22.png'))
    plt.close()

# Relationship Analysis: Scatter Plots Between Different Incident Types
for i, incident1 in enumerate(incident_columns):
    for incident2 in incident_columns[i+1:]:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[f'{incident}20'], y=df[f'{incident2}20'])
        plt.xlabel(f'{incident1} Incidents in 2020')
        plt.ylabel(f'{incident2} Incidents in 2020')
        plt.title(f'Relationship between {incident1} and {incident2} Incidents in 2020')
        plt.savefig(os.path.join(output_dir, f'relationship_{incident1}_{incident2}_2020.png'))
        plt.close()

for i, incident1 in enumerate(incident_columns):
    for incident2 in incident_columns[i+1:]:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[f'{incident}21'], y=df[f'{incident2}21'])
        plt.xlabel(f'{incident1} Incidents in 2021')
        plt.ylabel(f'{incident2} Incidents in 2021')
        plt.title(f'Relationship between {incident1} and {incident2} Incidents in 2021')
        plt.savefig(os.path.join(output_dir, f'relationship_{incident1}_{incident2}_2021.png'))
        plt.close()

for i, incident1 in enumerate(incident_columns):
    for incident2 in incident_columns[i+1:]:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[f'{incident}22'], y=df[f'{incident2}22'])
        plt.xlabel(f'{incident1} Incidents in 2022')
        plt.ylabel(f'{incident2} Incidents in 2022')
        plt.title(f'Relationship between {incident1} and {incident2} Incidents in 2022')
        plt.savefig(os.path.join(output_dir, f'relationship_{incident1}_{incident2}_2022.png'))
        plt.close()


# Correlation Heatmap for Incident Sums and Filters
correlation_columns = [f'{incident}_sum' for incident in incident_columns] + filter_columns
correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Incident Sums and Filters')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap_incidents_filters.png'))
plt.close()

# Interaction Terms Visualization
interaction_columns = [f'{filter_col}_x_{incident}20' for filter_col in filter_columns for incident in incident_columns]
for interaction in interaction_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[interaction], kde=True)
    plt.xlabel(interaction)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {interaction}')
    plt.savefig(os.path.join(output_dir, f'distribution_{interaction}.png'))
    plt.close()

# Sum of All Incidents for Each Year
df['total_incidents_2020'] = df[[f'{incident}20' for incident in incident_columns]].sum(axis=1)
df['total_incidents_2021'] = df[[f'{incident}21' for incident in incident_columns]].sum(axis=1)
df['total_incidents_2022'] = df[[f'{incident}22' for incident in incident_columns]].sum(axis=1)

total_incidents = {
    'Year': ['2020', '2021', '2022'],
    'Total Incidents': [df['total_incidents_2020'].sum(), df['total_incidents_2021'].sum(), df['total_incidents_2022'].sum()]
}

total_incidents_df = pd.DataFrame(total_incidents)

plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Total Incidents', data=total_incidents_df)
plt.xlabel('Year')
plt.ylabel('Total Incidents')
plt.title('Total Incidents for Each Year')
plt.savefig(os.path.join(output_dir, 'total_incidents_per_year.png'))
plt.close()

print(f"Visualizations saved in the {output_dir} directory.")