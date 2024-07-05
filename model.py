import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv('transformed_data/feature_engineered_oncampus_result.csv')
print("Data loaded successfully!")

filter_columns = ['FILTER20', 'FILTER21', 'FILTER22']
filter_interaction_columns = [col for col in df.columns if '_x_' in col]

start_idx = df.columns.get_loc('Total') + 1
target_columns = [col for col in df.columns[start_idx:] if col not in filter_columns + filter_interaction_columns]
print(f"Identified target columns: {len(target_columns)} columns")

def preprocess_data(data, categorical_features):
    numerical_features = data.drop(columns=categorical_features + target_columns).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = data.drop(columns=target_columns)
    y = data[target_columns] > 0  # Binary classification: whether an incident occurred or not

    X = preprocessor.fit_transform(X)
    return X, y, preprocessor

def create_and_train_model(X_train, y_train, X_test, y_test, num_features):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(num_features,)),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='sigmoid')  # Multi-output binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    return model, history

def train_and_predict(data, level, categorical_features):
    X, y, preprocessor = preprocess_data(data, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]

    model, history = create_and_train_model(X_train, y_train, X_test, y_test, num_features)

    predictions = model.predict(X)
    
    pred_df = pd.DataFrame(predictions, columns=[f'{target}_pred' for target in target_columns])
    result_df = data[[level]].reset_index(drop=True).join(pred_df)
    
    return result_df

categorical_features = ['INSTNM', 'State', 'BRANCH', 'Address', 'City']

print("Training and predicting at university level...")
university_predictions = train_and_predict(df, 'INSTNM', categorical_features)

print("Training and predicting at state level...")
state_predictions = train_and_predict(df, 'State', categorical_features)

def visualize_predictions(predictions, level):
    for col in predictions.columns:
        if '_pred' in col:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=predictions.sort_values(by=col, ascending=False).head(10), x=level, y=col, palette='viridis')
            plt.title(f'Likelihood of {col.split("_pred")[0]} by {level}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'results/{col}_by_{level}.png')
            plt.show()

print("Visualizing predictions at university level...")
visualize_predictions(university_predictions, 'INSTNM')

print("Visualizing predictions at state level...")
visualize_predictions(state_predictions, 'State')