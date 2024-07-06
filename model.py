import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import os

output_folder = 'results/noncampus incidents'
os.makedirs(output_folder, exist_ok=True)

print("Loading data...")
df = pd.read_csv('transformed_data/feature_engineered_noncampus_result.csv')
print("Data loaded successfully!")

filter_columns = ['FILTER20', 'FILTER21', 'FILTER22']
filter_interaction_columns = [col for col in df.columns if '_x_' in col]

# Identify the columns that end with '_sum' and '_avg'
sum_columns = [col for col in df.columns if col.endswith('_sum')]
avg_columns = [col for col in df.columns if col.endswith('_avg')]

# Order incidents as they appear in the CSV file by combining sum and average columns
incident_columns = [col for col in df.columns if col.endswith('_sum') or col.endswith('_avg')]
incident_types = sorted(set([col.rsplit('_', 1)[0] for col in incident_columns]), key=lambda x: df.columns.tolist().index(x + '_sum') if x + '_sum' in df.columns else df.columns.tolist().index(x + '_avg'))

print(f"Identified incident types: {incident_types}")

def preprocess_data(data, categorical_features, target_columns):
    numerical_features = data.drop(columns=categorical_features + target_columns).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = data.drop(columns=target_columns)
    y = data[target_columns] > 0  # Binary classification: whether an incident occurred or not

    return X, y, preprocessor

def create_and_train_model(X_train, y_train, X_test, y_test, preprocessor, level, incident_type):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    param_distributions = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42, return_train_score=True)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Save learning curves
    plt.figure(figsize=(12, 6))
    plt.plot(search.cv_results_['mean_train_score'], label='Training Score')
    plt.plot(search.cv_results_['mean_test_score'], label='Validation Score')
    plt.title(f'Learning Curves for {incident_type} at {level} level')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_folder}/{incident_type}/learning_curves_{incident_type}_{level}.png')
    plt.show()

    return best_model

def train_and_predict(data, level, categorical_features, target_columns, incident_type):
    X, y, preprocessor = preprocess_data(data, categorical_features, target_columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_and_train_model(X_train, y_train, X_test, y_test, preprocessor, level, incident_type)

    predictions = model.predict(X)
    
    pred_df = pd.DataFrame(predictions, columns=[f'{target}_pred' for target in target_columns])
    if level == 'INSTNM':
        result_df = data[['INSTNM', 'BRANCH']].reset_index(drop=True).join(pred_df)
    else:
        result_df = data[[level]].reset_index(drop=True).join(pred_df)
    
    return model, X_test, y_test.values, result_df  # Ensure y_test is a NumPy array

categorical_features = ['INSTNM', 'State', 'BRANCH', 'Address', 'City']

def get_top_predictions(predictions, level, incident_type, top_n=20):
    pred_cols = [col for col in predictions.columns if col.startswith(incident_type) and col.endswith('_pred')]
    predictions['combined_pred'] = predictions[pred_cols].mean(axis=1)
    top_predictions = predictions.sort_values(by='combined_pred', ascending=False).head(top_n)
    top_predictions.insert(0, 'Serial Number', range(1, 1 + len(top_predictions)))
    return top_predictions[[level, 'combined_pred']]

def get_top_distinct_states(predictions, incident_type, top_n=10):
    pred_cols = [col for col in predictions.columns if col.startswith(incident_type) and col.endswith('_pred')]
    predictions['combined_pred'] = predictions[pred_cols].mean(axis=1)
    top_predictions = predictions.drop_duplicates(subset=['State']).sort_values(by='combined_pred', ascending=False).head(top_n)
    top_predictions.insert(0, 'Serial Number', range(1, 1 + len(top_predictions)))
    return top_predictions[['State', 'combined_pred']]

def plot_confusion_matrix(y_test, y_pred, classes, level, incident_type):
    for i, class_name in enumerate(classes):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i] > 0.5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {class_name} at {level} level')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{output_folder}/{incident_type}/confusion_matrix_{class_name}_{level}.png')
        plt.show()

def evaluate_model(model, X_test, y_test, target_columns, level, incident_type):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    eval_file = open(f'{output_folder}/{incident_type}/evaluation_{level}.txt', 'a')
    
    # Classification report
    for i, column in enumerate(target_columns):
        eval_file.write(f"Evaluation for {column}:\n")
        eval_file.write(classification_report(y_test[:, i], y_pred_binary[:, i]))
        eval_file.write("\n")
        print(f"Evaluation for {column}:")
        print(classification_report(y_test[:, i], y_pred_binary[:, i]))
        print("\n")

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_binary, target_columns, level, incident_type)

    # ROC Curve
    plt.figure(figsize=(12, 6))
    for i, column in enumerate(target_columns):
        if len(np.unique(y_test[:, i])) == 1:  # Avoid AUC calculation if only one class present
            continue
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        auc = roc_auc_score(y_test[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{column} (AUC: {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {incident_type} at {level} level')
    plt.legend()
    plt.savefig(f'{output_folder}/{incident_type}/roc_curve_{incident_type}_{level}.png')
    plt.show()
    
    eval_file.close()

top_universities = {}
top_states = {}

for incident_type in incident_types:
    print(f"Processing incident type: {incident_type}")
    
    # Create a subfolder for the incident type
    incident_folder = os.path.join(output_folder, incident_type)
    os.makedirs(incident_folder, exist_ok=True)
    
    # Combine sum and average columns for the incident type
    target_columns = [col for col in df.columns if col.startswith(incident_type) and (col.endswith('_sum') or col.endswith('_avg'))]
    
    print(f"Training and predicting for {incident_type} at university level...")
    model_univ, _, y_test_univ, university_predictions = train_and_predict(df, 'INSTNM', categorical_features, target_columns, incident_type)

    print(f"Training and predicting for {incident_type} at state level...")
    model_state, _, y_test_state, state_predictions = train_and_predict(df, 'State', categorical_features, target_columns, incident_type)

    # Get top 20 universities
    top_univ = get_top_predictions(university_predictions, 'INSTNM', incident_type, top_n=20)
    top_universities[incident_type] = top_univ

    # Get top 10 distinct states
    top_state = get_top_distinct_states(state_predictions, incident_type, top_n=10)
    top_states[incident_type] = top_state

    print(f"Top 20 universities for {incident_type}:")
    print(top_univ)

    print(f"Top 10 states for {incident_type}:")
    print(top_state)

    # Evaluate models
    print("Evaluating university level model...")
    evaluate_model(model_univ, _, y_test_univ, target_columns, 'university', incident_type)

    print("Evaluating state level model...")
    evaluate_model(model_state, _, y_test_state, target_columns, 'state', incident_type)

    # Save prediction results to CSV with serial numbers
    university_predictions.insert(0, 'Serial Number', range(1, 1 + len(university_predictions)))
    university_predictions.to_csv(f'{output_folder}/{incident_type}/predictions_universities_{incident_type}.csv', index=False)

    state_predictions.insert(0, 'Serial Number', range(1, 1 + len(state_predictions)))
    state_predictions.to_csv(f'{output_folder}/{incident_type}/predictions_states_{incident_type}.csv', index=False)

    for incident_type, data in top_universities.items():
        data.to_csv(f'{output_folder}/{incident_type}/top_universities_{incident_type}.csv', index=False)

    for incident_type, data in top_states.items():
        data.to_csv(f'{output_folder}/{incident_type}/top_states_{incident_type}.csv', index=False)

    print("All predictions and evaluations saved successfully!")