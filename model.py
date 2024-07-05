import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

print("Loading data...")
df = pd.read_csv('/mnt/data/feature_engineered_oncampus_result.csv')
print("Data loaded successfully!")

filter_columns = ['FILTER20', 'FILTER21', 'FILTER22']
filter_interaction_columns = [col for col in df.columns if '_x_' in col]

# Identify the columns that end with '_sum' and '_average'
sum_columns = [col for col in df.columns if col.endswith('_sum')]
avg_columns = [col for col in df.columns if col.endswith('_average')]

print(f"Identified sum columns: {len(sum_columns)} columns")
print(f"Identified average columns: {len(avg_columns)} columns")

def preprocess_data(data, categorical_features, target_columns):
    numerical_features = data.drop(columns=categorical_features + target_columns).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = data.drop(columns=target_columns)
    y = data[target_columns] > 0  # Binary classification: whether an incident occurred or not

    X = preprocessor.fit_transform(X)
    y = y.values  # Convert y to NumPy array for correct indexing
    return X, y, preprocessor

def create_and_train_model(X_train, y_train, X_test, y_test, num_features):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(num_features,)),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='sigmoid')  # Multi-output binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    print("Training completed!")

    return model, history

def train_and_predict(data, level, categorical_features, target_columns):
    X, y, preprocessor = preprocess_data(data, categorical_features, target_columns)

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    model, history = create_and_train_model(X_train, y_train, X_test, y_test, num_features)

    predictions = model.predict(X)
    
    pred_df = pd.DataFrame(predictions, columns=[f'{target}_pred' for target in target_columns])
    result_df = data[[level]].reset_index(drop=True).join(pred_df)
    
    return model, history, X_test, y_test, result_df

categorical_features = ['INSTNM', 'State', 'BRANCH', 'Address', 'City']

print("Training and predicting at university level (sum)...")
model_univ_sum, history_univ_sum, X_test_univ_sum, y_test_univ_sum, university_predictions_sum = train_and_predict(df, 'INSTNM', categorical_features, sum_columns)

print("Training and predicting at university level (average)...")
model_univ_avg, history_univ_avg, X_test_univ_avg, y_test_univ_avg, university_predictions_avg = train_and_predict(df, 'INSTNM', categorical_features, avg_columns)

print("Training and predicting at state level (sum)...")
model_state_sum, history_state_sum, X_test_state_sum, y_test_state_sum, state_predictions_sum = train_and_predict(df, 'State', categorical_features, sum_columns)

print("Training and predicting at state level (average)...")
model_state_avg, history_state_avg, X_test_state_avg, y_test_state_avg, state_predictions_avg = train_and_predict(df, 'State', categorical_features, avg_columns)

def visualize_predictions(predictions, level, incident_type):
    pred_col = f'{incident_type}_pred'
    if pred_col in predictions.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=predictions.sort_values(by=pred_col, ascending=False).head(10), x=level, y=pred_col, palette='viridis')
        plt.title(f'Likelihood of {incident_type} by {level}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'results/{incident_type}_by_{level}.png')
        plt.show()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_test, y_pred, classes):
    for i, class_name in enumerate(classes):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i] > 0.5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {class_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

def evaluate_model(model, X_test, y_test, target_columns):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Classification report
    for i, column in enumerate(target_columns):
        print(f"Evaluation for {column}:")
        print(classification_report(y_test[:, i], y_pred_binary[:, i]))
        print("\n")

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, target_columns)

    # ROC Curve
    for i, column in enumerate(target_columns):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        auc = roc_auc_score(y_test[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{column} (AUC: {auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Select an incident type to visualize
incident_type = 'arrest_WEAPON'

print(f"Visualizing predictions for {incident_type}_sum at university level...")
visualize_predictions(university_predictions_sum, 'INSTNM', incident_type + '_sum')

print(f"Visualizing predictions for {incident_type}_average at university level...")
visualize_predictions(university_predictions_avg, 'INSTNM', incident_type + '_average')

print(f"Visualizing predictions for {incident_type}_sum at state level...")
visualize_predictions(state_predictions_sum, 'State', incident_type + '_sum')

print(f"Visualizing predictions for {incident_type}_average at state level...")
visualize_predictions(state_predictions_avg, 'State', incident_type + '_average')

print("Plotting learning curves for university level model (sum)...")
plot_learning_curves(history_univ_sum)

print("Plotting learning curves for university level model (average)...")
plot_learning_curves(history_univ_avg)

print("Plotting learning curves for state level model (sum)...")
plot_learning_curves(history_state_sum)

print("Plotting learning curves for state level model (average)...")
plot_learning_curves(history_state_avg)

print("Evaluating university level model (sum)...")
evaluate_model(model_univ_sum, X_test_univ_sum, y_test_univ_sum, sum_columns)

print("Evaluating university level model (average)...")
evaluate_model(model_univ_avg, X_test_univ_avg, y_test_univ_avg, avg_columns)

print("Evaluating state level model (sum)...")
evaluate_model(model_state_sum, X_test_state_sum, y_test_state_sum, sum_columns)

print("Evaluating state level model (average)...")
evaluate_model(model_state_avg, X_test_state_avg, y_test_state_avg, avg_columns)