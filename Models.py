# AI-Powered Task Management System

# ==============================================================================
# SETUP: Importing necessary libraries
# ==============================================================================
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    # Ensure punkt_tab is downloaded if needed for the specific NLTK version/config
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


print("Libraries imported and NLTK data checked.")

# ==============================================================================
# DATA LOADING: Load the generated dataset
# ==============================================================================
print("\n--- Loading Dataset ---")
# ✅ Load the dataframe from the generated CSV file
df = pd.read_csv("synthetic_task_dataset.csv")
print("Dataset loaded successfully.")
print("Dataset preview:")
print(df.head())
print("\nDataset Info:")
df.info()


# ==============================================================================
# WEEK 1: EDA and NLP Preprocessing
# ==============================================================================
print("\n--- WEEK 1: EDA & NLP Preprocessing ---")

plt.style.use('seaborn-v0_8-whitegrid')

# EDA Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(y='category', data=df, order=df['category'].value_counts().index, palette='viridis')
plt.title('Task Category Distribution')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='priority', data=df, order=['Low', 'Medium', 'High'], palette='plasma')
plt.title('Task Priority Distribution')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['user_workload'], bins=15, kde=True, color='skyblue')
plt.title('User Workload Distribution')
plt.xlabel('Number of Tasks (Workload)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(processed)

df['processed_description'] = df['task_description'].apply(preprocess_text)

# ==============================================================================
# WEEK 2: Feature Extraction and Task Classification
# ==============================================================================
print("\n--- WEEK 2: Task Classification ---")

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_description'])
y_category = df['category']

# Encode category labels
category_encoder = LabelEncoder()
y_category_encoded = category_encoder.fit_transform(y_category)

# Split data using encoded labels for XGBoost
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_category, test_size=0.25, stratify=y_category, random_state=42)
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_tfidf, y_category_encoded, test_size=0.25, stratify=y_category_encoded, random_state=42)


def evaluate_model(model, X_test, y_test, model_name, is_encoded=False, encoder=None, original_labels=None):
    y_pred = model.predict(X_test)

    # Decode predictions if necessary for evaluation metrics and report
    if is_encoded and encoder:
        y_test_decoded = encoder.inverse_transform(y_test)
        y_pred_decoded = encoder.inverse_transform(y_pred)
    else:
        y_test_decoded = y_test
        y_pred_decoded = y_pred


    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded):.4f}")
    print(f"Precision: {precision_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test_decoded, y_pred_decoded, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    plt.figure(figsize=(10, 6))
    # Use sorted unique labels from the original data for tick labels
    if original_labels is not None:
        labels_for_ticks = sorted(original_labels.unique())
    else: # Fallback to decoded test labels if original not provided (though original is preferred)
        labels_for_ticks = sorted(np.unique(y_test_decoded))

    # Ensure labels are passed as a list
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=list(labels_for_ticks), yticklabels=list(labels_for_ticks))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Train Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
evaluate_model(nb_classifier, X_test, y_test, "Naive Bayes", original_labels=y_category)

# Train SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
evaluate_model(svm_classifier, X_test, y_test, "SVM", original_labels=y_category)

# Train XGBoost
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# Fit XGBoost with encoded labels
xgb_classifier.fit(X_train_encoded, y_train_encoded)
# Evaluate XGBoost with encoded test labels and the encoder
evaluate_model(xgb_classifier, X_test_encoded, y_test_encoded, "XGBoost", is_encoded=True, encoder=category_encoder, original_labels=y_category)


# ==============================================================================
# WEEK 3: Priority Prediction and Workload Balancing
# ==============================================================================
print("\n--- WEEK 3: Priority Prediction ---")

y_priority = df['priority']
workload_feature = df['user_workload'].values.reshape(-1, 1)
X_combined = hstack([X_tfidf, workload_feature])

# Encode priority labels
priority_encoder = LabelEncoder()
y_priority_encoded = priority_encoder.fit_transform(y_priority)


# Split data using encoded labels for XGBoost priority
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_combined, y_priority, test_size=0.25, stratify=y_priority, random_state=42)
X_train_p_encoded, X_test_p_encoded, y_train_p_encoded, y_test_p_encoded = train_test_split(X_combined, y_priority_encoded, test_size=0.25, stratify=y_priority_encoded, random_state=42)


# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_p, y_train_p)
evaluate_model(rf_classifier, X_test_p, y_test_p, "Random Forest (Priority)", original_labels=y_priority)

# XGBoost for Priority
xgb_priority = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# Fit XGBoost priority with encoded labels
xgb_priority.fit(X_train_p_encoded, y_train_p_encoded)
# Evaluate XGBoost priority with encoded test labels and the encoder
evaluate_model(xgb_priority, X_test_p_encoded, y_test_p_encoded, "XGBoost (Priority)", is_encoded=True, encoder=priority_encoder, original_labels=y_priority)


# ==============================================================================
# WEEK 4: Finalization and Reporting
# ==============================================================================
print("\n--- WEEK 4: Final Models and Predictions ---")

# Select final models (change here if XGBoost outperforms)
final_classification_model = xgb_classifier
final_priority_model = xgb_priority

def assign_task_with_balancing(new_task_description, user_data):
    processed = preprocess_text(new_task_description)
    # Re-calculate task_vector within this function
    task_vector = tfidf_vectorizer.transform([processed])

    # Predict category using the encoded model and decode the prediction
    predicted_category_encoded = final_classification_model.predict(task_vector)[0]
    predicted_category = category_encoder.inverse_transform([predicted_category_encoded])[0]

    avg_workload = user_data.groupby('assigned_user')['user_workload'].mean()
    best_user = avg_workload.idxmin()
    print(f"New Task: '{new_task_description}'")
    print(f"Predicted Category: {predicted_category}")
    print(f"Recommended User: {best_user}")
    return best_user

user_workload_df = df[['assigned_user', 'user_workload']]

def generate_task_analysis(task_description, user_workload):
    print("\n" + "="*50)
    print("      AI Task Management System - Analysis")
    print("="*50)
    print(f"Task: {task_description}")
    print(f"Workload: {user_workload}")
    processed = preprocess_text(task_description)
    text_vector = tfidf_vectorizer.transform([processed])

    # Predict category using the encoded model and decode the prediction
    predicted_category_encoded = final_classification_model.predict(text_vector)[0]
    predicted_category = category_encoder.inverse_transform([predicted_category_encoded])[0]
    print(f"Category → {predicted_category}")


    workload_input = np.array([[user_workload]])
    combined_vector = hstack([text_vector, workload_input])

    # Predict priority using the encoded model and decode the prediction
    predicted_priority_encoded = final_priority_model.predict(combined_vector)[0]
    predicted_priority = priority_encoder.inverse_transform([predicted_priority_encoded])[0]
    print(f"Priority → {predicted_priority}")


    # Re-using the existing function for user assignment
    # Removed redundant print statements as assign_task_with_balancing already prints
    recommended_user = assign_task_with_balancing(task_description, user_workload_df)
    print("="*50)

# Run examples
generate_task_analysis("Create a new database index to speed up user queries", 5)
generate_task_analysis("The payment confirmation email is not being sent to users", 12)
generate_task_analysis("Design a new logo for the mobile application", 3)

print("\nAll tasks completed. Dataset saved, models trained (Naive Bayes, SVM, XGBoost, Random Forest), and predictions demonstrated.")