# 🤖 AI Task Management System

An intelligent, machine learning–powered platform built with **Flask**, **Scikit-learn**, and **XGBoost** to **predict task category**, **determine priority**, and **recommend the most appropriate user** based on task descriptions and workload data. Using NLP and classification techniques on a structured dataset, this system enables smarter, faster task assignment.

---

## ✅ Key Features

- 🧠 **Category Classification**  
  Automatically predicts the task category from natural language descriptions (e.g., Bug, Feature, Research).

- 🎯 **Priority Prediction**  
  Classifies tasks into Low, Medium, or High priority based on description and workload.

- 👥 **User Assignment**  
  Suggests the best user for a task by evaluating current workload.

- 🌐 **Interactive Web Interface**  
  Built using Flask with a clean, user-friendly layout.

---

## 📦 Dataset Overview

- **Records**: 3,000 tasks
- **Columns**:  
  `task_id`, `task_description`, `category`, `priority`, `assigned_user`, `due_date`, `user_workload`

- **Data Types**:  
  - Text: Task descriptions  
  - Categorical: Category, Priority, Assigned User  
  - Numeric: Workload  
  - Date: Due Date

---

## 📈 Model Performance

### 🧠 Task Category Classification

| Model       | Accuracy | F1 Score |
|-------------|----------|----------|
| Naive Bayes | 88.00%   | 89.99%   |
| SVM         | 88.53%   | 89.30%   |
| XGBoost     | 88.67%   | 88.81%   |

> ✔️ All models performed consistently well. XGBoost provided balanced macro and weighted scores across categories.

---

### 🔥 Priority Prediction

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| Random Forest | 99.87%   | 99.87%   |
| XGBoost       | 100.00%  | 100.00%  |

> 🚀 **XGBoost achieved perfect classification** for priority labels on the test data — strong indicator of high model reliability (though it may suggest potential overfitting on synthetic data).

---

## 🔍 Example Task Predictions

### 🧾 Task:
> Create a new database index to speed up user queries  
> **Workload**: 5

- 🏷 **Predicted Category**: Testing  
- ⚠️ **Priority**: Low  
- 👤 **Recommended User**: `user_18`

---

### 🧾 Task:
> The payment confirmation email is not being sent to users  
> **Workload**: 12

- 🏷 **Predicted Category**: Feature  
- ⚠️ **Priority**: Medium  
- 👤 **Recommended User**: `user_18`

---

### 🧾 Task:
> Design a new logo for the mobile application  
> **Workload**: 3

- 🏷 **Predicted Category**: Research  
- ⚠️ **Priority**: Low  
- 👤 **Recommended User**: `user_18`

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Infotact
