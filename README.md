# Comment Category Prediction Using Machine Learning

A machine learning-based text classification project that predicts the category label of online comments using a combination of **textual, categorical, numerical, and temporal features**. The project explores multiple classification approaches and develops a soft-voting ensemble to improve overall classification performance, with **Macro F1 Score** used as an important evaluation metric.

## 📌 Project Overview

The objective of this project is to automatically classify comments into predefined categories based on information contained in the comment and its associated metadata.

Rather than relying only on the comment text, the project combines multiple sources of information, including:

* Comment text
* Upvotes and downvotes
* Emoticon-related features
* Additional binary indicators
* Race
* Religion
* Gender
* Disability
* Comment length
* Word count
* Creation time
* Day of the week

This multi-modal feature approach allows the models to use both linguistic and contextual information when predicting comment categories.

---

## 🎯 Objectives

* Analyze and understand the comment-category dataset.
* Perform exploratory data analysis and data-quality checks.
* Extract meaningful features from raw comments.
* Convert text into numerical representations using TF-IDF.
* Encode categorical variables using One-Hot Encoding.
* Scale numerical features for machine learning models.
* Engineer additional temporal and text-based features.
* Compare multiple machine learning algorithms.
* Optimize model hyperparameters.
* Build an ensemble classification model.
* Evaluate performance using Accuracy and Macro F1 Score.
* Generate predictions for the test dataset.

---

## 🗂️ Dataset

The project uses the **Comment Category Prediction Challenge** dataset.

The main files are:

```text
train.csv
test.csv
Sample.csv
```

### Important Features

| Feature        | Type        | Description                    |
| -------------- | ----------- | ------------------------------ |
| `comment`      | Text        | Comment content                |
| `race`         | Categorical | Associated categorical feature |
| `religion`     | Categorical | Associated categorical feature |
| `gender`       | Categorical | Associated categorical feature |
| `disability`   | Categorical | Associated categorical feature |
| `upvote`       | Numerical   | Number of upvotes              |
| `downvote`     | Numerical   | Number of downvotes            |
| `emoticon_1`   | Numerical   | Emoticon-related feature       |
| `emoticon_2`   | Numerical   | Emoticon-related feature       |
| `emoticon_3`   | Numerical   | Emoticon-related feature       |
| `if_1`         | Numerical   | Additional indicator feature   |
| `if_2`         | Numerical   | Additional indicator feature   |
| `created_date` | Temporal    | Comment creation timestamp     |
| `label`        | Target      | Comment category               |

---

## 🔍 Exploratory Data Analysis

The notebook performs several exploratory analyses, including:

* Training and testing dataset dimensions
* Target-label distribution
* Missing-value analysis
* Numerical feature statistics
* Correlation analysis
* Feature inspection
* Comment-length analysis
* Word-count analysis
* Temporal feature analysis

Visualizations are generated using **Matplotlib** and **Seaborn**.

---

## 🧹 Data Preprocessing

### Missing Values

Missing comments are replaced with a placeholder value to prevent errors during text processing.

Categorical missing values are handled using:

```text
unknown
```

Numerical features are processed using appropriate scaling.

### Train-Validation Split

The training dataset is divided into training and validation sets using:

```python
test_size = 0.2
random_state = 42
stratify = y
```

Stratification ensures that the label distribution is maintained across the training and validation sets.

---

## 📝 Text Feature Engineering

The `comment` feature is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The final ensemble configuration uses:

```python
TfidfVectorizer(
    max_features=14000,
    ngram_range=(1, 2),
    stop_words='english'
)
```

Both unigrams and bigrams are considered to capture individual words as well as short word combinations.

---

## 🧮 Additional Feature Engineering

The project derives additional features from the original data.

### Text Features

From each comment:

* Character count
* Word count

### Temporal Features

From `created_date`:

* Hour
* Day of the week

These features provide additional contextual information that may help distinguish between comment categories.

---

## 🔤 Categorical Feature Processing

The categorical variables:

```text
race
religion
gender
disability
```

are processed using:

1. Missing-value imputation
2. One-Hot Encoding

Unknown categories encountered during prediction are handled using:

```python
handle_unknown='ignore'
```

---

## 📏 Numerical Feature Processing

Numerical variables are standardized using:

```python
StandardScaler()
```

The numerical feature set includes:

```text
upvote
downvote
emoticon_1
emoticon_2
emoticon_3
if_1
if_2
char_count
word_count
hour
day_of_week
```

---

## 🤖 Machine Learning Models

The notebook explores multiple machine learning approaches.

### 1. Dummy Classifier

A most-frequent baseline classifier is used to establish a simple reference performance.

### 2. Logistic Regression

Logistic Regression is used as a strong linear baseline for the combined feature representation.

The final ensemble configuration uses:

```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
```

### 3. SGD Classifier

An SGD-based classifier with logistic loss is explored, along with hyperparameter tuning using `GridSearchCV`.

The search considers:

* `alpha`
* `penalty`

### 4. Multinomial Naive Bayes

A Naive Bayes approach is explored using TF-IDF features, dimensionality reduction with Truncated SVD, categorical encoding, and numerical features.

### 5. Multi-Layer Perceptron

An MLP neural network is included in the final ensemble:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=42
)
```

### 6. LightGBM

A LightGBM classifier is also included:

```python
LGBMClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42
)
```

---

## 🤝 Soft Voting Ensemble

The final model combines three different classifiers:

```text
                 ┌──────────────────┐
                 │  Logistic        │
                 │  Regression      │
                 └────────┬─────────┘
                          │
                          │
                 ┌────────▼─────────┐
                 │                  │
Input Features ─►│ Soft Voting      │──► Final Prediction
                 │ Ensemble         │
                 │                  │
                 └────────▲─────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
       ┌──────┴──────┐         ┌──────┴──────┐
       │     MLP     │         │  LightGBM   │
       └─────────────┘         └─────────────┘
```

The ensemble uses **soft voting**, meaning the predicted class probabilities from the individual models are combined to determine the final prediction.

This approach combines the strengths of linear, neural-network, and gradient-boosting models.

---

## 📊 Evaluation Metrics

The project evaluates models using:

### Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Macro F1 Score

Macro F1 calculates the F1 score independently for each class and then takes their unweighted average.

This is particularly useful when evaluating performance across multiple categories because each class contributes equally to the final score.

The final ensemble reports:

```python
accuracy_score(y_val, val_preds)
f1_score(y_val, val_preds, average='macro')
```

---

## 🔄 Complete Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Missing Value Handling
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
 ┌───────────────┬────────────────┬────────────────┐
 │               │                │
Text Features  Categorical     Numerical/Temporal
 │               │                │
TF-IDF         One-Hot         Standard Scaling
 │               │                │
 └───────────────┴────────────────┘
                 ↓
         Train / Validation Split
                 ↓
       Multiple ML Classifiers
                 ↓
       Soft Voting Ensemble
                 ↓
        Validation Evaluation
                 ↓
       Test Dataset Prediction
                 ↓
          submission.csv
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Jupyter Notebook / Kaggle Notebook

---

## 📦 Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/comment-category-prediction.git
cd comment-category-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the dataset files in the appropriate directory or configure the notebook paths according to your environment.

### 4. Run the notebook

Open the Jupyter/Kaggle notebook and execute the cells sequentially.

The notebook performs:

1. Dataset loading
2. Data inspection
3. Exploratory analysis
4. Feature engineering
5. Feature preprocessing
6. Model training
7. Hyperparameter tuning
8. Ensemble training
9. Validation
10. Test prediction
11. Submission generation

---

## 📄 Output

The final prediction file is:

```text
submission.csv
```

with the following structure:

```text
ID,label
1,<predicted_label>
2,<predicted_label>
...
```

The predictions generated by the final soft-voting ensemble are saved for submission.

---

## ⭐ Key Highlights

* Multi-feature comment classification
* TF-IDF-based text representation
* Unigram and bigram features
* Categorical feature encoding
* Numerical feature standardization
* Text and temporal feature engineering
* Hyperparameter tuning with GridSearchCV
* Logistic Regression
* SGD Classifier
* Multinomial Naive Bayes
* MLP Neural Network
* LightGBM
* Soft-voting ensemble
* Accuracy and Macro F1 evaluation
* Automated submission generation

---

## 👨‍💻 Author

**Ishan Bandyopadhyay**

Developed as a machine learning project for automated comment category prediction using text and metadata-based feature engineering.

---

## 📜 License

This project is intended for educational, research, and portfolio purposes. Dataset usage and redistribution should follow the terms and conditions of the original **Comment Category Prediction Challenge**.
