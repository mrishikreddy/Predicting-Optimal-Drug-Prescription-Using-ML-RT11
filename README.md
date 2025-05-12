# Predicting Optimal Drug Prescription Using ML

Welcome to the **Predicting Optimal Drug Prescription Using ML** project, a Python-based machine learning application that predicts the appropriate drug for a patient based on their age, sex, blood pressure, cholesterol levels, and sodium-to-potassium ratio. Using a decision tree classifier, this project demonstrates how to preprocess data, train a model, and visualize the decision-making process. It's an excellent resource for learning classification techniques and decision tree modeling.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dataset](#dataset)

## Features
- **Data Preprocessing**: Encodes categorical variables (sex, blood pressure, cholesterol) for model compatibility.
- **Decision Tree Classifier**: Predicts the appropriate drug using patient attributes.
- **Model Evaluation**: Computes accuracy to assess model performance.
- **Visualization**: Generates a decision tree image (`tree.png`) to visualize the model's decision-making process.
- Built with popular Python libraries: NumPy, Pandas, Scikit-learn, and Matplotlib.
- Simple and educational code for learning classification and decision tree concepts.

## Installation
To set up the **Drug Classification with Decision Trees** project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/drug-classification.git
   cd drug-classification
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.6+ installed. Install the required libraries using pip:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. **Install Graphviz for Visualization** (optional, for decision tree image):
   Install Graphviz and the required Python packages:
   ```bash
   conda install -c conda-forge pydotplus python-graphviz
   ```
   Alternatively, install Graphviz manually:
   - Download and install Graphviz from [here](https://graphviz.org/download/).
   - Add Graphviz's `bin` directory to your system PATH.
   - Install Python packages:
     ```bash
     pip install pydotplus graphviz
     ```

4. **Run the Script**:
   Execute the Python script:
   ```bash
   python drug_classification.py
   ```

## Usage
Running the script performs the following steps:
1. Downloads and loads the `drug200.csv` dataset from a provided URL.
2. Preprocesses categorical variables (Sex, BP, Cholesterol) using label encoding.
3. Splits the data into 70% training and 30% testing sets.
4. Trains a decision tree classifier with entropy criterion and a maximum depth of 4.
5. Predicts drugs for the test set and prints a sample of predictions alongside actual values.
6. Outputs the model's accuracy (e.g., ~0.983).
7. Generates a decision tree visualization as `tree.png` (requires Graphviz).

Example output:
```
['drugY' 'drugC' 'drugX' 'drugA' 'drugB']
['drugY' 'drugC' 'drugX' 'drugA' 'drugB']
DecisionTrees's Accuracy: 0.9833333333333333
```

To modify the script (e.g., change the tree depth or features), edit the relevant parameters in the code.

## How It Works
The project uses a decision tree classifier to predict the appropriate drug for a patient based on their attributes. Here's a breakdown:

### Data Preprocessing
- **Dataset**: Loads `drug200.csv` using Pandas.
- **Features**: Uses `Age`, `Sex`, `BP` (blood pressure), `Cholesterol`, and `Na_to_K` (sodium-to-potassium ratio) as input features, with `Drug` as the target variable.
- **Encoding**: Converts categorical variables (`Sex`: F/M, `BP`: LOW/NORMAL/HIGH, `Cholesterol`: NORMAL/HIGH) to numerical values using Scikit-learn's `LabelEncoder`.
- **Train-Test Split**: Splits data into 70% training and 30% testing sets with a fixed random state for reproducibility.

### Model Training
- **Algorithm**: Scikit-learn's `DecisionTreeClassifier` with `entropy` criterion and `max_depth=4` to prevent overfitting.
- **Training**: Fits the model on the training data using the preprocessed features.

### Evaluation
- **Predictions**: Generates predictions for the test set.
- **Accuracy**: Computes the accuracy score using `metrics.accuracy_score`, comparing predicted and actual drug labels (e.g., accuracy ≈ 0.983).

### Visualization
- Exports the decision tree to a `.dot` file using `export_graphviz`.
- Converts the `.dot` file to a PNG image (`tree.png`) using Graphviz's `dot` command, showing the tree's structure with feature names and decision paths.

## Dataset
The dataset (`drug200.csv`) contains patient data with the following columns:
- `Age`: Patient age (numeric).
- `Sex`: Patient sex (F/M).
- `BP`: Blood pressure (LOW/NORMAL/HIGH).
- `Cholesterol`: Cholesterol level (NORMAL/HIGH).
- `Na_to_K`: Sodium-to-potassium ratio in blood (numeric).
- `Drug`: Target drug (e.g., drugA, drugB, drugC, drugX, drugY).

Source: [IBM Developer Skills Network](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv).

Dive into the **Drug Classification with Decision Trees** project and explore machine learning! If you have questions, suggestions, or issues, please open an issue on GitHub. Happy learning!
