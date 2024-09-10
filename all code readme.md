Step 1: Import Necessary Libraries

- Import the required libraries from scikit-learn:
    - load_iris for loading the Iris dataset
    - train_test_split for splitting the dataset into training and testing sets

Code:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


Step 2: Load the Iris Dataset

- Load the Iris dataset using load_iris
- Store the feature data in X and target labels in y

Code:

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels


Step 3: Split the Dataset into Training and Testing Sets

- Split the dataset into training (80%) and testing (20%) sets using train_test_split
- Set the random state to 42 for reproducibility

Code:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Step 4: Print the Number of Samples in Both Training and Testing Sets

- Print the number of samples in the training and testing sets

Code:

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")



 second code explanation




 Here's a step-by-step README file for your GitHub repository:

Iris Dataset Exploration

Step 1: Import Necessary Libraries

- Import the required libraries:
    - pandas for data manipulation and analysis
    - load_iris from scikit-learn for loading the Iris dataset

Code:

import pandas as pd
from sklearn.datasets import load_iris


Step 2: Load the Iris Dataset

- Load the Iris dataset using load_iris
- Convert the dataset into a Pandas DataFrame iris_df with feature names as columns

Code:

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


Step 3: Display the First Five Rows

- Print the first five rows of the dataset using head()

Code:

print("First five rows of the dataset:")
print(iris_df.head())


Step 4: Display the Shape of the Dataset

- Print the shape of the dataset (number of rows and columns) using shape

Code:

print("\nShape of the dataset:")
print(iris_df.shape)


Step 5: Display Summary Statistics

- Print summary statistics (mean, std, min, 25%, 50%, 75%, max) for each feature using describe()

Code:

print("\nSummary statistics of the dataset:")
print(iris_df.describe())


third code explanation




