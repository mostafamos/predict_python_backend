import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Custom estimator to capture detailed output from scikit-learn
class LogEstimator:
    def fit(self, X, y):
        self.estimator_ = LogisticRegression().fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator_.predict(X)
    
    def score(self, X, y):
        return self.estimator_.score(X, y)

# Step 1: Create the training dataset
data = {
    'error': [
        'cant insert record', 
        'network failure', 
        'cant insert record', 
        'network failure'
    ],
    'department': [
        'data', 
        'infra', 
        'data', 
        'infra'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Define the features (X) and the target (y)
X = df['error']
y = df['department']

# Step 3: Train the model
# Create a pipeline with TfidfVectorizer and custom LogEstimator
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('log', LogEstimator())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 4: Make predictions
# Define the new error
new_error = ['cant connect to db']

# Predict the department for the new error
predicted_department = model.predict(new_error)
print(f"The predicted department for the error 'cant connect to db' is: {predicted_department[0]}")

# Evaluate the model
scores = cross_val_score(model, X, y, cv=3)
