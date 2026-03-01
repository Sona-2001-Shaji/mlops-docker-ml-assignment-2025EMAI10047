import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load the Palmer Penguins dataset
data = sns.load_dataset("penguins")
data = data.dropna()

# Encode categorical columns
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
data['island'] = le.fit_transform(data['island'])
data['sex'] = le.fit_transform(data['sex'])

# Split features and target
X = data.drop('species', axis=1)
y = data['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluation metrics
precision = precision_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

print("Model Precision:", precision)
print("AUC Score:", auc)