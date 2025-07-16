# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Step 2: Load Dataset
df = pd.read_csv("insurance.csv")
print(df.head())

# Step 3: Preprocess the Data
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])       # male=1, female=0
df['smoker'] = le.fit_transform(df['smoker']) # yes=1, no=0
df['region'] = le.fit_transform(df['region']) # Encode regions numerically

# Step 4: Feature & Target Separation
X = df.drop('charges', axis=1)
y = df['charges']

# Step 5: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict & Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.show()
