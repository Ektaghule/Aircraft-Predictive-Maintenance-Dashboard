# ==============================
# Predictive Maintenance EDA
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# STEP 1: Create Simulated Data
# ------------------------------

n = 1500   # ✅ increased dataset

df = pd.DataFrame({
    "Flight_Hours": np.random.randint(500,5000,n),
    "Temperature": np.random.randint(60,120,n),
    "Vibration": np.random.uniform(0.1,1.0,n),
    "Component_Age": np.random.randint(6,60,n)
})

# Failure Logic
df["Failure"] = np.where(
    (df["Flight_Hours"]>3000) &
    (df["Temperature"]>90) &
    (df["Vibration"]>0.6) &
    (df["Component_Age"]>36),
    1,0
)

# Add small noise
noise = np.random.choice([0,1], size=n, p=[0.9,0.1])
df["Failure"] = df["Failure"] ^ noise

# ------------------------------
# STEP 2: Basic Exploration
# ------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary Stats:")
print(df.describe())


# ------------------------------
# STEP 3: Visual EDA
# ------------------------------

sns.countplot(x="Failure", data=df)
plt.title("Failure Count")
plt.show()

sns.histplot(df["Flight_Hours"], bins=30)  # more bins for larger data
plt.title("Flight Hours Distribution")
plt.show()

sns.boxplot(x="Failure", y="Flight_Hours", data=df)
plt.show()

sns.boxplot(x="Failure", y="Temperature", data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# ==============================
# STEP 4: Feature Selection
# ==============================

X = df[["Flight_Hours","Temperature","Vibration","Component_Age"]]
y = df["Failure"]


# ==============================
# STEP 5: Train-Test Split
# ==============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# STEP 6: Build ML Model
# ==============================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

model = RandomForestClassifier(
    n_estimators=150,   # slightly stronger for bigger data
    class_weight="balanced",
    random_state=42
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


# ==============================
# STEP 7: Evaluation
# ==============================

print("\n🎯 Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred,zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))


# ==============================
# STEP 8: EXPORT FOR POWER BI
# ==============================

df["Predicted_Failure"] = model.predict(X)

df.to_csv("maintenance_powerbi.csv",index=False)

print("\n✅ File exported for Power BI!")
