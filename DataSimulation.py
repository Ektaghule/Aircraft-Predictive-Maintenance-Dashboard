import pandas as pd
import numpy as np

n = 1500   # increased dataset size

data = {
    "Component_ID": [f"C{i}" for i in range(1, n+1)],
    "Flight_Hours": np.random.randint(500, 5000, n),
    "Temperature": np.random.randint(60, 120, n),
    "Vibration": np.random.uniform(0.1, 1.0, n),
    "Component_Age": np.random.randint(6, 60, n),
    "Maintenance_Cost": np.random.randint(1000, 50000, n)
}

df = pd.DataFrame(data)

# Failure logic
df["Failure"] = np.where(
    (
        (df["Flight_Hours"] > 3000) & 
        (df["Temperature"] > 90)
    ) |
    (
        (df["Vibration"] > 0.7) & 
        (df["Component_Age"] > 40)
    ),
    1, 0
)

# Add 10% noise (flip some labels)
noise = np.random.choice([0,1], size=n, p=[0.9,0.1])
df["Failure"] = df["Failure"] ^ noise

# Check balance
print("\nClass Balance:")
print(df["Failure"].value_counts())

# Save dataset
df.to_csv("aircraft_maintenance_data.csv", index=False)
