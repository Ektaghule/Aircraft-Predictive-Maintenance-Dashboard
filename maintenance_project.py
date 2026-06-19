import pandas as pd
import numpy as np

# create data
n = 1500   # ✅ increased dataset size

df = pd.DataFrame({
    "Flight_Hours": np.random.randint(500,5000,n),
    "Temperature": np.random.randint(60,120,n),
    "Vibration": np.random.uniform(0.1,1.0,n),
    "Component_Age": np.random.randint(6,60,n)
})


# preprocessing
df.fillna(df.median(numeric_only=True), inplace=True)

print(df.head())
