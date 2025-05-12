import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("ppa_dataset.csv")

# Define features (X) and target (Y)
X = df[["Clock (GHz)", "Pipeline Depth"]]
Y = df[["Power (mW)", "Frequency (ns)", "Area (µm²)"]]

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a simple regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict PPA for a new design config
new_design = [[1.8, 6]]  # Example: 1.8 GHz clock, 6-stage pipeline
predicted_ppa = model.predict(new_design)

print(f"Predicted PPA: Power = {predicted_ppa[0][0]:.2f} mW, Freq = {predicted_ppa[0][1]:.2f} ns, Area = {predicted_ppa[0][2]:.2f} µm²")
