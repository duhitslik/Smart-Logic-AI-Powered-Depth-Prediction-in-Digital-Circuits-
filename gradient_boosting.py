import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Define features and target variable
features = ["num_inputs", "num_outputs", "fan_in", "fan_out", "total_fan_io", "gate_density", "complexity_score"]
target = "logic depth"  # Adjust based on your dataset

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = gb_model.predict(X_test_scaled)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.5f}")
print(f"R² Score: {r2:.5f}")
plt.figure(figsize=(10, 5))
# Scatter plot of Actual vs Predicted values
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
plt.xlabel("Actual Combinational Depth")
plt.ylabel("Predicted Combinational Depth")
plt.title("Gradient Boosting - Predicted vs Actual")
plt.legend()
plt.show()

# Get feature importance from trained Gradient Boosting model
feature_importance = gb_model.feature_importances_

# Feature names (Ensure they match your dataset features)
feature_names = ["num_inputs", "num_outputs", "fan_in", "fan_out", "total_fan_io", "gate_density", "complexity_score"]

# Sort features by importance
sorted_idx = np.argsort(feature_importance)[::-1]  # Descending order

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Gradient Boosting Model")
plt.tight_layout()
# Show plot
plt.show()









