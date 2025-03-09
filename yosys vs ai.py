# Import necessary libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Upload dataset file
uploaded = files.upload()  # Upload `logical_depth_dataset.csv`

# Load dataset
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Define features and target variable
features = ["num_inputs", "num_outputs", "fan_in", "fan_out", "total_fan_io", "gate_density", "complexity_score"]
target = "logic depth"

X = df[features]
y = df[target]

# Split data into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train an optimized Gradient Boosting Model
optimized_gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
optimized_gb_model.fit(X_train_scaled, y_train)  # Train once

# ✅ Measure execution time for **inference only**
start_time_ai = time.time()
y_pred = optimized_gb_model.predict(X_test_scaled)  # AI prediction
end_time_ai = time.time()

execution_time_ai = end_time_ai - start_time_ai  # AI model execution time

# ✅ Measure Yosys Execution Time
verilog_code = f"""
module logical_depth_model(
    input [{df['num_inputs'].max()}:0] num_inputs,
    input [{df['num_outputs'].max()}:0] num_outputs,
    input [{df['fan_in'].max()}:0] fan_in,
    input [{df['fan_out'].max()}:0] fan_out,
    output [{df['logic depth'].max()}:0] logic_depth
);
assign logic_depth = num_inputs + num_outputs + fan_in + fan_out;
endmodule
"""

# Save Verilog file
verilog_file = "/content/logical_depth_model.v"
with open(verilog_file, "w") as f:
    f.write(verilog_code)

print("Verilog file generated successfully!")

# Run Yosys synthesis and measure execution time
yosys_command = f'yosys -p "read_verilog {verilog_file}; synth; stat"'
start_time_yosys = time.time()
!{yosys_command}
end_time_yosys = time.time()

execution_time_yosys = end_time_yosys - start_time_yosys  # Yosys execution time

# ✅ Print execution times
print(f"\nExecution Time (Optimized AI Model): {execution_time_ai:.5f} seconds")
print(f"Execution Time (Yosys): {execution_time_yosys:.5f} seconds")

# ✅ Compare AI Model vs. Yosys Execution Time
execution_times = {
    "Optimized AI Model": execution_time_ai,
    "Yosys Synthesis": execution_time_yosys
}

# ✅ Plot execution time comparison
plt.figure(figsize=(8, 5))
plt.bar(execution_times.keys(), execution_times.values(), color=["blue", "green"])
plt.ylabel("Execution Time (seconds)")
plt.title("Optimized AI Model vs Yosys Execution Time")
plt.ylim(0, max(execution_times.values()) * 1.2)

# Annotate values on top of bars
for i, (key, value) in enumerate(execution_times.items()):
    plt.text(i, value + 0.01, f"{value:.5f}s", ha="center", fontsize=12)

# Show the optimized plot
plt.show()
