import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Dataset
df = pd.read_csv("logical_depth_dataset.csv")

# Exploratory Data Analysis - Display the first few rows
print("Dataset Preview:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:", df.isnull().sum().sum())

# Define Features & Target
features = ["num_inputs", "num_outputs", "fan_in", "fan_out", "total_fan_io", "gate_density", "complexity_score"]
target = "logic depth"

X = df[features].values
y = df[target].values

# Feature Engineering: Add polynomial features and interaction terms
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"Original features: {X.shape[1]}, After polynomial expansion: {X_poly.shape[1]}")

# Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Split Dataset (Train: 70%, Validation: 15%, Test: 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch Tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

# FIXED: Create a more meaningful graph structure using kNN
def create_knn_graph(x, k=10):
    """Create a k-nearest neighbors graph from feature vectors"""
    # Compute pairwise distances
    n = x.size(0)
    dist = torch.cdist(x, x)

    # For each node, find k nearest neighbors
    _, indices = torch.topk(dist, k=k+1, largest=False)

    # Create edge index (exclude self-loops)
    rows = torch.arange(n, device=device).repeat_interleave(k)
    cols = indices[:, 1:].reshape(-1)  # Using reshape instead of view
    edge_index = torch.stack([rows, cols], dim=0)

    return edge_index

# Create meaningful edge indices using k-NN
k = 8  # Number of nearest neighbors
edge_index_train = create_knn_graph(X_train, k=k)
edge_index_val = create_knn_graph(X_val, k=k)
edge_index_test = create_knn_graph(X_test, k=k)

# Define Improved GNN Model
class ImprovedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ImprovedGNN, self).__init__()

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # GNN layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=2, dropout=0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, dropout=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index):
        # Initial projection
        x = self.input_proj(x)
        x = self.bn_input(x)
        x = torch.relu(x)

        # GNN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)

        # Output projection
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()

# Hyperparameters
hidden_dim = 128
learning_rate = 0.001
weight_decay = 1e-4
epochs = 500

# Model Initialization
model = ImprovedGNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers

# Print model architecture
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training Function with Early Stopping
def train_model(model, X_train, y_train, edge_index_train, X_val, y_val, edge_index_val, epochs=500):
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train, edge_index_train)
        loss = criterion(y_pred, y_train)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val, edge_index_val)
            val_loss = criterion(y_pred_val, y_val)

            # Calculate R² for validation set
            val_r2 = r2_score(y_val.cpu().numpy(), y_pred_val.cpu().numpy())

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_r2'].append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch}: New best model saved! Val Loss: {val_loss:.5f}, Val R²: {val_r2:.5f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {loss.item():.5f}, Val Loss: {val_loss.item():.5f}, Val R²: {val_r2:.5f}")

    return history

# Train the Model
history = train_model(model, X_train, y_train, edge_index_train, X_val, y_val, edge_index_val, epochs=epochs)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history['val_r2'], label='Validation R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.axhline(y=0.6, color='r', linestyle='--', label='Target R² = 0.6')
plt.legend()
plt.title('Validation R² Score')
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Model Evaluation on Test Set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test, edge_index_test)
    test_mse = mean_squared_error(y_test.cpu().numpy(), y_pred_test.cpu().numpy())
    test_r2 = r2_score(y_test.cpu().numpy(), y_pred_test.cpu().numpy())
    print(f"Test MSE: {test_mse:.5f}, R²: {test_r2:.5f}")

# Visual Analysis
plt.figure(figsize=(10, 8))

# Predictions vs Actual plot
plt.subplot(2, 2, 1)
sns.scatterplot(x=y_test.cpu().numpy(), y=y_pred_test.cpu().numpy())
plt.plot([min(y_test.cpu().numpy()), max(y_test.cpu().numpy())],
         [min(y_test.cpu().numpy()), max(y_test.cpu().numpy())],
         'r--')
plt.xlabel("Actual Logical Depth")
plt.ylabel("Predicted Logical Depth")
plt.title(f"GNN Predictions vs Actual (Test Set)\nR²: {test_r2:.3f}")

# Residuals plot
plt.subplot(2, 2, 2)
residuals = y_test.cpu().numpy() - y_pred_test.cpu().numpy()
sns.scatterplot(x=y_test.cpu().numpy(), y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Logical Depth")
plt.ylabel("Residuals")
plt.title("Residuals Plot")

# Distribution of residuals
plt.subplot(2, 2, 3)
sns.histplot(residuals, kde=True)
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")

# Relative error analysis
rel_errors = np.abs(y_test.cpu().numpy() - y_pred_test.cpu().numpy()) / (y_test.cpu().numpy() + 1e-10)
plt.subplot(2, 2, 4)
sns.scatterplot(x=y_test.cpu().numpy(), y=rel_errors)
plt.xlabel("Actual Logical Depth")
plt.ylabel("Relative Error")
plt.title(f"Relative Error Analysis\nMean: {np.mean(rel_errors):.3f}")

plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.show()

# Feature Importance Analysis (Optional)
try:
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    lr_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())

    # Get feature names after polynomial expansion
    try:
        feature_names = poly.get_feature_names_out(features)
    except:
        feature_names = poly.get_feature_names(features)

    # Plot top 15 most important features
    coeffs = np.abs(lr_model.coef_)
    idx = np.argsort(coeffs)[-15:]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[idx], coeffs[idx])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
except Exception as e:
    print(f"Skipping feature importance analysis due to: {e}")

# Traditional ML Comparison
from sklearn.ensemble import GradientBoostingRegressor
print("\nTraining traditional ML model for comparison...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
gb_preds = gb_model.predict(X_test.cpu().numpy())
gb_r2 = r2_score(y_test.cpu().numpy(), gb_preds)
gb_mse = mean_squared_error(y_test.cpu().numpy(), gb_preds)
print(f"Gradient Boosting - MSE: {gb_mse:.5f}, R²: {gb_r2:.5f}")

# Ensemble the predictions (GNN + GB)
ensemble_preds = (y_pred_test.cpu().numpy() + gb_preds) / 2
ensemble_r2 = r2_score(y_test.cpu().numpy(), ensemble_preds)
ensemble_mse = mean_squared_error(y_test.cpu().numpy(), ensemble_preds)
print(f"Ensemble Model - MSE: {ensemble_mse:.5f}, R²: {ensemble_r2:.5f}")

# Compare all models
models = ['Original GNN', 'Improved GNN', 'Gradient Boosting', 'Ensemble']
r2_scores = [-0.42889, test_r2, gb_r2, ensemble_r2]
mse_scores = [117.91771, test_mse, gb_mse, ensemble_mse]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(models, r2_scores)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.ylim(-0.5, 1.0)

plt.subplot(1, 2, 2)
plt.bar(models, mse_scores)
plt.title('MSE Comparison')
plt.ylabel('MSE')
plt.yscale('log')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()