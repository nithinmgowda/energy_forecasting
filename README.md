# Renewable Energy Production Prediction

An advanced hybrid machine learning model that combines XGBoost and LSTM neural networks to predict renewable energy production using meteorological and temporal features.

## 🎯 Overview

This project implements a sophisticated energy prediction system that leverages both traditional gradient boosting and deep learning approaches to forecast renewable energy output. The hybrid model combines the interpretability of XGBoost with the temporal modeling capabilities of LSTM networks.

## 🚀 Features

- **Hybrid Architecture**: Combines XGBoost and bidirectional LSTM with attention mechanism
- **Advanced Feature Engineering**: Creates temporal, interaction, and rolling statistical features
- **Robust Data Processing**: Handles outliers using RobustScaler and quantile-based filtering
- **Enhanced Training**: Implements early stopping, learning rate scheduling, and gradient clipping
- **Comprehensive Evaluation**: Multiple metrics including R², MAE, RMSE, and MAPE

## 📊 Model Architecture

### XGBoost Component
- **Purpose**: Captures non-linear feature interactions and provides leaf node features
- **Parameters**: 200 estimators, max depth 8, learning rate 0.05
- **Features**: Regularization (L1/L2), early stopping, cross-validation

### LSTM Component
- **Architecture**: Bidirectional LSTM with attention mechanism
- **Layers**: 2-layer bidirectional LSTM + 1-layer LSTM + Multi-head attention
- **Features**: Dropout regularization, batch normalization, residual connections

### Fusion Network
- **Input**: LSTM features + XGBoost leaf features + XGBoost predictions
- **Architecture**: Multi-layer feedforward network with skip connections
- **Output**: Final energy production prediction

## 📁 Data Requirements

The model expects a CSV file named `Renewable.csv` with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `Time` | Timestamp | DateTime |
| `GHI` | Global Horizontal Irradiance | Float |
| `wind_speed` | Wind speed measurement | Float |
| `humidity` | Humidity percentage | Float |
| `Energy delta[Wh]` | Energy production target | Float |

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd renewable-energy-prediction

# Install required packages
pip install pandas numpy torch scikit-learn xgboost scipy
```

### Dependencies

```
pandas >= 1.3.0
numpy >= 1.21.0
torch >= 1.9.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
scipy >= 1.7.0
```

## 🔧 Usage

### Basic Usage

```python
# Ensure your data file is in the correct location
# Update the file path in the code: df = pd.read_csv('/Renewable.csv', ...)

# Run the prediction model
python renewable_energy_prediction.py
```

### Configuration

Key hyperparameters can be modified:

```python
# Sequence length for LSTM
n_past = 24  # Hours of historical data

# Model architecture
hidden_size = 256  # LSTM hidden units
num_heads = 8     # Attention heads

# Training parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 150
```

## 🧠 Feature Engineering

The model creates several engineered features:

### Temporal Features
- Hour of day
- Day of year
- Month

### Interaction Features
- `GHI_squared`: Squared global horizontal irradiance
- `wind_squared`: Squared wind speed
- `GHI_wind_interaction`: GHI × wind speed interaction
- `temp_humidity_ratio`: Temperature-humidity relationship

### Rolling Statistics (7-day window)
- `GHI_rolling_mean`: Moving average of GHI
- `GHI_rolling_std`: Rolling standard deviation of GHI
- `wind_rolling_mean`: Moving average of wind speed

## 📈 Model Performance

### Evaluation Metrics

The model provides comprehensive evaluation:

- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **Accuracy**: MAE-based accuracy percentage
- **Within 20%**: Percentage of predictions within 20% of actual values

### Expected Performance

Typical performance on renewable energy datasets:
- R² > 0.85
- MAPE < 15%
- Within 20% accuracy > 80%

## 🔍 Data Processing Pipeline

### 1. Data Loading and Cleaning
```python
# Load data and remove missing values
df = pd.read_csv('/Renewable.csv', usecols=[...])
df.dropna(inplace=True)
```

### 2. Feature Engineering
```python
# Create temporal and interaction features
df['hour'] = df['Time'].dt.hour
df['GHI_wind_interaction'] = df['GHI'] * df['wind_speed']
```

### 3. Outlier Removal
```python
# Remove extreme outliers (beyond 99.5th percentile)
target_q99 = df[target].quantile(0.995)
mask = (df[target] >= target_q01) & (df[target] <= target_q99)
```

### 4. Scaling and Normalization
```python
# Use RobustScaler for outlier resistance
feature_scaler = RobustScaler()
target_scaler = RobustScaler()
```

### 5. Sequence Creation
```python
# Create sequences for LSTM input
for i in range(n_past, len(scaled_features)):
    X_seq.append(scaled_features[i - n_past:i])
    y_seq.append(scaled_target[i])
```

## 🎛️ Advanced Features

### Custom Loss Function
```python
class HybridLoss(nn.Module):
    def __init__(self, mse_weight=0.7, mae_weight=0.3):
        # Combines MSE and MAE for robust training
```

### Attention Mechanism
```python
self.attention = nn.MultiheadAttention(
    hidden_size, num_heads=8, dropout=0.1, batch_first=True
)
```

### Learning Rate Scheduling
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

## 📊 Output Example

```
ENHANCED EVALUATION RESULTS
============================================================
Test samples: 1,234
MAE:                    45.23
RMSE:                   67.89
R²:                     0.8756
MAPE:                   12.34%
Accuracy (MAE-based):   87.66%
Within 20% accuracy:    89.12%
============================================================

Model Comparison:
XGBoost R²:     0.8234
Hybrid R²:      0.8756
Improvement:    6.3%
```

## ⚙️ Hardware Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: Minimum 8GB, 16GB+ recommended for large datasets
- **GPU**: CUDA-compatible GPU optional but recommended for faster training
- **Storage**: 2GB+ free space for model checkpoints and data

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `batch_size = 32`
   - Use CPU: `device = torch.device("cpu")`

2. **File Not Found Error**
   - Update CSV file path in the code
   - Ensure all required columns are present

3. **Poor Performance**
   - Check data quality and missing values
   - Adjust sequence length (`n_past`)
   - Modify feature engineering parameters

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

