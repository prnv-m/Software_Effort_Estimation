# Software_Effort_Estimation
Estimating effort using deep learning techniques on the COCOMO dataset
This project demonstrates how to estimate software development effort using a deep learning model trained on a historical dataset (likely based on COCOMO81). The model uses various project features to predict the required effort and evaluates its performance using standard regression metrics.


## Dependencies
`pip install pandas numpy matplotlib seaborn scikit-learn tensorflow`

## Dataset
To enhance model learning and performance, several preprocessing and feature engineering steps were applied on the COCOMO 81 dataset, consisting of 63 software projects with effort multipliers and actual development effort values.
- **Product Attributes**: `rely`, `data`, `cplx`, `time`, `stor`, `virt`, `turn`
- **Personnel Attributes**: `acap`, `aexp`, `pcap`, `vexp`, `lexp`
- **Project Attributes**: `modp`, `tool`, `sced`
- **Scale Metric**: `loc` (Lines of Code)
- **Target Variable**: `actual` (Effort in person-months)
- 

## Exploratory Data Analysis
- **Correlation heatmaps** were used to understand the relationships between numerical features.
- **Box plots** helped visualize the distribution of effort multipliers.
- **Scatter plots** showed a non-linear correlation between `loc` and `actual`.
- The **target variable (`actual`)** was found to be **right-skewed** and required normalization.

##  Scaling
All input features were standardized using StandardScaler to improve neural network convergence:
```Python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Model Architecture
Implemented a deep learning regression model using TensorFlow/Keras to estimate software development effort from COCOMO 81 attributes.

The model was designed to capture non-linear relationships between project attributes and the actual effort required.

```plaintext
Input Layer (17 features)
        ↓
Dense Layer (64 units, ReLU activation)
        ↓
Dropout (rate=0.2)
        ↓
Dense Layer (32 units, ReLU activation)
        ↓
Dropout (rate=0.2)
        ↓
Dense Layer (16 units, ReLU activation)
        ↓
Output Layer (1 unit, Linear activation)

```

## Notes

The model uses early stopping to prevent overfitting.

The feature importance is derived from the weights of the first dense layer.

Performance can vary based on data quality, size, and feature engineering.
