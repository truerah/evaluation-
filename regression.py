import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Load the dataset
file_path = 'Fuel_cell_performance_data-Full.csv'
dataset = pd.read_csv(file_path)

# Display the first few rows to understand the structure
dataset.head(), dataset.columns

# Select the target 'Target4' and drop other target columns
X = dataset.drop(['Target1', 'Target2', 'Target3', 'Target4', 'Target5'], axis=1)
y = dataset['Target1']

# Split the dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define prediction models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "Decision Tree": DecisionTreeRegressor(),
   
}

# Run each model and evaluate performance
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Calculate R² score
    r2 = r2_score(y_test, y_pred)
    results[model_name] = r2

# Identify the best and worst models based on R² score
best_model_name = max(results, key=results.get)
best_r2_score = results[best_model_name]

worst_model_name = min(results, key=results.get)
worst_r2_score = results[worst_model_name]

# Display results in numerical form
print("Model Performance (R² Scores):")
for model_name, r2 in results.items():
    print(f"{model_name}: {r2:.4f}")

print(f"\nBest Model: {best_model_name} with R² Score: {best_r2_score:.4f}")
print(f"Worst Model: {worst_model_name} with R² Score: {worst_r2_score:.4f}")
