# Regression Model Comparison

## Overview
This repository contains a Python script to evaluate the performance of various regression models using a sample dataset. The goal is to identify the model that provides the best R² score for the given data.

The script compares the performance of multiple regression models on a dataset, with the target variable set to *Target4*. The models evaluated include:

- *Linear Regression*
- *Random Forest Regressor*
- *Gradient Boosting Regressor*
- *Lasso Regression*
- *Ridge Regression*
- *Decision Tree Regressor*

The R² score is used as the performance metric, with higher values indicating better model performance.

## Results
The performance of the models is summarized below:

| Model                  | R² Score  |
|------------------------|------------|
| Linear Regression      | 0.2511     |
| Random Forest          | 0.2010     |
| Gradient Boosting      | 0.1892     |
| Lasso Regression       | 0.1535     |
| Ridge Regression       | 0.2511 (Best) |
| Decision Tree          | -0.5944 (Worst) |

### Best Model
- *Ridge Regression* with an R² score of *0.2511*.

### Worst Model
- *Decision Tree Regressor* with an R² score of *-0.5944*.

## Output
The script outputs:

1. A numerical comparison of R² scores for all models.
2. Identification of the best and worst performing models.
3. A bar chart visualizing the R² scores of all models.

## Prerequisites
- Python 3.7+
- Required libraries:
  - pandas
  - matplotlib
  - scikit-learn

Install the dependencies using pip:

bash
pip install pandas matplotlib scikit-learn


## Usage
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/model-performance-comparison.git
   
2. Navigate to the project directory:
   bash
   cd model-performance-comparison
   
3. Replace the placeholder dataset path (*Fuel_cell_performance_data-Full.csv*) with your dataset file path.
4. Run the script:
   bash
   python script_name.py
   
5. View the results in the console and the generated bar chart for model comparison.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
