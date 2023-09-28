# OIBSIP_task_3
# car_price_prediction
Car Price Prediction
This Python program uses machine learning to predict car prices based on various features. It includes data preprocessing, feature selection, and linear regression modeling.

Requirements
Python 3.x
Libraries: numpy, pandas, seaborn, scikit-learn, statsmodels
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/anshul29292929/car_price_prediction.git
Install the required libraries:

bash
pip install numpy pandas seaborn scikit-learn statsmodels

bash
python car_price_prediction.pym.

Program Workflow

Data is loaded from the "CarPrice_Assignment.csv" file.
Data preprocessing is performed, including one-hot encoding of categorical features.
Features with high multicollinearity are removed iteratively.
The dataset is split into training and testing sets.
A linear regression model is built and trained on the training data.
Predictions are made on the testing data.
Evaluation metrics (MAE, MSE, RMSE, R-squared) are displayed.

Contributors

Your Name (@anshul29292929)

License
This project is licensed under the MIT License. See the LICENSE file for details.
