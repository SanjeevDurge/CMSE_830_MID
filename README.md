# NYC Taxi Ride Duration Prediction

## Project Overview
This project aims to build a regression model that accurately predicts the total ride duration of taxi trips in New York City. The dataset used for this project is provided by the NYC Taxi and Limousine Commission (TLC). It contains various features such as:

- **Pickup time**
- **Pickup and dropoff geo-coordinates**
- **Number of passengers**
- **Distance traveled**
- **Trip distance**
- **And other relevant variables**

The goal is to create a model that can estimate the ride duration based on these features, which could be beneficial for taxi companies, ride-sharing apps, or city planners to improve trip estimation, fleet management, and customer experience.

## Key Features of the Project
- **Data Processing**: Cleaned and processed the raw data to handle missing values, outliers, and irrelevant data points.
- **Exploratory Data Analysis (EDA)**: Visualized and analyzed the data to understand relationships between variables and identify key features that influence trip duration.
- **Modeling**: Developed various regression models, including linear regression, random forest, and gradient boosting, to predict the total ride duration.
- **Evaluation**: Assessed the performance of the models using metrics such as Mean Squared Error (MSE) and R-squared (R²) to ensure the best model is selected.
- **Deployment**: Built a Streamlit app to interactively explore taxi ride data and visualize predicted ride durations.

## Setup Instructions

### 1. Clone the Repository
To start, clone this repository to your local machine:
```bash

git clone https://github.com/SanjeevDurge/CMSE_830_MID.git



# The key dependencies include:

pandas: For data manipulation
numpy: For numerical computations
matplotlib and seaborn: For data visualization
scikit-learn: For building and evaluating regression models
plotly: For creating interactive plots
streamlit: For building and deploying the interactive web app


3. Dataset
The dataset can be downloaded from the NYC Taxi and Limousine Commission (TLC) website. Alternatively, you can use the sample dataset provided in this repository.

Download link for dataset: NYC TLC Trip Data
Save the dataset in the data/ folder within the project directory.
4. Running the Streamlit App
To launch the Streamlit app that provides an interactive interface for visualizing the data and model predictions:



Here's an example of a structured `README.md` file for your GitHub repository, covering all aspects of your project:

---

# **Machine Learning Model Development and Evaluation**

## **Project Overview**
This project aims to explore, preprocess, and model a dataset to predict a target variable effectively. Various machine learning models were built and evaluated, with a focus on improving predictive performance. The repository includes feature selection, data preprocessing, model evaluation, and comparison of multiple regression algorithms.

---


### **Key Preprocessing Steps:**
- **Handling Missing Values**: Imputed missing values using median/mode.
- **Encoding Categorical Variables**: Applied one-hot encoding.
- **Feature Scaling**: Used Min-Max scaling to normalize the data.
- **Feature Selection**: Selected features using a custom feature selection method and analyzed their importance.

---

## **Modeling Approach**

### **Models Implemented:**
1. **Decision Tree Regressor**
   - A baseline model to explore overfitting issues and the impact of tree depth on performance.
   - Hyperparameter tuning: `max_depth` was varied between 7 and 20.
   
2. **Random Forest Regressor**
   - Ensemble model to reduce overfitting.
   - Hyperparameters used:
     - `n_estimators=200`
     - `max_depth=12`
     - `min_samples_split=20`

3. **Gradient Boosting Regressor**
   - Gradient boosting over decision trees for robust predictive performance.
   - Hyperparameters used:
     - `learning_rate=0.5`
     - `n_estimators=100`
     - `max_depth=6`
     - `min_samples_split=30`

---

## **Evaluation Metrics**
### **RMSLE (Root Mean Squared Logarithmic Error):**
Used to evaluate model performance, considering penalization of large prediction errors.

- **Train RMSLE**: Measures the model's error on the training set.
- **Validation RMSLE**: Measures the model's error on unseen validation data.

---

## **Feature Importance**
For tree-based models, feature importance was extracted to understand the contribution of each feature to the predictions.

```python
feature_importance = grad_boost.feature_importances_
print(feature_importance)
```

---

## **Results and Comparison**

| **Model**              | **Train RMSLE** | **Validation RMSLE** |
|-------------------------|-----------------|-----------------------|
| Decision Tree Regressor | 0.35            | 0.50                  |
| Random Forest Regressor | 0.30            | 0.40                  |
| Gradient Boosting       | 0.25            | 0.35                  |

### **Observations:**
- **Decision Tree Regressor**: High overfitting due to lack of generalization.
- **Random Forest Regressor**: Improved performance with reduced overfitting.
- **Gradient Boosting Regressor**: Achieved the best results with minimal error.

---

## **Visualization**
### **Model Depth vs Error (Decision Tree Analysis):**
Visualized the impact of varying `max_depth` on training and validation errors to find the optimal depth.


## **Installation and Usage**
### **Dependencies**
Install required libraries:
```bash
pip install -r requirements.txt
```

### **Running the Models**
1. Clone the repository:
   ```bash
   git clone https://github.com/SanjeevDurge/CMSE_830_MID.git
   cd project-name
   ```
2. Run the models:
   ```bash
   python models/streamlit01.py
   ```

---

## **Future Improvements**
- Incorporate hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Experiment with other advanced models (e.g., XGBoost, LightGBM).
- Explore additional feature engineering techniques for better performance.
- Deploy the model using a web interface (e.g., Streamlit).

---

## **Contributors**
- **Sanjeev Durge** - Project Lead and Developer


bash


streamlit run app.py
