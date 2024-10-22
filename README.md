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

bash


streamlit run app.py
