import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import nbformat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn import cluster
import warnings
warnings.filterwarnings('ignore')

# Set up the dashboard structure with multiple pages
st.set_page_config(page_title="Project Dashboard", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(r'datasets/nyc_taxi_trip_duration_1.csv', encoding='utf-8')


taxi_data = load_data()


# Function to create different pages
def goal_and_overview():
    st.title("Goal and Overview")

    # Display the uploaded image (replace with path to image in your environment)
    img = Image.open(r"datasets/just_image_nyc.png")
    st.image(img, caption="Taxi Duration Prediction")

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
        test dataset: https://drive.google.com/file/d/1C2N2mfONpCVrH95xHJjMcueXvvh_-XYN/view?usp=sharing
        file with holiday dates: https://lms-cdn.skillfactory.ru/assets/courseware/v1/33bd8d5f6f2ba8d00e2ce66ed0a9f510/asset-v1:SkillFactory+DSPR-2.0+14JULY2021+type@asset+block/holiday_data.csv
        OSRM geographic data file for the training set: https://drive.google.com/file/d/1ecWjor7Tn3HP7LEAm5a0B_wrIfdcVGwR/view?usp=sharing
        file with OSRM geographic data for the test set: https://drive.google.com/file/d/1wCoS-yOaKFhd1h7gZ84KL9UwpSvtDoIA/view?usp=sharing
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


def regression_and_conclusion():
    st.title("Regression Plans and Conclusion")
    st.header("Regression Plans")
    st.write("""
    The project will involve multiple regression analysis to predict the taxi trip duration. Features such as geo-coordinates, trip distance, and
    time of day will be utilized in building the regression model.
    
    Further I will be using Ensemble learning using
    
        Linear Regression
        2nd Degree Linear Regression
        Decision Tree
        Random Forest
        Gradient Boosting 
    """)
    st.header("Conclusion")
    st.write("The project aims to deliver a reliable model that helps predict taxi trip durations accurately.")


# Dictionary to map page names to functions
pages = {
    "Goal and Overview": goal_and_overview,
    "Audience & Narrative and Project Structure": audience_and_structure,
    "Dataset Description": dataset_description,
    "IDA: Missing Values, Imputation, Data Types": ida_page,
    "EDA": eda_page,
    "Regression Plans and Conclusion": regression_and_conclusion,
}

# Sidebar for page navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page
pages[selection]()
