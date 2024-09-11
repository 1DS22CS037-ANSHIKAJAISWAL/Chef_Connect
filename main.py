import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Load and preprocess the dataset
def load_data(file_path):
    # Correct file path assignment
    file_path = r'C:\Users\EG\PycharmProjects\chef\data\pollen_grain.csv'

    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'ReportDateTime' to datetime format
    df['ReportDateTime'] = pd.to_datetime(df['ReportDateTime'])
    df['Month'] = df['ReportDateTime'].dt.month
    df['Day'] = df['ReportDateTime'].dt.day
    df['Hour'] = df['ReportDateTime'].dt.hour

    # Encode the 'PollenType' column using LabelEncoder
    le = LabelEncoder()
    df['PollenType'] = le.fit_transform(df['PollenType'])

    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    df[['AQI', 'PollenType', 'Month', 'Day', 'Hour']] = imputer.fit_transform(
        df[['AQI', 'PollenType', 'Month', 'Day', 'Hour']])

    # Handle missing values in target variable ('PollenCount')
    df['PollenCount'] = df['PollenCount'].fillna(df['PollenCount'].mean())

    return df


# Train the RandomForest model
def train_model(df):
    X = df[['AQI', 'PollenType', 'Month', 'Day', 'Hour']]
    y = df['PollenCount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model Evaluation:")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R2) Score: {r2}")

    return model


# Function to predict pollen concentration
def predict_pollen_concentration(model, aqi, pollen_type, month, day, hour):
    input_data = pd.DataFrame([[aqi, pollen_type, month, day, hour]],
                              columns=['AQI', 'PollenType', 'Month', 'Day', 'Hour'])
    return model.predict(input_data)[0]


# Function to check pollen concentration and trigger alerts
def check_alert(predicted_concentration, threshold=100):
    if predicted_concentration > threshold:
        return f"Alert! Pollen concentration is high: {predicted_concentration:.2f}. Take precautions."
    else:
        return f"Pollen concentration is safe: {predicted_concentration:.2f}."


# Visualization
def visualize_data(df):
    st.write("Pollen Concentration Over Time")

    num_rows = len(df)
    regions = ['North', 'South', 'East', 'West']
    # Create a list that exactly matches the number of rows in the DataFrame
    df['Region'] = regions * (num_rows // len(regions)) + regions[:num_rows % len(regions)]

    # Line plot for pollen concentration over time for different regions
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df['ReportDateTime'], y=df['PollenCount'], hue=df['Region'])
    plt.xlabel("Date")
    plt.ylabel("Pollen Concentration")
    plt.title("Pollen Concentration Over Time by Region")
    plt.xticks(rotation=45)
    st.pyplot(plt)


from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_tuning(X_train, y_train):
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                       n_iter=10, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
    random_search.fit(X_train, y_train)

    st.write(f"Best Hyperparameters: {random_search.best_params_}")
    return random_search.best_estimator_

from sklearn.model_selection import cross_val_score

def evaluate_with_cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    st.write(f"Cross-validation MAE: {-scores.mean():.4f}")

def check_alert_dynamic(predicted_concentration, user_threshold=100):
    if predicted_concentration > user_threshold:
        return f"Alert! Pollen concentration is high: {predicted_concentration:.2f}. Take precautions."
    else:
        return f"Pollen concentration is safe: {predicted_concentration:.2f}."

# Streamlit UI
def main():
    st.title("Pollen Concentration Prediction and Alert System")

    # Load the dataset
    df = load_data(r'C:\Users\EG\PycharmProjects\chef\data\pollen_grain.csv')

    # Train the model
    st.write("Training the model...")
    model = train_model(df)

    # User input for the prediction system
    st.write("Predict Pollen Concentration")
    aqi = st.number_input('Enter AQI:', min_value=0)
    pollen_type = st.number_input('Enter Pollen Type (encoded as a number):', min_value=0)
    month = st.number_input('Enter Month:', min_value=1, max_value=12)
    day = st.number_input('Enter Day:', min_value=1, max_value=31)
    hour = st.number_input('Enter Hour:', min_value=0, max_value=23)
    user_threshold = st.number_input('Enter Alert Threshold:', min_value=0)

    if st.button('Predict'):
        predicted_concentration = predict_pollen_concentration(model, aqi, pollen_type, month, day, hour)
        st.write(f"Predicted Pollen Concentration: {predicted_concentration:.2f}")
        alert_message = check_alert_dynamic(predicted_concentration, user_threshold)
        st.write(alert_message)

    # Visualization
    st.write("Visualizing Data")
    visualize_data(df)



if __name__ == '__main__':
    main()