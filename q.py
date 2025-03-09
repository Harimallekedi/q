import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Load data
def load_data():
    file_path = "university_student_dashboard_data.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Data Validation
def validate_data(df):
    # Check for missing values
    if df.isnull().any().any():
        st.error("Data contains missing values. Please check the dataset.")
        return False
    # Check for duplicate rows
    if df.duplicated().any():
        st.error("Data contains duplicate rows. Please check the dataset.")
        return False
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if (df[numeric_cols] < 0).any().any():
        st.error("Data contains negative values in numeric columns. Please check the dataset.")
        return False
    return True

if not validate_data(df):
    st.stop()

# Anomaly Detection
def detect_anomalies(df):
    # Use Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.05)
    df['anomaly'] = iso_forest.fit_predict(df.select_dtypes(include=[np.number]))
    anomalies = df[df['anomaly'] == -1]
    return anomalies

anomalies = detect_anomalies(df)
if not anomalies.empty:
    st.warning("Anomalies detected in the data. Please review the following records:")
    st.write(anomalies)

# Streamlit app
st.title("University Admissions and Student Satisfaction Dashboard")

# User Authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Filters
years = df['Year'].unique()
selected_year = st.selectbox("Select Year", years)

df_filtered = df[df['Year'] == selected_year]

# Applications, Admissions, and Enrollments over time
st.subheader("Applications, Admissions, and Enrollments")
fig, ax = plt.subplots()
ax.plot(df_filtered['Term'], df_filtered['Applications'], label='Applications', marker='o')
ax.plot(df_filtered['Term'], df_filtered['Admitted'], label='Admitted', marker='o')
ax.plot(df_filtered['Term'], df_filtered['Enrolled'], label='Enrolled', marker='o')
ax.set_ylabel("Count")
ax.set_title("Admissions Trends")
ax.legend()
st.pyplot(fig)

# Retention and Satisfaction Trends
st.subheader("Retention Rate and Student Satisfaction")
fig, ax = plt.subplots()
ax.plot(df_filtered['Term'], df_filtered['Retention Rate (%)'], label='Retention Rate', marker='o')
ax.plot(df_filtered['Term'], df_filtered['Student Satisfaction (%)'], label='Satisfaction', marker='o')
ax.set_ylabel("Percentage")
ax.set_title("Retention and Satisfaction Trends")
ax.legend()
st.pyplot(fig)

# Enrollment Breakdown by Department
st.subheader("Enrollment Breakdown by Department")
departments = ['Engineering Enrolled', 'Business Enrolled', 'Arts Enrolled', 'Science Enrolled']
department_counts = df_filtered[departments].sum()
fig, ax = plt.subplots()
ax.bar(departments, department_counts)
ax.set_ylabel("Number of Students")
ax.set_title("Department-wise Enrollment")
st.pyplot(fig)

# Insights and Summary
st.subheader("Key Insights")
insights = """
- **Admissions Trends**: Applications and enrollments fluctuate across terms, with a notable difference between Spring and Fall.
- **Retention & Satisfaction**: Retention and satisfaction trends indicate overall stability but show variations by year.
- **Department Analysis**: Engineering and Business typically see the highest enrollments, while Arts and Science have steadier numbers.
- **Spring vs. Fall**: Comparing across terms reveals key differences in admission rates and student preferences.
"""
st.markdown(insights)
