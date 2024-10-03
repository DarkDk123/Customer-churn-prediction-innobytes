import streamlit as st
import pandas as pd
import joblib  # For loading the trained classification model
from serial_required import drop_missing_convert_dt
import plotly.express as px

# Load your trained model
MODEL_PATH = "./Churn_predictor_pipeline_joblib"
model = joblib.load(MODEL_PATH)

# Define the column to data type mapping
ColumnTypeMapping = {
    "gender": (str, ["Male", "Female"]),
    "SeniorCitizen": (str, ["Yes", "No"]),
    "Partner": (str, ["Yes", "No"]),
    "Dependents": (str, ["Yes", "No"]),
    "tenure": (int,),
    "PhoneService": (str, ["Yes", "No"]),
    "MultipleLines": (str, ["Yes", "No", "No phone service"]),
    "InternetService": (str, ["DSL", "Fiber optic", "No"]),
    "OnlineSecurity": (str, ["Yes", "No", "No internet service"]),
    "OnlineBackup": (str, ["Yes", "No", "No internet service"]),
    "DeviceProtection": (str, ["Yes", "No", "No internet service"]),
    "TechSupport": (str, ["Yes", "No", "No internet service"]),
    "StreamingTV": (str, ["Yes", "No", "No internet service"]),
    "StreamingMovies": (str, ["Yes", "No", "No internet service"]),
    "Contract": (str, ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": (str, ["Yes", "No"]),
    "PaymentMethod": (
        str,
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    ),
    "MonthlyCharges": (float,),
    "TotalCharges": (float,),
}


# Function to process user input from form
def process_user_input():
    user_input = {}
    for col, col_type in ColumnTypeMapping.items():
        if col_type[0] is str:
            user_input[col] = st.selectbox(f"{col}:", col_type[1])
        elif col_type[0] is int:
            user_input[col] = st.number_input(f"{col}:", min_value=1, format="%d")
        elif col_type[0] is float:
            user_input[col] = st.number_input(f"{col}:", min_value=0.0, format="%.2f")

    return pd.DataFrame([user_input])


# Function to process the uploaded CSV
def process_uploaded_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    for col, col_type in ColumnTypeMapping.items():
        assert (
            df[col].dtype is not col_type[0]
        ), f"{col} column must be of type {col_type[0].__name__}"

        if col not in df.columns:
            st.error(f"Missing column: {col}")

    return df


# Streamlit UI

st.title("Customer Churn Prediction 🤖")
st.text("Use the input form below or upload a CSV file for predictions..")

# Sidebar
st.sidebar.image(
    "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fraw.githubusercontent.com%2Famansingh9097%2Famansingh9097.github.io%2Fmaster%2Fassets%2Fimg%2Fuploads%2Fchurn-prediction%2Fcustomer-churn.jpg&f=1&nofb=1&ipt=499af1f89224da5be15093906abdbc6cd2fddf4830d7ff2ec4d9048516048048&ipo=images"
)

# Option for individual input form or CSV upload
st.sidebar.title("Select Input Method")
option = st.sidebar.radio("Input Method", ("Form Input", "Upload CSV"))

# Prediction section
st.sidebar.write("---")
st.sidebar.subheader("Required Data Schema...")
# Expander to display input schema
with st.sidebar.expander("Input Data Schema"):
    df = pd.DataFrame.from_dict(
        ColumnTypeMapping, orient="index", columns=["Data Type", "Allowed Values"]
    )
    df["Data Type"] = df["Data Type"].apply(lambda x: x.__name__)
    st.write(df)

if option == "Form Input":
    # Display the form for individual input
    st.header("Input the details below:")
    input_df = process_user_input()

    if st.button("Predict"):
        prediction = model.predict_proba(input_df)
        st.success(
            f"Prediction: {'Churn! ✅' if prediction[0][1] >= 0.5 else 'No Churn! ❌'}"
        )

        st.plotly_chart(
            px.pie(
                values=prediction[0],
                names=["No Churn", "Churn"],
                color=["No Churn", "Churn"],
                hole=0.3,
                title="Churn Probability",
                hover_name=["No Churn", "Churn"],
                
            ),
        )

elif option == "Upload CSV":
    st.header("Upload a CSV file for batch prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        with st.expander("View Uploaded Data"):
            input_df = process_uploaded_csv(uploaded_file)
            if input_df is not None:
                st.write(input_df)

        if input_df is not None and st.button("Predict"):
            predictions = model.predict(input_df)

            # Predicted indexes!
            predicted_indexes = drop_missing_convert_dt(input_df).index

            st.write(
                pd.DataFrame(
                    predictions, columns=["Predictions"], index=predicted_indexes
                ).replace({1: "Churn", 0: "No Churn"})
            )
            st.subheader(f"{len(input_df) - len(predictions)} rows were dropped!!")

# Add footer
st.sidebar.write("---")
st.sidebar.write(
    "Developed by <a href='https://www.github.com/darkdk123'>DarkDk123</a> © 2024",
    unsafe_allow_html=True,
)
