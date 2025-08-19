import streamlit as st
import requests
import os

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Get backend URL from environment variable or use Render URL
backend_url = os.environ.get("BACKEND_URL", "https://your-app-name.onrender.com")

# App title and description
st.title("🏥 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of a patient having diabetes based on medical details.
Enter the patient's information below and click **Predict** to see the results.
""")

# Create form for user input
with st.form("prediction_form"):
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 0)
        glucose = st.slider("Glucose Level", 0, 200, 100)
        blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    
    with col2:
        insulin = st.slider("Insulin", 0, 850, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.slider("Age", 10, 100, 25)
    
    submitted = st.form_submit_button("Predict")

# When form is submitted
if submitted:
    # Create the data to send to the API
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    
    try:
        # Make request to the FastAPI backend
        response = requests.post(
            f"{backend_url}/predict",
            json=input_data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.header("Prediction Results")
            
            if result["prediction"] == 1:
                st.error(f"**Result:** {result['result']}")
            else:
                st.success(f"**Result:** {result['result']}")
            
            st.info(f"**Confidence:** {result['confidence'] * 100:.2f}%")
            
            # Add some explanation
            if result["prediction"] == 1:
                st.warning("This prediction suggests the patient may have diabetes. Please consult with a healthcare professional for further evaluation.")
            else:
                st.success("This prediction suggests the patient is not likely to have diabetes. Maintain a healthy lifestyle with regular checkups.")
        
        else:
            st.error(f"Error making prediction: {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the prediction server. Please make sure the backend is running. Error: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    This diabetes prediction model is based on the Pima Indians Diabetes Dataset.
    The model uses machine learning to predict the likelihood of diabetes based on input parameters.
    
    **Note:** This is a prediction tool, not a medical diagnosis. Always consult with healthcare professionals for medical advice.
    """)
    
    # Add a button to check API health
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{backend_url}/health")
            if response.status_code == 200:
                st.success("✅ API is healthy and responding")
            else:
                st.error("❌ API is not responding correctly")
        except:
            st.error("❌ Cannot connect to API")
    
    # Add a button to view model metrics
    if st.button("View Model Metrics"):
        try:
            response = requests.get(f"{backend_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                st.subheader("Model Performance Metrics")
                st.write(f"**Accuracy:** {metrics['accuracy'] * 100:.2f}%")
                st.write(f"**Precision:** {metrics['precision'] * 100:.2f}%")
                st.write(f"**Recall:** {metrics['recall'] * 100:.2f}%")
                st.write(f"**F1 Score:** {metrics['f1_score'] * 100:.2f}%")
            else:
                st.error("Could not retrieve metrics")
        except:
            st.error("Cannot connect to API")