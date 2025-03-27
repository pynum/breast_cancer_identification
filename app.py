import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Load the saved model
MODEL_PATH = "breast_cancer_model.keras"  # Update if needed
model = load_model(MODEL_PATH)

# Class labels
class_labels = ["Benign", "Malignant"]

# Image preprocessing function
def preprocess_image(img):
    """Preprocess image for model prediction."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))  # Resize to match model input size/------=
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def predict_image(img):
    """Predict whether the image is benign or malignant."""
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)  # Get class index
    confidence = np.max(prediction) * 100  # Convert to percentage

    return class_labels[predicted_class], confidence

# Recommendation function based on inputs and prediction
def get_recommendations(prediction, age, family_history, breast_density, lump):
    """Generate fully personalized recommendations based on risk factors."""
    recommendations = {"Possible Causes": [], "Preventive Measures": [], "Medical Recommendations": [], "Treatment Options": []}

    # Adjust recommendations based on inputs
    if prediction == "Benign":
        recommendations["Possible Causes"].extend([
            "Hormonal changes (common in younger women)",
            "Fibroadenomas or cysts",
            "Benign breast conditions (e.g., fibrocystic changes)"
        ])
        recommendations["Preventive Measures"].extend([
            "Perform regular breast self-exams",
            "Routine mammograms (every 2 years if above 40)",
            "Maintain a healthy lifestyle with exercise & balanced diet"
        ])
        recommendations["Medical Recommendations"].extend([
            "Consult a doctor for monitoring if necessary",
            "Follow-up mammograms every 6-12 months if high risk"
        ])
        recommendations["Treatment Options"].extend([
            "No treatment required unless symptoms appear",
            "Surgical removal only if discomfort persists"
        ])

    else:  # Malignant Case
        recommendations["Possible Causes"].extend([
            "Genetic mutations (e.g., BRCA1, BRCA2)",
            "Family history of breast cancer",
            "Hormone-related factors"
        ])
        recommendations["Preventive Measures"].extend([
            "Early detection through regular screening",
            "Genetic counseling if high risk",
            "Healthy lifestyle (diet, exercise, no smoking/alcohol)"
        ])
        recommendations["Medical Recommendations"].extend([
            "Immediate consultation with an oncologist",
            "Further diagnostic tests (MRI, biopsy, etc.)"
        ])
        recommendations["Treatment Options"].extend([
            "Surgery (Lumpectomy, Mastectomy)",
            "Radiation Therapy",
            "Chemotherapy or Targeted Therapy",
            "Hormone Therapy (if hormone receptor-positive)"
        ])

    # Additional Risk-Based Modifications
    if age >= 45:
        recommendations["Preventive Measures"].append("Regular mammograms every year")
        recommendations["Medical Recommendations"].append("More frequent breast exams due to age-related risks")

    if family_history == "Yes":
        recommendations["Possible Causes"].append("Increased risk due to hereditary factors")
        recommendations["Preventive Measures"].append("Genetic testing to check for BRCA mutations")
        recommendations["Medical Recommendations"].append("Early screening and proactive lifestyle changes")

    if breast_density == "High":
        recommendations["Possible Causes"].append("Higher breast density can mask tumors in mammograms")
        recommendations["Preventive Measures"].append("Consider additional imaging tests (MRI, ultrasound)")
        recommendations["Medical Recommendations"].append("More frequent screenings due to higher risk")

    if lump == "Yes":
        recommendations["Medical Recommendations"].append("Immediate ultrasound or biopsy to rule out malignancy")
        recommendations["Treatment Options"].append("Consult a breast specialist for further tests")

    return recommendations

# Streamlit Web Interface
st.title("Breast Cancer Detection AI with Fully Personalized Recommendations")

# **User Input Fields**
age = st.slider("Select Age", 18, 80, 45)
family_history = st.radio("Family History of Breast Cancer?", ["Yes", "No"])
breast_density = st.selectbox("Breast Density", ["High", "Moderate", "Low"])
lump = st.radio("Do you have a palpable lump?", ["Yes", "No"])

# File uploader
uploaded_file = st.file_uploader("Upload a breast scan image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read and process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict when the button is clicked
    if st.button("Predict"):
        result, confidence = predict_image(img)
        st.success(f"**Prediction: {result}**")
        st.info(f"Confidence Score: {confidence:.2f}%")

        # Get and display personalized recommendations
        recommendations = get_recommendations(result, age, family_history, breast_density, lump)

        st.subheader("📌 Possible Causes")
        for cause in recommendations["Possible Causes"]:
            st.write(f"- {cause}")

        st.subheader("🛡️ Preventive Measures")
        for measure in recommendations["Preventive Measures"]:
            st.write(f"- {measure}")

        st.subheader("🩺 Medical Recommendations")
        for med in recommendations["Medical Recommendations"]:
            st.write(f"- {med}")

        st.subheader("💊 Treatment Options")
        for treat in recommendations["Treatment Options"]:
            st.write(f"- {treat}")
