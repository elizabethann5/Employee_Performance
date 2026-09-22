import streamlit as st
import numpy as np
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊"
)

st.title("📊 Employee Performance Predictor")
st.write("Enter Training Hours and Attendance to predict employee performance.")

# Cache the trained model so it runs instantly and builds natively in the cloud environment
@st.cache_resource
def get_trained_model():
    # 1. Dataset from the practical
    X = np.array([
        [2, 60], [3, 65], [4, 62], [5, 68], [6, 70],
        [7, 72], [8, 75], [9, 78], [10, 80], [11, 82],
        [12, 85], [13, 86], [14, 88], [15, 90], [16, 92]
    ], dtype=float)
    
    # 0 = Needs Improvement, 1 = Good
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

    # 2. Build ANN (as specified in student task: 8 neurons, 4 neurons, 1 output neuron)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 3. Compile and train for 100 epochs
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)
    return model

model = get_trained_model()

# User inputs
training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=30.0,
    value=8.0,
    step=1.0
)
attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# Prediction button
if st.button("Predict Performance"):
    input_data = np.array([[training_hours, attendance]])
    probability = model.predict(input_data, verbose=0)[0][0]

    result = "Good" if probability >= 0.5 else "Needs Improvement"

    st.subheader("Prediction Result")
    if result == "Good":
        st.success("Performance: GOOD")
    else:
        st.warning("Performance: NEEDS IMPROVEMENT")

    st.write("Good Probability:", round(float(probability) * 100, 2), "%")
    st.write("Training Hours:", training_hours)
    st.write("Attendance:", attendance, "%")
