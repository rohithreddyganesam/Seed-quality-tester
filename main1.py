import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_img):
    model = tf.keras.models.load_model('Part2_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Find our seed quality"])

# Home page
if app_mode == "Home":
    st.header("Seed Quality Finders")
    image_path = "Home1.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
        Welcome to the Seed Quality Finders!  🔍
    
    Our mission is to help in identifying Seed Quality efficiently. Upload an image of a Seed, and our system will analyze it to detect any signs of Quality. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Find our seed quality** page and upload an image of a Seed with suspected Quality.
    2. **Quality Analysis:** Quality analysis may involve comparing the extracted features against predefined quality criteria or standards.
    3. **Feature Extraction:** CNNs are capable of learning hierarchical representations of image features, starting from low-level features like edges and textures and gradually progressing to higher-level features like shapes and objects
    4. **Classification:** Classification may involve assigning a quality score or label to each seed, such as "Average quality," "Bad quality,"  "Excellent quality," " Good quality," " Worst quality."
    5.  **Results:** View the results and recommendations for further action.
                

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate Quality detection.
    - **User-Friendly:**  With simple navigation and clear visualizations, you can easily interpret the results of the seed quality analysis and make informed decisions.
    - **Fast :** Receive results in seconds, allowing for quick decision-making.
    - **Efficiency:** Say goodbye to manual seed inspection processes that are time-consuming and prone to errors.

    

    ### Get Started
    Click on the **Seed Quality** page in the sidebar to upload an image and experience the power of our Seed Quality Finders !
    """)

# Prediction Page
elif app_mode == "Find our seed quality":
    st.header("Seed Quality")
    st.markdown("""
        <style>
            .animate-fade-in {
                animation: fade-in-animation 1s ease-in;
            }
            @keyframes fade-in-animation {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .animate-slide-down {
                animation: slide-down-animation 1s ease-in-out;
            }
            @keyframes slide-down-animation {
                from { transform: translateY(-100px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .animate-zoom-in {
                animation: zoom-in-animation 1s ease-out;
            }
            @keyframes zoom-in-animation {
                from { transform: scale(0.5); opacity: 0; }
                to { transform: scale(1); opacity: 1; }
            }
            .animate-rotate {
                animation: rotate-animation 1s linear infinite;
            }
            @keyframes rotate-animation {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            .animate-shake {
                animation: shake-animation 1s ease-in-out infinite;
            }
            @keyframes shake-animation {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
                20%, 40%, 60%, 80% { transform: translateX(10px); }
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="animate-fade-in"><h3 class="animate-slide-down">Upload an Image to Analyze Seed Quality</h3></div>', unsafe_allow_html=True)
    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            class_name = ['Average', 'Bad', 'Excellent', 'Good', 'Worst']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
