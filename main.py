# Import required libraries
import PIL.Image
import streamlit as st
from ultralytics import YOLO

# Set the path to the YOLOv8 model weights file
model_path = 'best.pt'

# Configure Streamlit page settings
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar configuration for image upload and model parameters
with st.sidebar:
    st.header("Image Configuration")
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    confidence = float(st.slider("Select Model Confidence Level", 25, 100, 40)) / 100

# Main page title and instructions
st.title("YOLOv8 Object Detection")
st.caption('Upload a photo and then click the "Detect Objects" button to see the results.')

# Layout for displaying uploaded and detected images
col1, col2 = st.columns(2)

# Display the uploaded image
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Load the model and perform object detection
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model from {model_path}. Ensure the model path is correct.")
    st.exception(e)

if st.sidebar.button('Detect Objects'):
    if source_img:
        try:
            # Perform detection
            res = model.predict(uploaded_image, conf=confidence)
            boxes = res.pandas().xyxy[0]  # Get detection results
            res_plotted = res.render()[0]  # Get image with rendered detections
            
            with col2:
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                # Display detection results
                with st.expander("Detection Results"):
                    st.dataframe(boxes[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']])
        except Exception as e:
            st.error("Error in object detection.")
            st.exception(e)
    else:
        st.warning("Please upload an image first.")



