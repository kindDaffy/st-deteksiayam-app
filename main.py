# Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import subprocess

# Install dependencies manually if not installed
required_packages = ["opencv-python-headless", "streamlit", "ultralytics"]

for package in required_packages:
    subprocess.call([os.sys.executable, "-m", "pip", "install", package])


# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory of the current file
ROOT = FILE.parent

# Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

# Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'ayam.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedayam.jpg'


# Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'best.pt'

# In case of your custom model
# DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

# Page Layout
st.set_page_config(
    page_title="DETEKSI AYAM MATI",
    page_icon="🐔"
)

# Header
st.header("Deteksi Ayam Mati Menggunakan YOLO11")

# SideBar
st.sidebar.header("Model Configurations")

# No need for task selection since we're focusing only on Detection
# Directly set the model path for Detection
model_path = Path(DETECTION_MODEL)

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

# Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(e)

# Image Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type=("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_container_width=True)
            else:
                uploaded_image = Image.open(source_image)
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption="Detected Image", use_container_width=True)
            else:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf=confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:, :, ::-1]
                    st.image(result_plotted, caption="Detected Image", use_container_width=True)

                    # Extracting detected classes and counts
                    class_counts = {}
                    for box in boxes:
                        class_name = box.cls[0]  # Get the class index
                        class_label = model.names[int(class_name)]  # Get the class label
                        if class_label in class_counts:
                            class_counts[class_label] += 1
                        else:
                            class_counts[class_label] = 1
                    
                    # Displaying Detection Results
                    try:
                        with st.expander("Detection Results"):
                            for class_label, count in class_counts.items():
                                st.write(f"{class_label}: {count} objects")
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)

# Video Configuration
elif source_radio == VIDEO:
    # Upload video file from UI
    uploaded_video = st.sidebar.file_uploader("Choose a Video...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name  # Temporary file path for the uploaded video

        # Show video to user
        st.video(uploaded_video)

        # Flag to prevent re-detection of the video
        detect_button_clicked = st.sidebar.button("Detect Video Objects")
        
        if detect_button_clicked:
            try:
                video_cap = cv2.VideoCapture(video_path)
                st_frame = st.empty()  # This will be used to display the video frame
                
                # Create an empty container for the detection results
                detection_text = st.empty()

                while video_cap.isOpened():
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9 / 16))))

                        # Predict the objects in the image using YOLO11
                        result = model.predict(image, conf=confidence_value)

                        # Plot the detected objects on the video frame
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption="Detected Video",
                                       channels="BGR", use_container_width=True)

                        # Extracting detected classes and counts for video
                        class_counts = {}
                        for box in result[0].boxes:
                            class_name = box.cls[0]  # Get the class index
                            class_label = model.names[int(class_name)]  # Get the class label
                            if class_label in class_counts:
                                class_counts[class_label] += 1
                            else:
                                class_counts[class_label] = 1

                        # Displaying detection results in real-time on the same position
                        detection_text.markdown("### Real-time Detection Results:")
                        for class_label, count in class_counts.items():
                            detection_text.markdown(f"**{class_label}:** {count} objects")

                    else:
                        video_cap.release()
                        break

            except Exception as e:
                st.sidebar.error("Error Loading Video" + str(e))

