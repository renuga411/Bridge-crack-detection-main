# import streamlit as st
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import os
# import pathlib
# import tempfile
# import cv2
# import time

# st.title("Crack Detection App")
# st.sidebar.title("Upload Model & Files")

# # **Model Upload**
# model_file = st.sidebar.file_uploader("Choose a model file...", type=["onnx"])

# # **Load YOLO Model with Error Handling**
# @st.cache_resource
# def load_model(model_bytes):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_model:
#         temp_model.write(model_bytes)
#         return YOLO(temp_model.name)

# # **Initialize session state**
# if 'model' not in st.session_state:
#     st.session_state.model = None
# if 'predicted_video_bytes' not in st.session_state:
#     st.session_state.predicted_video_bytes = None

# # **Load model if uploaded**
# if model_file is not None and st.session_state.model is None:
#     try:
#         model_bytes = model_file.read()
#         st.session_state.model = load_model(model_bytes)
#         st.sidebar.success("✅ Model uploaded and loaded successfully!")
#     except Exception as e:
#         st.sidebar.error(f"⚠️ Error loading model: {e}")

# # **Sample Input and Output Toggle Button**
# if "show_samples" not in st.session_state:
#     st.session_state.show_samples = True  # Default: Show samples

# if st.button("Show/Hide Sample Input and Output"):
#     st.session_state.show_samples = not st.session_state.show_samples  # Toggle visibility

# if st.session_state.show_samples:
#     st.subheader("Sample Input and Output")

#     base_dir = pathlib.Path(__file__).parent
#     public_folder_path = base_dir / "public"

#     sample_input_paths = [
#         public_folder_path / "4.jpg",
#         public_folder_path / "3.jpg",
#         public_folder_path / "15.jpg",
#         public_folder_path / "13.jpg",
#         public_folder_path / "12.jpg",
#         public_folder_path / "16.jpg"
#     ]

#     sample_images = []
#     for path in sample_input_paths:
#         try:
#             img = Image.open(path).convert("RGB")  # Convert to RGB to avoid channel issues
#             sample_images.append(img)
#         except Exception as e:
#             st.error(f"⚠️ Error loading sample image: {path}, {e}")

#     # **Detect Cracks in Sample Images**
#     detected_images = []
#     if st.session_state.model is not None:
#         for img in sample_images:
#             image_np = np.array(img)
#             image_np = image_np[:, :, :3]  
#             result = st.session_state.model.predict(image_np)
#             detected_image = result[0].plot()
#             detected_images.append(Image.fromarray(detected_image))

#         # **Display Sample Images**
#         col1, col2 = st.columns(2)
#         for i in range(len(sample_images)):
#             with col1:
#                 st.image(sample_images[i], caption=f"Sample Input {i + 1}", use_column_width=True)
#             with col2:
#                 st.image(detected_images[i], caption=f"Detected Image {i + 1}", use_column_width=True)

# # **Manual Prediction Section**
# st.subheader("Manual Prediction")

# st.markdown("This application allows you to upload images and predict cracks.")

# # **Upload Images or Videos**
# uploaded_files = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=True)

# if uploaded_files:
#     image_files = [file for file in uploaded_files if file.type.startswith("image")]
#     video_files = [file for file in uploaded_files if file.type.startswith("video")]

#     # **Process Images**
#     if image_files:
#         st.subheader("Uploaded Images")
#         num_cols = 2
#         cols = st.columns(num_cols)
#         uploaded_images = [Image.open(file).convert("RGB") for file in image_files]

#         for i, uploaded_image in enumerate(uploaded_images):
#             with cols[i % num_cols]:
#                 st.image(uploaded_image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

#         if st.button("Predict Images"):
#             if st.session_state.model is None:
#                 st.error("⚠️ Please upload a valid model before running predictions.")
#             else:
#                 detected_images = []
#                 for img in uploaded_images:
#                     image_np = np.array(img)
#                     image_np = image_np[:, :, :3]
#                     result = st.session_state.model.predict(image_np)
#                     detected_image = result[0].plot()
#                     detected_images.append(Image.fromarray(detected_image))

#                 st.success("✅ Image Prediction Completed!")
#                 st.subheader("Detected Images")
#                 for i, detected_image in enumerate(detected_images):
#                     with cols[i % num_cols]:
#                         st.image(detected_image, caption=f"Detected Image {i + 1}", use_column_width=True)

#     # **Process Videos**
#     if video_files:
#         if st.button("Predict Video"):
#             if st.session_state.model is None:
#                 st.error("⚠️ Please upload a valid model before running predictions.")
#             else:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
#                     temp_video.write(video_files[0].read())
#                     video_path = temp_video.name

#                 cap = cv2.VideoCapture(video_path)
#                 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 fps = cap.get(cv2.CAP_PROP_FPS)

#                 fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#                 temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

#                 progress_bar = st.progress(0)
#                 frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#                 frame_processed = 0
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     results = st.session_state.model.predict(frame, imgsz=640, conf=0.2)
#                     annotated_frame = results[0].plot()
#                     out.write(annotated_frame)

#                     frame_processed += 1
#                     progress_bar.progress(int((frame_processed / frame_count) * 100))

#                 cap.release()
#                 out.release()

#                 with open(temp_output_path, "rb") as f:
#                     st.session_state.predicted_video_bytes = f.read()
#                 os.remove(temp_output_path)

#                 st.success("✅ Video Processing Completed!")

#     # **Download Processed Video**
#     if st.session_state.predicted_video_bytes:
#         st.download_button(
#             label="Download Processed Video",
#             data=st.session_state.predicted_video_bytes,
#             file_name="predicted_video.mp4",
#             mime="video/mp4"
#         )


import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pathlib
import tempfile
import cv2
import time

st.title("Crack Detection App")
st.sidebar.title("Upload Model & Files")

# **Model Upload**
model_file = st.sidebar.file_uploader("Choose a model file...", type=["onnx"])

# **Load YOLO Model with Error Handling**
@st.cache_resource
def load_model(model_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_model:
        temp_model.write(model_bytes)
        return YOLO(temp_model.name)

# **Initialize session state**
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predicted_video_bytes' not in st.session_state:
    st.session_state.predicted_video_bytes = None

# **Load model if uploaded**
if model_file is not None and st.session_state.model is None:
    try:
        model_bytes = model_file.read()
        st.session_state.model = load_model(model_bytes)
        st.sidebar.success("✅ Model uploaded and loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading model: {e}")

# **Sample Input and Output Toggle Button**
if "show_samples" not in st.session_state:
    st.session_state.show_samples = True  # Default: Show samples

if st.button("Show/Hide Sample Input and Output"):
    st.session_state.show_samples = not st.session_state.show_samples  # Toggle visibility

if st.session_state.show_samples:
    st.subheader("Sample Input and Output")

    base_dir = pathlib.Path(".")  # Use current directory
    public_folder_path = base_dir / "public"

    sample_input_paths = [
        public_folder_path / "4.jpg",
        public_folder_path / "3.jpg",
        public_folder_path / "15.jpg",
        public_folder_path / "13.jpg",
        public_folder_path / "12.jpg",
        public_folder_path / "16.jpg"
    ]

    sample_images = []
    for path in sample_input_paths:
        try:
            img = Image.open(path).convert("RGB")  # Convert to RGB to avoid channel issues
            sample_images.append(img)
        except Exception as e:
            st.error(f"⚠️ Error loading sample image: {path}, {e}")

    # **Detect Cracks in Sample Images**
    detected_images = []
    if st.session_state.model is not None:
        for img in sample_images:
            image_np = np.array(img)
            image_np = image_np[:, :, :3]  
            result = st.session_state.model.predict(image_np)
            detected_image = result[0].plot()
            detected_images.append(Image.fromarray(detected_image))

        # **Display Sample Images**
        col1, col2 = st.columns(2)
        for i in range(len(sample_images)):
            with col1:
                st.image(sample_images[i], caption=f"Sample Input {i + 1}", use_column_width=True)
            with col2:
                st.image(detected_images[i], caption=f"Detected Image {i + 1}", use_column_width=True)

# **Manual Prediction Section**
st.subheader("Manual Prediction")

st.markdown("This application allows you to upload images and predict cracks.")

# **Upload Images or Videos**
uploaded_files = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=True)

if uploaded_files:
    image_files = [file for file in uploaded_files if file.type.startswith("image")]
    video_files = [file for file in uploaded_files if file.type.startswith("video")]

    # **Process Images**
    if image_files:
        st.subheader("Uploaded Images")
        num_cols = 2
        cols = st.columns(num_cols)
        uploaded_images = [Image.open(file).convert("RGB") for file in image_files]

        for i, uploaded_image in enumerate(uploaded_images):
            with cols[i % num_cols]:
                st.image(uploaded_image, caption=f"Uploaded Image {i + 1}", use_column_width=True)

        if st.button("Predict Images"):
            if st.session_state.model is None:
                st.error("⚠️ Please upload a valid model before running predictions.")
            else:
                detected_images = []
                for img in uploaded_images:
                    image_np = np.array(img)
                    image_np = image_np[:, :, :3]
                    result = st.session_state.model.predict(image_np)
                    detected_image = result[0].plot()
                    detected_images.append(Image.fromarray(detected_image))

                st.success("✅ Image Prediction Completed!")
                st.subheader("Detected Images")
                for i, detected_image in enumerate(detected_images):
                    with cols[i % num_cols]:
                        st.image(detected_image, caption=f"Detected Image {i + 1}", use_column_width=True)

    # **Process Videos**
    if video_files:
        st.subheader("Uploaded Video Preview")

        # **Show uploaded video before processing**
        video_bytes = video_files[0].read()
        st.video(video_bytes)

        if st.button("Predict Video"):
            if st.session_state.model is None:
                st.error("⚠️ Please upload a valid model before running predictions.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    temp_video.write(video_bytes)
                    video_path = temp_video.name

                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_processed = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = st.session_state.model.predict(frame, imgsz=640, conf=0.2)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    frame_processed += 1
                    progress_bar.progress(int((frame_processed / frame_count) * 100))

                cap.release()
                out.release()

                with open(temp_output_path, "rb") as f:
                    st.session_state.predicted_video_bytes = f.read()
                os.remove(temp_output_path)

                st.success("✅ Video Processing Completed!")

    # **Download Processed Video**
    if st.session_state.predicted_video_bytes:
        st.download_button(
            label="Download Processed Video",
            data=st.session_state.predicted_video_bytes,
            file_name="predicted_video.mp4",
            mime="video/mp4"
        )
