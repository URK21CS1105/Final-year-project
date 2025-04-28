import io
import time
import cv2
import torch
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os
import winsound

# TTS Function
def tts(text):
    """Text-to-speech function that plays audio in real-time."""
    tts_t = gTTS(text)
    
    # Save to an in-memory bytes buffer
    mp3_fp = io.BytesIO()
    tts_t.write_to_fp(mp3_fp)
    mp3_fp.seek(0)  # Seek to the beginning of the in-memory buffer
    
    # Load the audio using pydub
    sound = AudioSegment.from_file(mp3_fp, format="mp3")
    
    # Create a custom temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="D:/Viren/Qriocity/Blind") as tmp_file:
        temp_wav_path = tmp_file.name
        sound.export(temp_wav_path, format="wav")
    
    # Play the sound using winsound
    winsound.PlaySound(temp_wav_path, winsound.SND_FILENAME)
    
    # Clean up the temporary file after playing the sound
    os.remove(temp_wav_path)


def inference(model=None):
    """Performs real-time object detection on video input using YOLO."""
    check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
    import streamlit as st
    from ultralytics import YOLO

    # Set html page configuration
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

    # Add video source selection dropdown
    st.sidebar.title("User Configuration")
    source = st.sidebar.selectbox("Video", ("webcam", "video"))

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        vid_file_name = 0

    # Add dropdown menu for model selection
    available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
    if model:
        available_models.insert(0, model.split(".pt")[0])  # insert model without suffix as *.pt is added later

    selected_model = st.sidebar.selectbox("Model", ['yolov8n'])
    with st.spinner("Model is downloading..."):
        model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
        class_names = list(model.names.values())  # Convert dictionary to list of class names
    st.success("Model loaded successfully!")

    # Multiselect box with class names and get indices of selected classes
    selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
    selected_ind = [class_names.index(option) for option in selected_classes]

    if not isinstance(selected_ind, list):  # Ensure selected_options is a list
        selected_ind = list(selected_ind)

    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

    fps_display = st.sidebar.empty()  # Placeholder for FPS display

    if st.sidebar.button("Start"):
        videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

        if not videocapture.isOpened():
            st.error("Could not open webcam.")

        stop_button = st.button("Stop")  # Button to stop the inference

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                break

            prev_time = time.time()  # Store initial time for FPS calculation

            # Store model predictions
            if enable_trk == "Yes":
                results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
            else:
                results = model(frame, conf=conf, iou=iou, classes=selected_ind)
            annotated_frame = results[0].plot()  # Add annotations on frame

            # Text-to-Speech for detected classes
            for detection in results[0].boxes:
                class_id = int(detection.cls[0])
                class_name = class_names[class_id]
                tts(f"Detected {class_name}")  # Call TTS to announce detected class

            # Calculate model FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)

            # Display the frame in an external OpenCV window
            cv2.imshow("Object Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if stop_button:
                videocapture.release()  # Release the capture
                torch.cuda.empty_cache()  # Clear CUDA memory
                st.stop()  # Stop streamlit app

            # Display FPS in sidebar
            fps_display.metric("FPS", f"{fps:.2f}")

        # Release the capture
        videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy OpenCV window
    cv2.destroyAllWindows()


# Main function call
if __name__ == "__main__":
    inference()
