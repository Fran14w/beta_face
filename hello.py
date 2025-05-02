print("Hello")

import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from PIL import Image

# Create application title and file uploader widget.
st.title("Welcome to the Face Detection App(BETA)")
img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4'])

# Check if the file was uploaded
if img_file_buffer is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(img_file_buffer.read())
        tmp_file.close()

    # Debug: Print the file type for checking
    st.write(f"Uploaded file type: {img_file_buffer.type}")

    # Check if the uploaded file is a video
    if img_file_buffer.type.startswith('video'):
        # Initialize video capture from uploaded file
        video_cap = cv2.VideoCapture(tmp_file.name)

        # Check if video capture was successful
        if not video_cap.isOpened():
            st.error("Error opening video file!")
        else:
            st.write("Video opened successfully!")

            frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))

            size = (frame_w, frame_h)
            output_video_path = 'video_out.mp4'  # Save processed video here
            video_out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

            # Load pre-trained Haar Cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            while True:
                ret, frame = video_cap.read()
                if not ret:
                    st.write("End of video file.")
                    break

                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame like size 
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                video_out.write(frame)

            # Release video objects after processing
            video_cap.release()
            video_out.release()

            # Check if the output video was created successfully
            if os.path.exists(output_video_path):
                st.write("Video processing complete!")
                st.video(output_video_path) 
            else:
                st.error("Error creating the output video!")
    else:
        # If it's an image file, just perform face detection
        img = Image.open(tmp_file.name)
        img_array = np.array(img)

        # Convert image to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Load pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

        img_pil = Image.fromarray(img_array)
        st.image(img_pil, caption='Processed Image with Face Detection', use_column_width=True)
