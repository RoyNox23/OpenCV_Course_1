import SessionState as SessionState
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

#---------------------------------------------------------------------------------------
# The SessionState class allows us to initialize and save variables across for across
# session states. This is a valuable feature that enables us to take different actions
# depending on the state of selected variables in the code. If this is not done then
# all variables are reset any time the application state changes (e.g., when a user
# interacts with a widget). For example, the confidence threshold of the slider has
# changed, but we are still working with the same image, we can detect that by
# comparing the current file_uploaded_id (img_file_buffer.id) with the
# previous value (ss.file_uploaded_id) and if they are the same then we know we
# don't need to call the face detection model again. We just simply need to process
# the previous set of detections.
#---------------------------------------------------------------------------------------
USE_SS = True
if USE_SS:
    ss = SessionState.get(file_uploaded_id=-1, # Initialize file uploaded index.
                          detections=None)     # Initialize detections.


@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net_load = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net_load


class VideoFaceProcessor:
    def detect_face_dnn(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        return detections

    def process_detections(self, frame, detections):
        bboxes = []
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_w)
                y1 = int(detections[0, 0, i, 4] * frame_h)
                x2 = int(detections[0, 0, i, 5] * frame_w)
                y2 = int(detections[0, 0, i, 6] * frame_h)
                bboxes.append([x1, y1, x2, y2])
                bb_line_thickness = max(1, int(round(frame_h / 200)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)
        return frame, bboxes

    def recv(self, frame):
        frame1 = frame.to_ndarray(format = "bgr24")
        frame1 = cv2.flip(frame1, 1)
        detections = self.detect_face_dnn(frame1)
        out_frame, _ = self.process_detections(frame1, detections)
        return av.VideoFrame.from_ndarray(out_frame, format = "bgr24")


st.title("OpenCV Face Detector Real Time")

net = load_model()
conf_threshold = st.slider("Set Confidence Level", min_value = 0.0, max_value = 1.0,
                           step = 0.01, value = 0.5)

webrtc_streamer("Web Streamer", video_processor_factory=VideoFaceProcessor)

