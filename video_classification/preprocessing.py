from timm.data.transforms_factory import create_transform
import torch
from feature_extractor import build_feature_extractor
from timm.data import resolve_data_config
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

IMG_SIZE = 299
BATCH_SIZE = 8
EPOCHS = 20
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
NUM_CLASSES=2
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature_extractor = build_feature_extractor().to(DEVICE)
config = resolve_data_config({}, model=feature_extractor)
timm_transform = create_transform(**config)

# FACE DETECTION + CROPPING
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_and_crop_face(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    h, w, _ = frame.shape
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
        margin = 0.15
        dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
        cropped = frame[y1:y2, x1:x2]
    else:
        min_dim = min(h, w)
        start_x, start_y = (w - min_dim)//2, (h - min_dim)//2
        cropped = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    print('face crop done')
    return cv2.resize(cropped, target_size)


#frames and feature
def frame_constructor(video_path, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return np.array([])
    stride = max(1, total // max_frames)
    frames, count = [], 0
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frame = detect_and_crop_face(frame)

            frames.append(frame[:, :, ::-1])
            count += 1
            if count >= max_frames:
                break
    cap.release()
    return np.array(frames)

def normalize_frame(frame):
    img = Image.fromarray(frame)
    print('normalization done')
    return timm_transform(img).unsqueeze(0)

def extract_video_features(video_path, feature_extractor, max_seq_len=MAX_SEQ_LENGTH):
    frames = frame_constructor(video_path)
    print('frame construction done')
    if frames.size == 0:
        return torch.zeros((max_seq_len, NUM_FEATURES)), torch.zeros(max_seq_len, dtype=torch.bool)
    features = torch.zeros((max_seq_len, NUM_FEATURES))
    mask = torch.zeros(max_seq_len, dtype=torch.bool)
    for i in range(min(max_seq_len, len(frames))):
        frame_tensor = normalize_frame(frames[i]).to(DEVICE)
        #print('normalization done')
        with torch.no_grad():
            feat = feature_extractor(frame_tensor)
        features[i] = feat.squeeze(0).cpu()
        mask[i] = True
    return features, mask

if __name__=='__main__':
    print('starting preprocessing')
    path=r'C:\Users\Aliza Momin\Desktop\New_folder\alia_deepfake.mp4'
    features, mask = extract_video_features(path, feature_extractor)
    print(features, mask)