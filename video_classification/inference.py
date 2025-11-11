import torch 
from preprocessing import extract_video_features
import torch.nn.functional as F
from model import LSTMSequenceModel
import torch
from feature_extractor import build_feature_extractor
from timm.data.transforms_factory import create_transform
import torch
from feature_extractor import build_feature_extractor
from timm.data import resolve_data_config

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

def predict_single_video(video_path, model, feature_extractor, DEVICE="cuda"):
    model.eval()
    with torch.no_grad():
        features, mask = extract_video_features(video_path, feature_extractor)
        features = features.unsqueeze(0).to(DEVICE)
        mask = mask.unsqueeze(0).to(DEVICE)
        logits = model(features, mask)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = "Real" if pred.item() == 1 else "Fake"
    print(f"\nVideo: {video_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf.item():.4f}")
    print(f"Probabilities: {probs.squeeze().cpu().tolist()}")


model =model = LSTMSequenceModel(num_features=NUM_FEATURES, max_seq_length=MAX_SEQ_LENGTH, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(r"C:\Users\Aliza Momin\Desktop\New_folder\deep_fake_initalstage\video_classification\model.pth", map_location=DEVICE))
model.to(DEVICE)
test_video = r'C:\Users\Aliza Momin\Desktop\New_folder\alia_deepfake.mp4'


feature_extractor = build_feature_extractor().to(DEVICE)
config = resolve_data_config({}, model=feature_extractor)
timm_transform = create_transform(**config)

predict_single_video(test_video, model, feature_extractor, DEVICE)
