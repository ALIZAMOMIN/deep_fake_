import torch 
from preprocessing import extract_video_features
import torch.nn.functional as F

def predict_single_video(video_path, model, feature_extractor, device="cuda"):
    model.eval()
    with torch.no_grad():
        features, mask = extract_video_features(video_path, feature_extractor)
        features = features.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        logits = model(features, mask)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = "Real" if pred.item() == 1 else "Fake"
    print(f"\nVideo: {video_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf.item():.4f}")
    print(f"Probabilities: {probs.squeeze().cpu().tolist()}")