
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def build_feature_extractor():
    model = timm.create_model(
        'xception',
        pretrained=True,
        num_classes=0,     # removes classifier
        global_pool='avg'  # keeps 2048-dim pooled features
    )
    model.eval()
    return model



device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = build_feature_extractor().to(device)
print('feature extractor model loaded')
# Get the proper preprocessing config for normalization
#config = resolve_data_config({}, model=feature_extractor)
#timm_transform = create_transform(**config)

#print(config)