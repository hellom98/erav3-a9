import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests

# Download the ImageNet class labels mapping
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(url)
class_idx = response.json()

# Create a mapping from class indices to labels
idx2label = {int(k): v[1] for k, v in class_idx.items()}

# Initialize the ResNet50 model
model = models.resnet50(pretrained=False)

# Load checkpoint
checkpoint = torch.load("checkpoint_epoch_55.pth", map_location=torch.device('cpu'))

# Check if the checkpoint contains a state_dict
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

# Set model to evaluation mode
model.eval()

# Define image preprocessing - using standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image):
    # Preprocess the image
    img_tensor = preprocess(Image.fromarray(image))
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Create result dictionary with human-readable labels
    results = {}
    for i in range(5):
        class_id = top5_catid[i].item()
        class_label = idx2label.get(class_id, f"Unknown class {class_id}")
        results[f"{class_label}"] = float(top5_prob[i])
    
    return results

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Custom Trained ResNet50 Classification",
    description="Upload an image to classify it using custom trained ResNet50 model"
)

if __name__ == "__main__":
    iface.launch()