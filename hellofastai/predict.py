import os
from pathlib import Path

import torch
import torch
from torchvision import transforms
from PIL import Image
import io

resize = 256

labels = ['CHOP', 'LONG_CALL', 'LONG_PUT', 'RANGE', 'SHORT_CALL', 'SHORT_PUT', 'SKIP']

def predict(model, img: bytes):

    image = Image.open(io.BytesIO(img)).convert('RGB')

    # transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor()])
    # w/o resize
    transform = transforms.Compose([transforms.ToTensor()])

    # with normalization
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example for pretrained models
    # ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)


    n = 3
    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)

    print("Input tensor:", input_tensor)
    print("Logits:", output)

    probabilities = torch.softmax(output, dim=1)
    print("Probabilities:", probabilities)

    top_values, top_indices = probabilities.topk(n, dim=1)
    print("Top values:", top_values)
    print("Top indices:", top_indices)

    # Get the predicted index and its probability
    idx = probs.argmax(dim=1).item()
    conf = probs[0, idx].item()  # Assuming batch size = 1
    print(f"Predicted top label index: {idx}, Confidence: {conf:.4f}")

    n = 3  # Number of top predictions
    top_n_probs, top_n_indices = torch.topk(probs, n, dim=1)  # Top `n` probabilities and indices

    print("Top Predictions:")
    for i in range(n):
        label = labels[top_n_indices[0, i].item()]  # Assuming batch size = 1
        confidence = top_n_probs[0, i].item()
        print(f"index {top_n_indices[0, i]}, conf: {confidence:.4f}")

    return labels[idx], conf

if __name__ == "__main__":
    # Load pt model
    dir = os.path.expandvars("$HOME/tradebot/data/models")
    name = "chart2.pt"
    model_path = os.path.join(dir, name)
    #print(model_path)

    model = torch.jit.load(model_path)
    model.eval()
    # # Confirm it's on CPU
    # print(next(model.parameters()).device)  # Should print: cpu

    # dir = os.path.expandvars("$HOME/tradebot/charts/2024-12-10")
    # name = "SMH_NVDA_2024-12-10-010.png"
    # dir = os.path.expandvars("$HOME/Documents/github/PY/TBPY/labeled-images")
    dir = "/Users/ivostoyanov/Documents/github/GO/go-practice/predict_torch"
    # name = "LONG_CALL/XLC_META_2024-12-06-020.png"
    name = "images/XLC_META_2024-12-06-020-256.png"
    image_path = os.path.join(dir, name)
    print("image "+image_path)
    with open(image_path, "rb") as file:
        img = file.read()
        label, conf = predict(model, img)
        # print(label, conf)

"""
Predicted top label index: 1, Confidence: 0.5816
LONG_CALL 0.5816490054130554


Top Predictions:
LONG_CALL: 0.5816
SKIP: 0.2937
SHORT_CALL: 0.0920

"""
