import torch
import time
import torchvision.models as models

#import warnings
# Suppress only this specific warning
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", UserWarning)
#     model = models.resnet34(weights='DEFAULT')

def benchmark_model(model_func, device=torch.device("cpu"), input_size=(3, 224, 224)):

    # model = model_func(weights='DEFAULT').to(device)
    model = model_func(weights=None).to(device)

    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Timing
    start = time.time()
    for _ in range(100):
        _ = model(input_tensor)
    end = time.time()

    print(f"{model_func.__name__}: {end - start:.4f} seconds")

    print("Learner type:", type(model_func))
    print("Model device:", next(model_func.parameters()).device)
    #print("Model type:", type(model_func.model))
    return

# List of models to test
models_to_benchmark = [
    models.resnet18,
    # models.resnet34,
    # models.resnet50,
    models.efficientnet_b0,
    #models.mobilenet_v3_small,
]

device = torch.device("mps")
#device = torch.device("cpu")

for model_func in models_to_benchmark:
    benchmark_model(model_func,  device)

"""
resnet18: 0.3587 seconds
efficientnet_b0: 1.3757 seconds

resnet18: 1.3002 seconds
efficientnet_b0: 5.7117 seconds

"""
