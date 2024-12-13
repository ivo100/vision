import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, device, num_epochs=5, batch_size=32):
    # Create synthetic data for benchmarking
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    target_tensor = torch.randint(0, 10, (batch_size,))

    # Move data to device
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

    end_time = time.time()
    return end_time - start_time

def benchmark_training():
    # Models to test
    models_to_benchmark = [
        models.resnet18,
        models.resnet34,
        models.efficientnet_b0,
        models.mobilenet_v3_small
    ]

    # Devices to test
    devices = [
        torch.device("cpu"),
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ]

    # Benchmark results
    results = {}

    for device in devices:
        print(f"\nBenchmarking on {device}")
        device_results = {}

        for model_func in models_to_benchmark:
            # Create model
            model = model_func(weights=None)
            model = model.to(device)

            # Train and time
            training_time = train_model(model, device)
            device_results[model_func.__name__] = training_time
            print(f"{model_func.__name__}: {training_time:.4f} seconds")

        results[str(device)] = device_results

    return results

# Run benchmark
benchmark_results = benchmark_training()

"""
Benchmarking on cpu
resnet18: 9.1984 seconds
resnet34: 14.7912 seconds
efficientnet_b0: 30.4496 seconds
mobilenet_v3_small: 7.1157 seconds

Benchmarking on cuda
resnet18: 1.5381 seconds
resnet34: 0.4866 seconds
efficientnet_b0: 0.7442 seconds
mobilenet_v3_small: 0.3421 seconds

Benchmarking on mps
resnet18: 3.0601 seconds
resnet34: 1.2396 seconds
efficientnet_b0: 11.1522 seconds
mobilenet_v3_small: 6.1773 seconds

"""
