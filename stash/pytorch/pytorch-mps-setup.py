import torch
import platform
import numpy as np
import time

def setup_mps_device():
    """
    Set up and verify MPS (Metal Performance Shaders) device for PyTorch on Apple Silicon
    Returns the appropriate device for model training
    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because this device does not have GPU support")
        return torch.device("cpu")
        
    return torch.device("mps")

def verify_mps_performance(device):
    """
    Run a simple benchmark to verify MPS is working and compare performance
    """
    # Matrix multiplication test
    size = 2000
    
    # Test on CPU
    print(f"\nRunning matrix multiplication test ({size}x{size})...")
    print("-" * 50)
    
    # CPU test
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # MPS test
    if device.type == "mps":
        a_mps = a_cpu.to(device)
        b_mps = b_cpu.to(device)
        
        # Warm-up run
        _ = torch.matmul(a_mps, b_mps)
        
        start_time = time.time()
        c_mps = torch.matmul(a_mps, b_mps)
        mps_time = time.time() - start_time
        print(f"MPS Time: {mps_time:.4f} seconds")
        print(f"Speedup: {cpu_time/mps_time:.2f}x")
        
        # Verify results match
        max_diff = torch.max(torch.abs(c_cpu - c_mps.cpu())).item()
        print(f"Maximum difference between CPU and MPS results: {max_diff}")

def test_basic_neural_network(device):
    """
    Test a simple neural network training on MPS
    """
    print("\nTesting basic neural network training...")
    print("-" * 50)
    
    # Generate dummy data
    X = torch.randn(1000, 20).to(device)
    y = torch.randint(0, 2, (1000,)).to(device)
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    start_time = time.time()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    print(f"Training time: {time.time() - start_time:.4f} seconds")

def print_system_info():
    """
    Print relevant system information
    """
    print("\nSystem Information:")
    print("-" * 50)
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

def main():
    """
    Main function to run all tests
    """
    print_system_info()
    
    # Set up device
    device = setup_mps_device()
    print(f"\nUsing device: {device}")
    
    # Run verification tests
    verify_mps_performance(device)
    test_basic_neural_network(device)

if __name__ == "__main__":
    main()
