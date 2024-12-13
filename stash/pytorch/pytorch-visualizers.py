import torch
import torch.nn as nn
from torchview import draw_graph
from torchinfo import summary
import hiddenlayer as hl
import graphviz
import numpy as np

# Sample model for demonstration
class SampleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SampleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def visualize_model_torchinfo(model, input_size=(1, 3, 224, 224)):
    """
    Use torchinfo to show model summary
    """
    print("\nModel Summary using torchinfo:")
    print("-" * 50)
    return summary(model, input_size=input_size, 
                  col_names=["input_size", "output_size", "num_params", "kernel_size"],
                  depth=4)

def visualize_model_torchview(model, input_size=(1, 3, 224, 224)):
    """
    Use torchview to create model graph
    """
    print("\nGenerating model graph using torchview...")
    model_graph = draw_graph(model, input_size=input_size, 
                           expand_nested=True,
                           graph_name="Model Architecture")
    return model_graph

def save_training_visualizations(model, train_loader, num_epochs=5, save_path='training_vis'):
    """
    Visualize training progress using HiddenLayer
    """
    print("\nGenerating training visualizations...")
    history = {'loss': [], 'acc': []}
    
    # Create HiddenLayer canvas
    canvas = hl.Canvas()
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = np.random.rand()  # Simulate loss
        epoch_acc = np.random.rand()   # Simulate accuracy
        
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        # Plot metrics
        canvas.draw_plot(history['loss'], name='Loss')
        canvas.draw_plot(history['acc'], name='Accuracy')
        
        # Save visualization
        canvas.save(f'{save_path}_epoch_{epoch}.png')
    
    return history

def main():
    # Create model instance
    model = SampleCNN()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # 1. Model summary using torchinfo
    model_summary = visualize_model_torchinfo(model)
    
    # 2. Model graph using torchview
    model_graph = visualize_model_torchview(model)
    
    # 3. Create dummy data loader for training visualization
    dummy_loader = [(torch.randn(32, 3, 224, 224), torch.randint(0, 2, (32,))) 
                   for _ in range(10)]
    
    # 4. Training visualizations
    history = save_training_visualizations(model, dummy_loader)
    
    print("\nVisualization complete! Check the output files for results.")

if __name__ == "__main__":
    main()
