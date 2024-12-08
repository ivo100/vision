import os
import torch

# Convert to TorchScript
#scripted_model = torch.jit.script(model)
# Save the TorchScript model
#scripted_model.save(model_path.replace(".pth", ".pt"))

# Load pt model
dir = os.path.expandvars("$HOME/tradebot/data/models")
name = "chart2.pt"
model_path = os.path.join(dir, name)
#print(model_path)

model = torch.jit.load(model_path)
model.eval()

# Confirm it's on CPU
print(next(model.parameters()).device)  # Should print: cpu

