import torch
import fastai
from fastai.vision.all import *

print(torch.__version__)
print(fastai.__version__)

# Ensure you're using the right device
device = torch.device("cuda")
print(device)

path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path / "images"), pat=r'(.+)_\d+.jpg', item_tfms=Resize(224))


# Try this specific approach for creating the learner
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Explicitly move to device
learn.model = learn.model.to(device)
learn.dls.to(device)

# Some additional debugging steps
print("Model device:", next(learn.model.parameters()).device)
print("Learner type:", type(learn))
print("Model type:", type(learn.model))

# Try fine-tuning
try:
    learn.fine_tune(1)
except Exception as e:
    print("Error details:", e)

"""
on cuda
epoch	train_loss	valid_loss	error_rate	time
0	1.476345	0.322440	0.112991	00:26
epoch	train_loss	valid_loss	error_rate	time
0	0.448729	0.267481	0.089986	00:35
"""
