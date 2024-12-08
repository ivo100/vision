import torch
from fastai.vision.all import *
print(URLs.PETS)

# Load a dataset
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path / "images"), pat=r'(.+)_\d+.jpg', item_tfms=Resize(224))

print(torch.backends.mps.is_available())  # Should return True

# Set the device to MPS
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")

# Move model to MPS

#learn = vision_learner(dls, resnet34, metrics=error_rate).to(device)

learn = vision_learner(dls, resnet34, metrics=error_rate)

# Train the model
learn.fine_tune(1)

