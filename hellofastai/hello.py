import torch
from fastai.vision.all import *
# print(URLs.PETS)
# https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz

# Load a dataset
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path / "images"), pat=r'(.+)_\d+.jpg', item_tfms=Resize(224))

# Set the device to MPS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(torch.backends.mps.is_available())  # Should return True

learn = vision_learner(dls, resnet34, metrics=error_rate).to(device)

learn.model = learn.model.to(device)
learn.dls.to(device)

# Move model to MPS
#learn = vision_learner(dls, resnet34, metrics=error_rate).to(device)
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Train the model
learn.fine_tune(1)

"""
MPS Available: True
Model device: mps:0
Learner type: <class 'fastai.learner.Learner'>
Model type: <class 'torch.nn.modules.container.Sequential'>
epoch     train_loss  valid_loss  error_rate  time    
0         1.522445    0.331212    0.112991    01:00     
epoch     train_loss  valid_loss  error_rate  time    
0         0.426120    0.270795    0.089986    01:20     

"""
