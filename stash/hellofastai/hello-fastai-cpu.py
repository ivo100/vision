import torch
import fastai
from fastai.vision.all import *

print(torch.__version__)
# 2.5.0
print(fastai.__version__)
# 2.7.18

import multiprocessing
print(multiprocessing.cpu_count())
# 10 on mac mini M2 Pro

# Load a dataset
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path / "images"), pat=r'(.+)_\d+.jpg', item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)

# Some additional debugging steps
print("Model device:", next(learn.model.parameters()).device)
print("Learner type:", type(learn))
print("Model type:", type(learn.model))

# Train the model
learn.fine_tune(1)

"""
mac mini M2 PRO
MPS Available: True
Model device: mps:0
Learner type: <class 'fastai.learner.Learner'>
Model type: <class 'torch.nn.modules.container.Sequential'>
epoch     train_loss  valid_loss  error_rate  time    
0         1.522445    0.331212    0.112991    01:00     
epoch     train_loss  valid_loss  error_rate  time    
0         0.426120    0.270795    0.089986    01:20     

MPS Available: True
Model device: mps:0
Learner type: <class 'fastai.learner.Learner'>
Model type: <class 'torch.nn.modules.container.Sequential'>
epoch     train_loss  valid_loss  error_rate  time    
0         1.496453    0.308799    0.093369    00:59     
epoch     train_loss  valid_loss  error_rate  time    
0         0.443210    0.224046    0.074425    01:18     

===

MPS Available: True
Model device: mps:0
Learner type: <class 'fastai.learner.Learner'>
Model type: <class 'torch.nn.modules.container.Sequential'>
epoch     train_loss  valid_loss  error_rate  time    
0         1.496453    0.308799    0.093369    00:59     
epoch     train_loss  valid_loss  error_rate  time    
0         0.443210    0.224046    0.074425    01:18     

CPU

===
cuda T4 on kaggle

epoch	train_loss	valid_loss	error_rate	time
0	1.476345	0.322440	0.112991	00:26
epoch	train_loss	valid_loss	error_rate	time
0	0.448729	0.267481	0.089986	00:35

CPU 
epoch	train_loss	valid_loss	error_rate	time
0	1.424496	0.344468	0.109608	29:07
epoch	train_loss	valid_loss	error_rate	time
0	0.430106	0.263924	0.077131	40:57
add Codeadd Markdown
  
"""
