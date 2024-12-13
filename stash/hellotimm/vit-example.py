from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2,
    label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))

# if a string is passed into the model argument, it will now use timm (if it is installed)
learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate)

learn.fine_tune(1)

"""
epoch     train_loss  valid_loss  error_rate  time    
0         0.209249    0.007400    0.003383    00:56     
epoch     train_loss  valid_loss  error_rate  time    
0         0.029550    0.004972    0.003383    01:02     
"""
