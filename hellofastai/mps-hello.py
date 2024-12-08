import torch
from fastai.vision.all import *
from dotenv import load_dotenv
#path = "/Users/ivostoyanov/.fastai/data/oxford-iiit-pet"

load_dotenv()
print("PYTORCH_ENABLE_MPS_FALLBACK", os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"))

path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path / "images"), pat=r'(.+)_\d+.jpg', item_tfms=Resize(224))

# Verify MPS availability
print("MPS Available:", torch.backends.mps.is_available())

# Ensure you're using the right device
device = torch.device("mps")

# Try this specific approach for creating the learner
learn = vision_learner(dls, resnet34, metrics=error_rate)

# Explicitly move to MPS
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
