import urllib
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# import certifi
# print(certifi.where())

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# flip-flop on certificates -

model = timm.create_model('efficientnet_b0', pretrained=True)
model.eval()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

filename = "dog.jpg"
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0) # transform and add batch dimension
print(tensor)


