"""
# once?
import ssl
import certifi

ssl._create_default_https_context = ssl.create_default_context
ssl.create_default_context().load_verify_locations(certifi.where())

"""
"""
import requests

response = requests.get('https://www.google.com')
print(response.status_code)

import requests
response = requests.get('https://www.google.com')
print(response.status_code)

FAILS
import urllib.request

url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"
try:
    with urllib.request.urlopen(url) as response:
        print(response.status)
except Exception as e:
    print(f"Error: {e}")

"""

import ssl
import certifi
import urllib.request

url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"

ssl_context = ssl.create_default_context(cafile=certifi.where())

try:
    with urllib.request.urlopen(url, context=ssl_context) as response:
        print(response.status)
except Exception as e:
    print(f"Error: {e}")

import torch
from fastai.vision.all import *

# Load a dataset
path = untar_data(URLs.PETS)
