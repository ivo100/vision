import ssl
from fastdownload import FastDownload

# Create an unverified SSL context
#ssl_context = ssl._create_unverified_context()
import ssl
import certifi

url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"

ssl_context = ssl.create_default_context(cafile=certifi.where())

ssl.SSLContext(ssl_context)

# Monkey patch download_url in FastDownload
def patched_download_url(url, dest, overwrite=False, timeout=4*60):
    import urllib.request
    print(f"Downloading {url} with SSL verification disabled.")
    urllib.request.urlretrieve(url, dest, context=ssl_context)

FastDownload.download_url = patched_download_url

# Test the patch
from fastai.data.external import untar_data, URLs

path = untar_data(URLs.PETS)
print(f"Data downloaded to: {path}")
