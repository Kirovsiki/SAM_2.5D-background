import os
import requests
from tqdm import tqdm

def download_model(model_url, save_path):
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(block_size):
            progress_bar.update(len(chunk))
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    print(f"Model downloaded and saved at {save_path}")

model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
save_path = os.path.join("pretrained_models", "sam_vit_h_4b8939.pth")

os.makedirs("pretrained_models", exist_ok=True)
download_model(model_url, save_path)
