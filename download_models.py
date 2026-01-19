import os
from model_info import model_info
from huggingface_hub import snapshot_download

class Downloader:
    def __init__(self) -> None:
        self.token = os.environ['HF_TOKEN']

    def download_model_param(self, repo_id:str, local_dir:str):
        snapshot_download(repo_id=repo_id, 
                          local_dir=local_dir, 
                          repo_type='model', 
                          token=self.token, 
                          resume_download=True)

downloader = Downloader()

for name, info in model_info.items():
    if name in ['GPT4o'] or 'repo_id' not in info:
        continue
    print(f'Downloading {name}...')
    downloader.download_model_param(info['repo_id'], os.path.expanduser(info['path']))
    print('-'* 30)