import os
import json
from PIL import Image
from torch.utils.data import Dataset

class RPTS(Dataset):
    def __init__(self, 
                 path:str=os.path.expanduser('~/datasets/RPTS'), 
                 language:str='en') -> None:
        super().__init__()
        self.path = path
        with open(os.path.join(self.path, f'texts_{language}.json'), 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
    
    def __getitem__(self, index):
        data = self.datas[index]
        text = {
            'ability': data['ability'], 
            'statement': data['statement'], 
            'context': data['context'], 
            'textual_clue': data['textual_clue'], 
            'visual_clue': [os.path.join(self.path, 'images', path) for path in data['visual_clue']], 
            'reasoning': data['reasoning'], 
            'answer': data['answer']
        }
        images = [Image.open(path) for path in text['visual_clue']]
        return text, images

    def __len__(self):
        return len(self.datas)