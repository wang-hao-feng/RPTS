import os
import json
from tqdm import tqdm

import torch

from argparser import parse_args
from dataset import RPTS
from prompt import prompt_map
from prompt_fn import prompt_fn_regisiter, image_token_prompt_regisiter
from models import Model

args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rpts = RPTS(path=args.dataset_path, language=args.language)

base_prompt = prompt_map[args.text_prompt]
text_prompt = prompt_fn_regisiter[args.prompt_fn]
image_token_prompt = image_token_prompt_regisiter[args.image_token_prompt]
kwargs = {}
if args.nf4:
    kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
model = Model(model_name=args.model, 
              device=device, 
              bf16=args.bf16 and not args.nf4, 
              fp16=args.fp16 and not args.nf4, 
              load_in_4bit=args.nf4, 
              load_in_8bit=args.load_in_8bit, 
              use_flash_attention_2=args.use_flash_attention_2, 
              **kwargs
)

output_path = os.path.join(args.result_path, args.output)
def save(obj):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

results = []
with torch.inference_mode():
    for original_text, original_images in tqdm(rpts):
        text, images = text_prompt(original_text, original_images, base_prompt, args.language)
        text = image_token_prompt(text, images, args.language)
        response = model(text, images)
        results.append(response)
        save(results)