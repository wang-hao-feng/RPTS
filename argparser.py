import os
import argparse
from prompt import prompt_map
from prompt_fn import prompt_fn_regisiter, image_token_prompt_regisiter
from models import model_regisiter

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset_path', type=str, default=os.path.expanduser('~/datasets/RPTS'))
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'])

    parser.add_argument('-tp', '--text_prompt', type=str, choices=list(prompt_map.keys()))
    parser.add_argument('-pf', '--prompt_fn', type=str, choices=list(prompt_fn_regisiter.keys()))
    parser.add_argument('-itp', '--image_token_prompt', type=str, choices=list(image_token_prompt_regisiter.keys()))

    parser.add_argument('-m', '--model', type=str, choices=list(model_regisiter.keys()))
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--nf4', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--use_flash_attention_2', action='store_true')

    parser.add_argument('-r', '--result_path', type=str, default='./results')
    parser.add_argument('-o', '--output', type=str, default='result.json')

    args = parser.parse_args()
    return args