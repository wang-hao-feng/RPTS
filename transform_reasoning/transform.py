import os
import re
import json
from tqdm import tqdm
from argparse import ArgumentParser

from openai import OpenAI, BadRequestError

from transform_prompt import *
import sys
sys.path.append('..')
from dataset import RPTS

def api_call(prompt:str, 
             model_name:str='gpt-4o'):
    messages = [
        {
            'role': 'user', 
            'content': [
                {
                    'type': 'text', 
                    'text': prompt, 
                }
            ]
        }
    ]
    response = model.chat.completions.create(model=model_name, 
                                             messages=messages,
                                             n=1,  
                                             temperature=0)
    return response.choices[0].message.content

def transform(result:str, data:dict) -> str:
    is_en = args.language == 'en'
    prompt = REASONING_TRANSFORM_PROMPT_EN if is_en else REASONING_TRANSFORM_PROMPT_ZH
    prompt.replace('[TEXTUAL_CLUE]', 
                   '\n'.join(['文本线索:' if is_en else 'Textual clues:'] + 
                             [f'文本{i}:{clue}' if is_en else f'Text{i}: {clue}' for i, clue in enumerate(data['textual_clue'])]))
    prompt = prompt.replace('[IMAGE_NUM]', str(len(data['visual_clue'])))
    prompt = prompt.replace('[RESULT]', f'Model\'s reasonings:\n{result}' if is_en else f'模型的推理:\n{result}')
    response = api_call(prompt)
    response = re.findall(r'```.+```', response, re.DOTALL)[0]
    reasoning = []
    for line in response.split('\n'):
        if line == '```':
            continue
        if '->' in line:
            reasoning.append(line)
        else:
            reasoning[-1] += '\n' + line
    return reasoning

def extract_answer(result:str, data:dict) -> str:
    is_en = args.language == 'en'
    prompt = ANSWER_EXTRACT_PROMPT_EN if is_en else ANSWER_EXTRACT_PROMPT_ZH
    prompt = prompt.replace('[STATEMENT]', ('Statement:' if is_en else '声明:') + data['statement'])
    prompt = prompt.replace('[RESULT]', f'Sentence:\n{result}' if is_en else f'模型的推理:\n{result}')
    try:
        response = api_call(prompt, 'gpt-4-turbo')
    except BadRequestError as e:
        return 'Cannot be determined'
    return response
        
if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('-l', '--language', choices=['zh', 'en'], type=str)
    parse.add_argument('-f', '--file_name', type=str)
    parse.add_argument('-ds', '--dataset_path', type=str, default=os.path.expanduser('~/    datasets/RPTS'))
    args = parse.parse_args()
    path = os.path.join(f'../results/cot_{args.language}', args.file_name)
    output_dir = f'../results/cot_{args.language}/trans'
    output_path = os.path.join(output_dir, args.file_name)

    model = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    dataset = RPTS(args.dataset_path, language=args.language)

    with open(path, encoding='utf-8') as f:
        results = json.load(f)
    trans_results = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            trans_results = json.load(f)
    for i, (result) in tqdm(enumerate(results), desc=args.file_name, total=len(results)):
        if i < len(trans_results):
            continue
        data = dataset[i][0]
        trans_results.append({'reasoning': transform(result, data), 
                              'answer': extract_answer(result['reasoning'][-1].split('->')[-1], data)})
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trans_results, f, ensure_ascii=False)
