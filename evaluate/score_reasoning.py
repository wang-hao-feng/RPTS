import re
import os
import sys
sys.path.append('..')
import json
from tqdm import tqdm
from dataset import RPTS
from preprocess import process
from argparse import ArgumentParser

from openai import OpenAI, BadRequestError

from score_prompt import *
from reasonging_tree import is_image, is_text, is_conclusion, is_context

def get_image_info(o_reasonings:list[str], image_num:int):
    image_info = [[] for _ in range(image_num)]
    for reasoning in o_reasonings:
        antecedents, consequent = reasoning.split('->')
        antecedents = antecedents.split('+')
        for antecedent in antecedents:
            if '图' in antecedent or 'image' in antecedent.lower():
                idx = int(re.findall(r'\d+', antecedent)[0])
                if len(image_info[idx]) != 0:
                    image_info[idx] += [consequent] + [f'{info}; {consequent}' for info in image_info[idx]]
                image_info[idx].append(consequent)
    return image_info

def api_call(prompt:str):
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
    response = model.chat.completions.create(model='gpt-4-turbo', 
                                             messages=messages,
                                             n=1,  
                                             temperature=0)
    return response.choices[0].message.content

def score_reasoning(antecedents:list[str], 
                    consequent:str, 
                    conclusions:list[str], 
                    data:dict) -> float:
    prompt = PROMPT2
    prompt = prompt.replace('[CONTEXT]', 'Context:\n{0}'.format(data['context']))
    prompt = prompt.replace('[CLUES]', 'Clues:\n{0}'.format('\n'.join(antecedents)))
    prompt = prompt.replace('[CONCLUSION]', 'Conclusion:\n{0}'.format(consequent))
    try:
        response = api_call(prompt)
    except BadRequestError as e:
        if 'repetitive patterns' in e.message:
            return 0.0
    score1 = float(re.findall(r'\d+\.\d+', response)[0])
    if score1 < 0.5:
        prompt = PROMPT2
        prompt = prompt.replace('[CONTEXT]', 'Context:\n{0}'.format(data['context']))
        prompt = prompt.replace('[CLUES]', 'Clues:\n{0}'.format('\n'.join(antecedents + data['textual_clue'] + conclusions)))
        prompt = prompt.replace('[CONCLUSION]', 'Conclusion:\n{0}'.format(consequent))
        response = api_call(prompt)
    score2 = round(float(re.findall(r'\d+\.\d+', response)[0]) * 0.8, 1)
    return max(score1, score2)

def score_image(image_info:list[str], sentence:str):
    prompt = PROMPT1
    prompt = prompt.replace('[FACTS]', 'Facts:\n{0}'.format('\n'.join(image_info)))
    prompt = prompt.replace('[SENTENCE]', 'Sentence:\n{0}'.format(sentence))
    try:
        response = api_call(prompt)
    except BadRequestError as e:
        if 'repetitive patterns' in e.message:
            return 0.0
    score = float(re.findall(r'\d+\.\d+', response)[0])
    return score 

def scoring(reasonings:list[str], data:dict):
    textual_clues = data['textual_clue']
    o_reasonings = data['reasoning']
    context = [data['context']]
    scores = []
    image_info = get_image_info(o_reasonings, len(data['visual_clue']))
    conclusions = []
    for reasoning in reasonings:
        antecedents, consequent = reasoning.split('->')
        antecedents = antecedents.split('+')
        consequent = ':'.join(consequent.split(':')[1:])
        if len(antecedents) == 1 and is_image(antecedents[0]):
            idx = int(re.findall(r'\d+', antecedents[0])[0])
            if idx < len(image_info):
                if len(image_info[idx]) == 0:
                    scores.append(1.0)
                else:
                    scores.append(score_image(image_info[idx], consequent))
            else:
                scores.append(0)
        else:
            ants = []
            score = 0
            for antecedent in antecedents:
                idx = re.findall(r'\d+', antecedent)
                idx = -1 if len(idx) == 0 else int(idx[0])
                for check_fn, texts in zip([is_image, is_text, is_conclusion, is_context], [image_info, textual_clues, conclusions, context]):
                    if check_fn(antecedent) and idx < len(texts):
                        if idx != -1:
                            ants += [f'{texts[idx]}'] if isinstance(texts[idx], str) else [f'{string}' for string in texts[idx]]
                        else:
                            ants += texts
            score = score_reasoning(ants, consequent, conclusions, data)
            scores.append(score)
        conclusions.append(consequent)
    return scores

if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('-l', '--language', choices=['zh', 'en'], type=str)
    parse.add_argument('-f', '--file_name', type=str)
    parse.add_argument('-ds', '--dataset_path', type=str, default=os.path.expanduser('~/    datasets/RPTS'))
    args = parse.parse_args()
    path = os.path.join(f'../results/cot_{args.language}/trans', args.file_name)
    output_dir = f'../results/cot_{args.language}/scoring'
    output_path = os.path.join(output_dir, args.file_name)
    
    model = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    dataset = RPTS(args.dataset_path, language=args.language)

    with open(path, encoding='utf-8') as f:
        results = json.load(f)
    scoring_results = results.copy()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as f:
            scoring_results = json.load(f)
    for i in tqdm(range(len(scoring_results)), desc='{0} preprocessing'.format(args.file_name.split('-cot')[0])):
        scoring_results[i]['reasoning'] = process(scoring_results[i]['reasoning'], dataset[i][0]['textual_clue'], args.language)
    reasonings = [result['reasoning'] for result in scoring_results]
    for i, reasoning in tqdm(enumerate(reasonings), desc='{0} scoring'.format(args.file_name.split('-cot')[0]), total=len(reasonings)):
        if 'scores' in scoring_results[i] and not any([line.split('->')[1].count(':') > 1 for line in reasoning]):
            continue
        scoring_results[i]['scores'] = scoring(reasoning, dataset[i][0])
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scoring_results, f, ensure_ascii=False)