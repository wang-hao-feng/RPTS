import os
import sys
sys.path.append('..')
import json
from tqdm import tqdm
from dataset import RPTS
from argparse import ArgumentParser
from reasonging_tree import ReasoningTree, ReasoningNode

def rpts(tree:ReasoningTree, 
         lambda_:float=0.95, 
         hf:int=1) -> float:
    assert isinstance(hf, int) and hf > 0
    def calculate_node_score(node:ReasoningNode)-> tuple[float, float, int]:
        if node.is_leaf:
            return 0, -hf, 0
        scores, exps, sum_weights = 0, [], 0
        for child in node.children:
            score, exp, sum_weight = calculate_node_score(child)
            scores += score
            exps.append(exp)
            sum_weights += sum_weight
        max_exp = max(exps)
        weight = lambda_ ** abs(max_exp + 1)
        return node.score * weight + scores, max_exp + 1, weight + sum_weights
    if tree.root is None:
        return 0.0
    tree_score, _, sum_weights = calculate_node_score(tree.root)
    if sum_weights == 0:
        return 1
    return tree_score / sum_weights

def same(result, answer):
    m = {'agree': ['支持', 'yes'], 'disagree': ['矛盾', '反对', 'no']}
    if answer in result.lower():
        return True
    return any(a in result.lower() for a in m[answer])

if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('-l', '--language', choices=['zh', 'en'], type=str)
    parse.add_argument('-f', '--file_name', type=str)
    parse.add_argument('-ds', '--dataset_path', type=str, default=os.path.expanduser('~/datasets/RPTS'))
    parse.add_argument('--lambda_', type=float, default=0.9)
    parse.add_argument('--hf', type=int, default=1)
    parse.add_argument('--filter', action='store_true')
    parse.add_argument('--threshold', type=float, default=0.5)
    args = parse.parse_args()
    path = os.path.join(f'../results/cot_{args.language}/scoring', args.file_name)

    dataset = RPTS(args.dataset_path, language=args.language)
    with open(path, encoding='utf-8') as f:
        results = json.load(f)
    rpts_scores = []
    acc_scores = []
    for i in tqdm(range(len(results))):
        reasoning = results[i]['reasoning']
        scores = results[i]['scores']
        texts, images = dataset[i]
        tree = ReasoningTree(len(images), texts['textual_clue'], reasoning, scores)
        o_tree = ReasoningTree(len(images), texts['textual_clue'], texts['reasoning'], [1] * len(texts['reasoning']))
        rpts_score = rpts(tree, args.lambda_, args.hf)
        rpts_scores.append(rpts_score)
        acc = same(results[i]['answer'], texts['answer'])
        if args.filter:
            acc = acc and rpts_score >= args.threshold
        acc_scores.append(acc)
    print(f'RPTS: {sum(rpts_scores) / len(rpts_scores):.2f}')
    print(f'Accuracy: {sum(acc_scores) / len(acc_scores) * 100:.2f}%')