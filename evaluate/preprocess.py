import re
from sentence_transformers import SentenceTransformer
from reasonging_tree import get_index, is_image, is_text, is_conclusion, is_context

st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

def cut_reasoning(reasoning:str, language:str) -> list[list[str]|str]:
    antecedents, consequent = reasoning.split('->')
    consequent = consequent.replace('：', ':')
    consequent = [get_index(consequent), ':'.join(consequent.split(':')[1:])]
    antecedents = antecedents.split('+')
    new_antecedents = []
    for antecedent in antecedents:
        idx = get_index(antecedent)
        if is_image(antecedent):
            new_antecedents.append(['Image' if language == 'en' else '图片', idx])
        elif is_text(antecedent):
            new_antecedents.append(['Text' if language == 'en' else '文本', idx])
        elif is_conclusion(antecedent):
            new_antecedents.append(['Conclusion' if language == 'en' else '结论', idx])
        elif is_context(antecedent):
            new_antecedents.append(['Context' if language == 'en' else '上下文', 0])
    return [new_antecedents, consequent]

def paste_reasoning(reasoning:list[list[str, int], list[int, str]], language:str) -> str:
    antecedents, consequent = reasoning
    antecedents = ' + '.join([f'{antecedent[0]}{antecedent[1]}' if not is_context(antecedent[0]) else antecedent[0] for antecedent in antecedents])
    consequent = '{0}{1}:{2}'.format('Conclusion' if language == 'en' else '结论', *consequent)
    return f'{antecedents} -> {consequent}'

def error_type(i:int, 
               reasonings:list[list[list[list[str, int]], list[int, str]]], 
               image_conclusion:dict[int, int]):
    error_code = 'correct'
    antecedents, consequent = reasonings[i]
    if len(antecedents) == 0:
        error_code = 'empty_antecedent'
    elif len(antecedents) == 1 and is_image(antecedents[0][0]):
        index = antecedents[0][1]
        if index in image_conclusion:
            error_code = 'merge_image_conclusion'
        else:
            image_conclusion[index] = consequent[0]
    elif any([is_image(antecedent[0]) for antecedent in antecedents]):
        index = int(sum([(is_image(antecedent[0]) * antecedent[1]) for antecedent in antecedents]))
        if index in image_conclusion:
            error_code = 'replace_image2conclusion'
    elif any([antecedent[1] == -1 for antecedent in antecedents]):
        error_code = 'without_index'
    elif len(antecedents) == 1 and is_text(antecedents[0][0]):
        error_code = 'single_textual_clue'
    return error_code, image_conclusion

def delete_line(i:int, 
                reasonings:list[list[list[list[str, int]], list[int, str]]], 
                **kwargs):
    reasonings = reasonings[:i] + reasonings[i+1:]
    for j in range(i, len(reasonings)):
        reasonings[j][1][0] -= 1
        for k in range(len(reasonings[j][0])):
            if is_conclusion(reasonings[j][0][k][0]) and reasonings[j][0][k][1] > i:
                reasonings[j][0][k][1] -= 1
    return reasonings

def replace_conclusion(reasonings:list[list[list[list[str, int]], list[int, str]]], 
                       conclusion_idx:int, 
                       target:list[str, int]):
    for i in range(len(reasonings)):
        for j in range(len(reasonings[i][0])):
            if is_conclusion(reasonings[i][0][j][0]) and reasonings[i][0][j][1] == conclusion_idx:
                reasonings[i][0][j] = target
    return reasonings

def merge_image_conclusion(i:int, 
                           reasonings:list[list[list[list[str, int]], list[int, str]]], 
                           image_conclusion:dict[int, int], 
                           language:str, 
                           **kwargs):
    index, conclusion = image_conclusion[reasonings[i][0][0][1]], reasonings[i][1]
    reasonings[index][1][1] += ';' + conclusion[1]
    reasonings = replace_conclusion(reasonings, conclusion[0], ['Conclusion' if language == 'en' else '结论', index])
    return delete_line(i, reasonings)

def replace_image2conclusion(i:int, 
                             reasonings:list[list[list[list[str, int]], list[int, str]]],
                             image_conclusion:dict[int, int], 
                             language:str, 
                             **kwargs):
    antecedents = reasonings[i][0]
    for j, antecedent in enumerate(antecedents):
        if is_image(antecedent[0]):
            index = image_conclusion[antecedent[1]]
            reasonings[i][0][j] = ['Conclusion' if language == 'en' else '结论', index]
    return reasonings

def without_index(i:int, 
                  reasonings:list[list[list[list[str, int]], list[int, str]]], 
                  image_conclusion:dict[int, int], 
                  textual_clues:list[str], 
                  language:str, 
                  **kwargs):
    antecedents = reasonings[i][0]
    retain = [antecedent for antecedent in antecedents if antecedent[1] != -1]
    without = [antecedent for antecedent in antecedents if antecedent[1] == -1]
    addition = []
    if any([is_image(antecedent[0]) for antecedent in without]):
        addition += [['Conclusion' if language == 'en' else '结论', idx] for idx in image_conclusion.values()]
    if any([is_text(antecedent[0]) for antecedent in without]):
        addition += [['Text' if language == 'en' else '文本', idx] for idx in range(len(textual_clues))]
    if any([is_image(antecedent[0]) for antecedent in without]):
        addition += [['Conclusion' if language == 'en' else '结论', idx] for idx in range(i)]
    reasonings[i][0] = retain + addition
    return reasonings

def single_textual_clue(i:int, 
                     reasonings:list[list[list[list[str, int]], list[int, str]]], 
                     textual_clues:list[str], 
                     language:str, 
                     **kwargs):
    index, conclusion = reasonings[i][0][0][1], reasonings[i][1]
    if index >= len(textual_clues):
        return reasonings
    clue = textual_clues[index]
    conclusion_embedding = st_model.encode([conclusion[1]], normalize_embeddings=True)
    clue_embedding = st_model.encode([clue], normalize_embeddings=True)
    sim = st_model.similarity(conclusion_embedding, clue_embedding).item()
    if sim > 0.75:
        reasonings = replace_conclusion(reasonings, conclusion[0], ['Text' if language == 'en' else '文本', index])
        reasonings = delete_line(i, reasonings)
    return reasonings

processor = {
    'correct': lambda reasonings, **kwargs: reasonings, 
    'merge_image_conclusion': merge_image_conclusion, 
    'replace_image2conclusion': replace_image2conclusion, 
    'without_index': without_index, 
    'single_textual_clue': single_textual_clue, 
    'empty_antecedent': delete_line
}

def process(o_reasonings:list[str], 
            textual_clues:list[str], 
            language:str) -> list[str]:
    image_conclusion = {}
    o_reasonings = [cut_reasoning(o_reasoning, language) for o_reasoning in o_reasonings]
    i = 0
    reasonings = o_reasonings
    reasoning_length = len(reasonings)
    while i < len(reasonings):
        error, image_conclusion = error_type(i=i, 
                                             reasonings=reasonings, 
                                             image_conclusion=image_conclusion)
        reasonings = processor[error](i=i,
                                      reasonings=reasonings, 
                                      image_conclusion=image_conclusion, 
                                      textual_clues=textual_clues, 
                                      language=language)
        if len(reasonings) == reasoning_length:
            i += 1
        else:
            reasoning_length = len(reasonings)
    reasonings = [paste_reasoning(reasoning, language) for reasoning in reasonings]
    return reasonings