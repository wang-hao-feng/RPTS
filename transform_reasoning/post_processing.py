import re

def get_index(antecedent:str):
    return int(re.findall(r'\d+', antecedent)[0])

def is_image(antecedent:str):
    return '图' in antecedent or 'image' in antecedent.lower()

def is_text(antecedent:str):
    return '文' in antecedent or 'text' in antecedent.lower()

def is_conclusion(antecedent:str):
    return '结论' in antecedent or 'conclusion' in antecedent.lower()

def cut_reasoning(reasoning:str) -> list[list[str]|str]:
    antecedents, consequent = reasoning.split('->')
    antecedents = antecedents.split('+')
    return [antecedents, consequent]

def paste_reasoning(antecedents:list[str], consequent:str) -> str:
    antecedents = ' + '.join(antecedents)
    return f'{antecedents} -> {consequent}'

def error_type(antecedents:list[str],
               consequent:str,  
               conclusion_index:int, 
               image_conclusion:dict[int, int]):
    if len(antecedents) == 1 and is_image(antecedents[0]):
        index = get_index(antecedents[0])
        if index in image_conclusion:
            return 'merge_image_conclusion', image_conclusion
        else:
            image_conclusion[index] = conclusion_index
    elif any([is_image(antecedent) for antecedent in antecedents]):
        index = sum([(is_image(antecedent)) for antecedent in antecedents])
        if index in image_conclusion:
            return 'replace_image2conclusion', image_conclusion
    return 'correct', image_conclusion

def merge_image_conclusion(antecedents:list[str],
                           consequent:str, 
                           **kwargs):
    pass

def replace_image2conclusion(antecedents:list[str],
                             consequent:str,
                             image_conclusion:dict[int, int], 
                             language:str, 
                             **kwargs):
    for i, antecedent in antecedents:
        if is_image(antecedent):
            index = get_index(antecedent)
            index = image_conclusion[index]
            antecedents[i] = f'Conclusion {index}' if language == 'en' else f'结论 {index}'
    return [antecedents, consequent]

processor = {
    'correct': lambda reasoning, **kwargs: reasoning, 
    'merge_image_conclusion': merge_image_conclusion, 
    'replace_image2conclusion': replace_image2conclusion
}

def post_processing(o_reasonings:list[str], 
                    textual_clues:list[str], 
                    language:str) -> list[str]:
    reasonings = []
    image_conclusion = {}
    o_reasonings = [cut_reasoning(o_reasoning) for o_reasoning in o_reasonings]
    for i, o_reasoning in enumerate(o_reasonings):
        error, image_conclusion = error_type(*o_reasoning)
        reasonings.append(processor[error](*o_reasoning, 
                                           image_conclusion=image_conclusion, 
                                           language=language))
    reasonings = [paste_reasoning(*reasoning) for reasoning in reasonings]
    return reasonings
    