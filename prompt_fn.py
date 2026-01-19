import os
from PIL import Image
from register import Register

prompt_fn_regisiter = Register()
image_token_prompt_regisiter = Register()

def ReplaceText(text:dict[str, str|list[str]], images:list[Image.Image], prompt:str, language:str):
    if language == 'en':
        prompt = prompt.replace('[CONTEXT]', '\Context:\n{0}\n'.format(text['context']))
        prompt = prompt.replace('[CLUE]', '\nTextual clues:\n{0}\n'.format('\n'.join([f'Text{i}：{clue}' for i, clue in enumerate(text['textual_clue'])])))
        prompt = prompt.replace('[STATEMENT]', '\nStatement:\n{0}\n'.format(text['statement']))
    elif language == 'zh':
        prompt = prompt.replace('[CONTEXT]', '\n上下文：\n{0}\n'.format(text['context']))
        prompt = prompt.replace('[CLUE]', '\n文本线索：\n{0}\n'.format('\n'.join([f'文本{i}：{clue}' for i, clue in enumerate(text['textual_clue'])])))
        prompt = prompt.replace('[STATEMENT]', '\n声明：\n{0}\n'.format(text['statement']))
    return prompt

@prompt_fn_regisiter('normal')
def Normal(text:dict[str, str|list[str]], images:list[Image.Image], prompt:str, language:str) -> tuple[str, list[Image.Image]]:
    prompt = ReplaceText(text, images, prompt, language)
    return prompt, images

@prompt_fn_regisiter('image_path')
def ImagePath(text:dict[str, str|list[str]], images:list[Image.Image], prompt:str, language:str) -> tuple[str, list[Image.Image]]:
    prompt = ReplaceText(text, images, prompt, language)
    return prompt, [os.path.abspath(path) for path in text['visual_clue']]

@image_token_prompt_regisiter('special_image_token')
def SpecialImageToken(prompt:str, images:list[Image.Image], language:str) -> str:
    if language == 'en':
        return prompt.replace('[IMAGE]', '\nVisual clues:\n{0}\n'.format('\n'.join(['<image>' for _ in range(len(images))])))
    elif language == 'zh':
        return prompt.replace('[IMAGE]', '\n图像线索：\n{0}\n'.format('\n'.join(['<image>' for _ in range(len(images))])))

@image_token_prompt_regisiter('without')
def Empty(prompt:str, **kwargs) -> str:
    return prompt.replace('\n\n[IMAGE]', '')