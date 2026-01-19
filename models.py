import os
import math
from PIL import Image

import torch
from openai import OpenAI
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from transformers import PreTrainedModel, LlamaTokenizer, PreTrainedTokenizer, CLIPImageProcessor
from ShareGPT4V.share4v.model import Share4VLlamaForCausalLM
from ShareGPT4V.share4v.model.builder import load_pretrained_model
from ShareGPT4V.share4v.constants import IMAGE_TOKEN_INDEX
from ShareGPT4V.share4v.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from model_info import model_info
from register import Register
from image_process import base64_images, padding_images, concat_images

model_regisiter = Register()

intervl_transform = transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

@model_regisiter('GPT4o')
def GPT4o(text:str, 
          images:list[Image.Image], 
          model:OpenAI, 
          **kwargs) -> str:
    images = base64_images(images)
    messages = [
        {
            'role': 'user', 
            'content': [
                {
                    'type': 'text', 
                    'text': text, 
                }
            ] + [
                {
                    'type': 'image_url', 
                    'image_url': {
                        'url': f'data:image/png;base64,{image}'
                    }
                } for image in images
            ]
        }
    ]
    response = model.chat.completions.create(model='gpt-4o', messages=messages)
    return response.choices[0].message.content

# InstructBLIP
@model_regisiter('InstructBLIP-vicuna-13b')
def InstructBLIP(text:str, 
                 images:list[Image.Image], 
                 model:InstructBlipForConditionalGeneration, 
                 processor:InstructBlipProcessor):
    images = concat_images(images)[0]
    prompt = text
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, 
                            max_length=2048, 
                            do_sample=True, 
                            num_beams=5,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                            temperature=1,)
    return processor.batch_decode(output, skip_special_tokens=True)[0].strip()

# InternVL2
@model_regisiter('InternVL2-Llama3-76B')
def InternVL(text:str, 
             images:list[Image.Image], 
             model:PreTrainedModel, 
             processor:PreTrainedTokenizer):
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    def process_image(image, input_size=448, max_num=12):
        transform = intervl_transform
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    images = [process_image(image).to(dtype=model.dtype).cuda() for image in images]
    image = torch.cat(images, dim=0)
    prompt = text
    generation_config = dict(max_new_tokens=2048, do_sample=False)
    output = model.chat(processor, image, prompt, generation_config)
    return output

# ShareGPT4V
@model_regisiter('ShareGPT4V-13B')
def ShareGPT4V(text:str, 
               images:list[Image.Image], 
               model:Share4VLlamaForCausalLM, 
               processor:tuple[PreTrainedTokenizer, CLIPImageProcessor]):
    tokenizer, image_processor = processor
    prompt = text
    images = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    output = model.generate(input_ids, images=images, max_length=2048, use_cache=True, do_sample=False)
    return tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

# Llava
@model_regisiter('Llava-v1.5-13B')
def Llava_vicuna(text:str, 
                 images:list[Image.Image], 
                 model:LlavaForConditionalGeneration|LlavaNextForConditionalGeneration, 
                 processor:LlavaProcessor|LlavaNextProcessor) -> str:
    prompt = 'You are a famous detective who is skilled at discovering details and inferring based on clues very well.'
    prompt += f'USER: {text}\nASSISTANT:'
    inputs = processor(prompt, images, return_tensors='pt', truncation=True, padding=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return processor.decode(output[0][2:], skip_special_tokens=True).split('ASSISTANT:')[-1]

@model_regisiter('Llava-Next-34B')
def Llava_Next(text:str, 
               images:list[Image.Image], 
               model:LlavaNextForConditionalGeneration, 
               processor:LlavaNextProcessor) -> str:
    images = padding_images(images)
    prompt = f'<|im_start|>system\nYou are a famous detective who is skilled at discovering details and inferring based on clues very well.<|im_end|><|im_start|>user\n{text}<|im_end|><|im_start|>assistant\n'
    inputs = processor(prompt, images, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
    return processor.decode(output[0][2:], skip_special_tokens=True).split('assistant')[-1]

# Qwen-VL
@model_regisiter('Qwen-VL-Chat')
def QwenVL(text:str, 
           images:list[str], 
           model:PreTrainedModel, 
           processor) -> str:
    query = []
    for path in images:
        query.append({'image':path})
    query.append({'text':text})
    query = processor.from_list_format(query)
    output, _ = model.chat(processor, query=query, history=None, system='You are a famous detective who is skilled at discovering details and inferring based on clues very well.')
    return output

class Model():
    def __init__(self, 
                 model_name:str, 
                 device:str, 
                 bf16:bool=True, 
                 fp16:bool=False, 
                 load_in_8bit:bool=False, 
                 load_in_4bit:bool=False, 
                 use_flash_attention_2:bool=True, 
                 **kwargs):
        info = model_info[model_name]
        model_path = os.path.expanduser(info['path'])
        processor_path = os.path.expanduser(info['processor_path'])
        if 'gpt' not in model_name.lower():
            self.processor = info['processor'](processor_path)
            self.model = info['model'](model_path, 
                                       torch_dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32), 
                                       low_cpu_mem_usage=True, 
                                       load_in_8bit=load_in_8bit, 
                                       load_in_4bit=load_in_4bit, 
                                       use_flash_attention_2=use_flash_attention_2, 
                                       device_map='auto' if 'internvl' not in model_name.lower() else self.split_intervl_model(model_name), 
                                       **kwargs).eval()
            if not load_in_4bit and not load_in_8bit:
                self.model = self.model.to(device)
        elif 'share' in model_name.lower():
            name = get_model_name_from_path(model_path)
            tokenizer, self.model, image_processor, _ = load_pretrained_model(model_path, None, name)
            self.processor = (tokenizer, image_processor)
        else:
            self.model = info['model']()
            self.processor = None
        self.model_fn = model_regisiter[model_name]
    
    def __call__(self, text:str, images:list[Image.Image]):
        return self.model_fn(text=text, images=images, model=self.model, processor=self.processor)

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
    
    def split_intervl_model(self, model_name):
        # Copy from https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
            'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
        return device_map

    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device