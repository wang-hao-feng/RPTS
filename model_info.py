import os
from functools import partial

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from ShareGPT4V.share4v.model import Share4VLlamaForCausalLM

models_param_path = os.environ['MODEL_PATH']

model_info = {
    "GPT4o": {
        "path": ".", 
        "processor_path": ".", 
        "model": lambda: OpenAI(api_key=os.getenv('OPENAI_API_KEY')), 
    }, 
    # InstructBLIP
    "InstructBLIP-vicuna-13b": {
        "repo_id": "Salesforce/instructblip-vicuna-13b", 
        "path": os.path.join(models_param_path, "instructblip-vicuna-13b"), 
        "processor_path": os.path.join(models_param_path, "instructblip-vicuna-13b"), 
        "processor": InstructBlipProcessor.from_pretrained, 
        "model": InstructBlipForConditionalGeneration.from_pretrained,
    }, 
    # Llava-v1.5
    "Llava-v1.5-13B": {
        "repo_id": "llava-hf/llava-1.5-13b-hf", 
        "path": os.path.join(models_param_path, "llava-1.5-13b-hf"), 
        "processor_path": os.path.join(models_param_path, "llava-1.5-13b-hf"), 
        "processor": LlavaProcessor.from_pretrained, 
        "model": LlavaForConditionalGeneration.from_pretrained, 
    }, 
    # InternVL2
    "InternVL2-Llama3-76B": {
        "repo_id": "OpenGVLab/InternVL2-Llama3-76B", 
        "path": os.path.join(models_param_path, "intervl2-llama3-76b"), 
        "processor_path": os.path.join(models_param_path, "intervl2-llama3-76b"), 
        "processor": partial(AutoTokenizer.from_pretrained, trust_remote_code=True), 
        "model": partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True), 
    }, 
    # ShateGPT4V
    "ShareGPT4V-13B": {
        "repo_id": "Lin-Chen/ShareGPT4V-13B", 
        "path": os.path.join(models_param_path, "sharegpt4v-13b"), 
        "processor_path": os.path.join(models_param_path, "sharegpt4v-13b"), 
        "processor": AutoTokenizer.from_pretrained, 
        "model": Share4VLlamaForCausalLM.from_pretrained, 
    }, 
    "ShareGPT4V-Vision": {
        "repo_id": "Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12", 
        "path": os.path.join(models_param_path, "sharegpt4v-13b-pretrained_vit-large336-l12"), 
    }, 
    # Llava-Next
    "Llava-Next-34B": {
        "repo_id": "llava-hf/llava-v1.6-34b-hf", 
        "path": os.path.join(models_param_path, "llava-v1.6-34b-hf"), 
        "processor_path": os.path.join(models_param_path, "llava-v1.6-34b-hf"), 
        "processor": LlavaNextProcessor.from_pretrained, 
        "model": LlavaNextForConditionalGeneration.from_pretrained, 
    }, 
    # Qwen-VL
    "Qwen-VL-Chat": {
        "repo_id": "Qwen/Qwen-VL-Chat", 
        "path": os.path.join(models_param_path, "Qwen-VL-Chat"), 
        "processor_path": os.path.join(models_param_path, "Qwen-VL-Chat"), 
        "processor": partial(AutoTokenizer.from_pretrained, trust_remote_code=True), 
        "model": partial(AutoModelForCausalLM.from_pretrained, trust_remote_code=True), 
    }, 
}