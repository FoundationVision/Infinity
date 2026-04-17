import json
import os
import os.path as osp
import argparse
import sys
import re
import http.client
import base64
import tqdm
import random
import traceback
import time
from io import BytesIO

from PIL import Image
import openai

from tools.llm_provider import chat_completion, get_llm_client


def encode_image(image_path, size=(512, 512)):
    """
    Resize an image and encode it as a Base64 string.

    Args:
    - image_path (str): Path to the image file.
    - size (tuple): New size as a tuple, (width, height).

    Returns:
    - str: Base64 encoded string of the resized image.
    """
    if size is None:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.ANTIALIAS)
        img_buffer = BytesIO()
        img_resized.save(img_buffer, format=img.format)
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.read()).decode("utf-8")


SYSTEM = """
You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say.
For example, outputting the prompt and parameters like "<prompt:a beautiful morning in the woods with the sun peaking through the trees><cfg:3>" will trigger your partner bot to output an image of a forest morning, as described.
You will be prompted by users looking to create detailed, amazing images. The way to accomplish this is to refine their short prompts and make them extremely detailed and descriptive.
- You will only ever output a single image description sentence per user request.
- Each image description sentence should be consist of "<prompt:xxx><cfg:xxx>", where <prompt:xxx> is the image description, <cfg:xxx> is the parameter that control the image generation.
Here are the guidelines to generate image description <prompt:xxx> :
- Refine users' prompts and make them extremely detailed and descriptive but keep the meaning unchanged (very important).
- For particularly long users' prompts (>50 words), they can be outputted directly without refining. Image descriptions must be between 8-512 words. Extra words will be ignored.
- If the user's prompt requires rendering text, enclose the text with single quotation marks and prefix it with "the text".
Here are the guidelines to set <cfg:xxx> :
- Please first determine whether the image to be generated based on the user prompt is likely to contain a clear face. If it does, set <cfg:1>; if not, set <cfg:3>.
"""

FEW_SHOT_HISTORY = [
    {"role": "user", "content": "a tree"},
    {"role": "assistant", "content": "<prompt:A photo of a majestic oak tree stands proudly in the middle of a sunlit meadow, its branches stretching out like welcoming arms. The leaves shimmer in shades of vibrant green, casting dappled shadows on the soft grass below.><cfg:3>"},
    {"role": "user", "content": "a young girl with red hair"},
    {"role": "assistant", "content": "<prompt:A young girl with vibrant red hair, close-up face, in the style of hyper-realistic portraiture, warm and inviting atmosphere, soft lighting, freckles, vintage effect><cfg:1>"},
    {"role": "user", "content": "a man, close-up"},
    {"role": "assistant", "content": "<prompt:close-up portrait of a young man with freckles and curly hair, in the style of chiaroscuro, strong light and shadow contrast, intense gaze, background fades into darkness><cfg:1>"},
    {"role": "user", "content": "Generate Never Stop Learning"},
    {"role": "assistant", "content": "<prompt:Generate an image with the text 'Never Stop Learning' in chalkboard style.><cfg:3>"},
]

class PromptRewriter(object):
    def __init__(self, system, few_shot_history):
        if not system:
            system = SYSTEM
        if not len(few_shot_history):
            few_shot_history = FEW_SHOT_HISTORY
        self.system = [{"role": "system", "content": system}]
        self.few_shot_history = few_shot_history

    def rewrite(self, prompt):
        messages = self.system + self.few_shot_history + [{"role": "user", "content": prompt}]
        result, _ = get_gpt_result(messages=messages, retry=5, return_json=False)
        assert result
        return result


def get_gpt_result(model_name=None, messages=None, retry=5, ak=None, return_json=False):
    """
    Retrieves a chat response using the configured LLM provider.

    The provider is determined by environment variables (see tools/llm_provider.py).
    Supports OpenAI, Azure OpenAI, MiniMax, and any OpenAI-compatible API.

    Args:
        model_name (str, optional): The model to use. If None, uses provider default.
        messages (list): Chat messages.
        retry (int): Number of retry attempts on failure.
        ak (str, optional): API key override (deprecated, use env vars instead).
        return_json (bool): Whether to request JSON response format.

    Returns:
        tuple: (response_content, None) on success, (None, -1) on failure.
    """
    for i in range(retry):
        try:
            result = chat_completion(
                messages=messages,
                model=model_name,
                api_key=ak if ak else None,
                return_json=return_json,
            )
            return result, None
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, KeyboardInterrupt):
                exit(0)
            sleep_time = 10 + random.randint(2, 5) ** (i + 1)
            time.sleep(sleep_time)
    return None, -1

if __name__ == '__main__':
    times = 0
    prompt_list = []

    var_t2i_prompt_rewriter = PromptRewriter(system='', few_shot_history=[])

    prompt_list = [
        'a tree',
        'two dogs',
        'an oil painting of a house',
        'a Chinese model sits in the train. Magazine style',
        'two girls',
        'countryside',
        'a rabbit fights with a tiger',
        'a beach in Hawaii',
    ]

    for prompt in prompt_list:
        times += 1
        result = var_t2i_prompt_rewriter.rewrite(prompt)
        print(f'prompt: {prompt}, result: {result}')
