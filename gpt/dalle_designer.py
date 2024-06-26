# coding: utf-8
import argparse
import os
import json
import base64
import io
from enum import Enum
from typing import Union
from dataclasses import dataclass
from openai import OpenAI
from PIL import Image
import numpy as np
from .gpt import base_path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def save_json(obj, f, pretty: bool=True):
    indent = 4 if pretty is True else None
    with open(f, 'w') as f:
        json.dump(obj, f, indent=indent)
        
class Size(Enum):
    SQUARE_SMALL = "512x512"
    SQUARE = "1024x1024"
    HORIZONTAL = "1792x1024"
    VERTICAL = "1024x1792"


class Style(Enum):
    VIVID = "vivid"
    NATURAL = "natural"

class Quality(Enum):
    HD = "hd"
    STANDARD = "standard"

@dataclass
class DalleResult:
    image: Image.Image
    prompt: str
    revised_prompt: str

def generate_dalle_image(prompt, model: str = "dall-e-3", quality: Union[Quality, str] = Quality.HD, size: Union[Size, str] = Size.SQUARE, style: Union[Style, str] = Style.VIVID):
    if isinstance(quality, Quality):
        quality = quality.value
    if isinstance(size, Size):
        size = size.value
    if isinstance(style, Style):
        style = style.value
    response = client.images.generate(
        model=model,
        size=size,
        quality=quality,
        style=style,
        n=1,
        prompt=prompt,
        response_format="b64_json"
    )
    revised_prompt = response.data[0].revised_prompt
    image_str = response.data[0].b64_json
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    return DalleResult(image, prompt=prompt, revised_prompt=revised_prompt)

def img2bytes(img):
    b = io.BytesIO()
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(b, format="PNG")
    return b

def edit_dalle_image(prompt, image, mask, model: str = "dall-e-2",  size: Union[Size, str] = Size.SQUARE):
    if isinstance(size, Size):
        size = size.value
    if isinstance(image, Image.Image) or isinstance(image, np.ndarray):
        image = img2bytes(image)
    if isinstance(mask, Image.Image) or isinstance(mask, np.ndarray):
        mask = img2bytes(mask)
    response = client.images.edit(
        model=model,
        image=image,
        mask=mask,
        size=size,
        n=1,
        prompt=prompt,
        response_format="b64_json"
    )
    image_str = response.data[0].b64_json
    image_bytes = base64.b64decode(image_str)
    image = Image.open(io.BytesIO(image_bytes))
    return DalleResult(image, prompt=prompt, revised_prompt=None)

def run_dalle(prompt, output_path, quality: Union[Quality, str] = Quality.HD, size: Union[Size, str] = Size.SQUARE, style: Union[Style, str] = Style.VIVID):
    result = generate_dalle_image(prompt, quality=quality, size=size, style=style)
    result.image.save(output_path)
    output_json_path = output_path[:output_path.rfind(".")]
    save_json({"prompt": result.prompt, "revised_prompt": result.revised_prompt}, f"{output_json_path}.json")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("output")
    parser.add_argument("--size", "-s", default=Size.SQUARE, choices=[s.value for s in Size])
    parser.add_argument("--quality", "-q", default=Quality.HD, choices=[s.value for s in Quality])
    parser.add_argument("--style", "-y", default=Style.VIVID, choices=[s.value for s in Style])

    return parser.parse_args()

def main():
    args = parse_args()
    run_dalle(args.prompt, args.output, quality=args.quality, size=args.size, style=args.style)

if __name__ == "__main__":
    main()
