# coding: utf-8
import os
import json
import base64
import io
from enum import Enum
from typing import Union
from dataclasses import dataclass
from openai import OpenAI
from gpt.gpt import base_path
from PIL import Image

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def save_json(obj, f):
    with open(f, 'w') as f:
        json.dump(obj, f)
        
class Size(Enum):
    SQUARE = "1024x1024"
    VERTICAL = "1792x1024"
    HORIZONTAL = "1024x1792"


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

def generate_dalle_image(prompt, quality: Union[Quality, str] = Quality.HD, size: Union[Size, str] = Size.SQUARE, style: Union[Style, str] = Style.VIVID):
    if isinstance(quality, Quality):
        quality = quality.value
    if isinstance(size, Size):
        size = size.value
    if isinstance(style, Style):
        style = style.value
    response = client.images.generate(
        model="dall-e-3",
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

def run_dalle(prompt, output_path, quality: Union[Quality, str] = Quality.HD, size: Union[Size, str] = Size.SQUARE, style: Union[Style, str] = Style.VIVID):
    result = generate_dalle_image(prompt, quality, size, style)
    result.image.save(output_path)
    output_json_path = output_path[:output_path.rfind(".")]
    save_json({"prompt": result.prompt, "revised_prompt": result.revised_prompt}, f"{output_json_path}.json")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--output", "-o")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    des
