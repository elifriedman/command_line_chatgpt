import io
import os
import requests
import base64
import mimetypes
import PIL
from pathlib import Path
from PIL import Image
from openai import OpenAI
from .utils import load_dotenv, base_path
import numpy as np

load_dotenv(env_path=base_path / ".env")


def image_to_base64(image):
    is_pil = isinstance(image, Image.Image)
    image = image if is_pil else Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = "data:image/png;base64," + str(base64.b64encode(buffered.getvalue()), "utf-8")
    return img_base64


def encode_image_from_path(image_path):
    image_path = os.path.expanduser(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Could not determine the MIME type of the file {image_path}.")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_image}"


def validate_image_url(url):
    try:
        response = requests.head(url)
        if response.status_code != 200:
            raise ValueError(f"The URL {url} is not reachable.")
    except requests.RequestException as e:
        raise ValueError(f"An error occurred while validating the URL: {e}")


def is_image(path):
    try:
        Image.open(path)
        return True
    except (FileNotFoundError, PIL.UnidentifiedImageError):
        return False


def create_image_content(content):
    if isinstance(content, Image.Image) or isinstance(content, np.ndarray):
        content = image_to_base64(content)
    elif type(content) in [str, Path] and is_image(os.path.expanduser(content)):
        content = encode_image_from_path(content)
    elif isinstance(content, str) and content.startswith("http"):
        validate_image_url(content)
    else:
        raise ValueError(f"input not a valid image type: {content}")
    return content



class ImageChatSession:
    def __init__(self, model="gpt-4-vision-preview", max_tokens=300):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.messages = []

    def create_chat_completion(self, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=self.max_tokens,
                **kwargs,
            )
            return response
        except Exception as e:
            print(f"An error occurred while creating the chat completion: {e}")
            raise

    def add_message(self, content, role="user", index=None):
        message_type = "text"
        try:
            parsed_content = self.create_image_content(content)
            content = {"url": parsed_content}
            message_type = "iamge_url"
        except ValueError:
            pass
        if index is None:
            self.messages.append(
                {
                    "role": role,
                    "content": [{"type": message_type, message_type: content}],
                }
            )
        else:
            self.messages[index] = {
                "role": role,
                "content": [{"type": message_type, message_type: content}],
            }
