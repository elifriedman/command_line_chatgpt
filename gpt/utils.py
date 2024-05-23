import re
import os
from pathlib import Path
from colorama import Fore, Style


base_path = Path(os.path.expanduser("~/.gpt"))
base_path.mkdir(exist_ok=True)

def load_dotenv(env_path=Path(__file__).parent / ".env"):
    if not Path(env_path).exists():
        print(
            f"{Fore.RED}!!ERROR!!{Style.RESET_ALL} Please put create a file called `~/.gpt/.env` file with your OpenAI key: OPENAI_API_KEY=sk..."
        )
        exit(1)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            key, value = line.split("=")
            os.environ[key] = value

def split_text_around_image_tags(s):
    # Define the regex pattern to match /image=<image_url_or_path>
    pattern = r"/image=(?:'(?P<single_quoted>[^']*)'|\"(?P<double_quoted>[^\"]*)\"|(?P<unquoted>[^\s]+))"

    # Find all matches and their positions
    matches = list(re.finditer(pattern, s))

    # Initialize the result list and the start position for the next text segment
    result = []
    last_end = 0

    for match in matches:
        text = s[last_end:match.start()]
        if text.strip() != "":
            result.append(("text", text))

        image_path = match.group('single_quoted') or match.group('double_quoted') or match.group('unquoted')
        result.append(("image_url", image_path))
        last_end = match.end()

    result.append(("text", s[last_end:]))

    return result