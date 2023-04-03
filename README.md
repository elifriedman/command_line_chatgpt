# Big News!

# Command Line ChatGPT Bot

This is a simple chat-bot that uses the OpenAI ChatGPT API.

# Setup

Make sure you have python3 installed:

```
python3 --version
```

Create a virtual environment and install the dependencies:

# Configuration

Copy `env.sample` to `.env` and add your OpenAI API key to the file.

```
OPENAI_API_KEY=<<YOUR_API_KEY>>
```

# ChatGPT

This is a command-line tool for controlling ChatGPT, a large language model trained by OpenAI. ChatGPT can generate human-like text in response to prompts.

# Setup
Install the requirements
```
pip install -r requirements.txt
```

Create a file `.env` and put in your [OpenAI API Key](https://platform.openai.com/account/api-keys):
```
OPENAI_API_KEY=<your OPENAI API Key>
```

If you want, add an entry to your .bashrc file
```
echo alias gpt="$(pwd)/gpt.py" >> ~/.bashrc
```

## Usage

```
usage: gpt.py [-h] [--instructions_path INSTRUCTIONS_PATH] [--temperature TEMPERATURE] [--max_tokens MAX_TOKENS]
              [--frequency_penalty FREQUENCY_PENALTY] [--presence_penalty PRESENCE_PENALTY]
              [--max_contexts MAX_CONTEXTS]
```

## Arguments

- `-h`, `--help`: show help message and exit
- `--instructions_path`, `-i`: filepath for initial ChatGPT instruction prompt (default instructions.txt). See https://github.com/f/awesome-chatgpt-prompts for inspiration
- `--temperature`: temperature value for generating text
- `--max_tokens`: maximum number of tokens to generate
- `--frequency_penalty`: frequency penalty value for generating text
- `--presence_penalty`: presence penalty value for generating text
- `--max_contexts`: maximum number of questions to include in prompt

