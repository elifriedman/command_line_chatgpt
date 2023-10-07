# ChatGPT

This is a tool for controlling ChatGPT from the command line.

![example.png](https://github.com/elifriedman/command_line_chatgpt/blob/main/example.png)

# Setup

## Install

### From pip

```
pip install --upgrade 'git+https://github.com/elifriedman/command_line_chatgpt.git'
```

### Environment
or you can git clone it and install the requirements

```
git clone https://github.com/elifriedman/command_line_chatgpt.git
cd command_line_chatgpt
pip install .
```

## API Key
Create a file `~/.gpt/.env` and put in your [OpenAI API Key](https://platform.openai.com/account/api-keys):
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
- `--instructions_path`, `-i`: filepath for initial ChatGPT instruction prompt (default instructions.txt). See https://github.com/f/awesome-chatgpt-prompts for inspiration or an instruction string
- `--temperature`: temperature value for generating text
- `--max_tokens`: maximum number of tokens to generate
- `--frequency_penalty`: frequency penalty value for generating text
- `--presence_penalty`: presence penalty value for generating text
- `--max_contexts`: maximum number of questions to include in prompt

