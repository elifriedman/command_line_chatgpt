#!/home/eli/workspace/gpt/venv/bin/python
import argparse
import sys
import os
import openai
import importlib
import json

from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
from prompt_toolkit import PromptSession, ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from colorama import Fore, Back, Style
from pathlib import Path
from enum import Enum
from .vision import create_image_content


def save_json(obj, f, mode="w"):
    with open(f, mode) as f:
        return json.dump(obj, f)


def load_json(f):
    with open(f) as f:
        return json.load(f)


# load values from the .env file if it exists
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


base_path = Path(os.path.expanduser("~/.gpt"))
base_path.mkdir(exist_ok=True)
load_dotenv(env_path=base_path / ".env")

MODEL_MAP = {
    "haiku": "claude-3-haiku-20240307",
    "sonnet": "claude-3-sonnet-20240229",
    "opus": "claude-3-opus-20240229",
    "claude": "claude-3-opus-20240229",
    "gpt-4": "gpt-4o",
    "gpt4": "gpt-4o",
    "gpt": "gpt-4o",
}


def read_instructions(path):
    if not os.path.exists(path):
        return path
    with open(path) as f:
        return f.read().strip()


def read_function_list(path):
    return json.load(path)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Context:
    def __init__(self, instructions, max_contexts: int = 100):
        self.instructions = instructions
        self.max_contexts = max_contexts
        self._context = []

    def reset(self):
        self._context = []

    def make_context_item(self, content, role: Role, **kwargs):
        return {"role": role.value, "content": content, **kwargs}

    def add(self, content, role: Role, **kwargs):
        new_context = self.make_context_item(content=content, role=role, **kwargs)
        self._context.append(new_context)

    @property
    def context(self):
        return self.get_contexts(num_contexts=self.max_contexts)

    def get_contexts(self, num_contexts: int = None):
        start_index = num_contexts
        end_index = None
        if num_contexts is None:
            start_index = 0
        elif num_contexts == 0:
            start_index = 0
            end_index = 0
        system_prompt = self.make_context_item(content=self.instructions, role=Role.SYSTEM)
        contexts = self._context[-start_index:end_index]
        return [system_prompt] + contexts

    def to_dict(self):
        return {
            "instructions": self.instructions,
            "contexts": self._context,
            "max_contexts": self.max_contexts,
        }

    @classmethod
    def from_dict(cls, data):
        out = cls(instructions=data["instructions"], max_contexts=data["max_contexts"])
        out._context = data["contexts"]
        return out

    def save(self, path, model: str = None):
        data = self.to_dict()
        if model is not None:
            data["model"] = model
        save_json(data, path)

    @classmethod
    def load(cls, path):
        data = load_json(path)
        return cls.from_dict(data)

    def get_response(
        self,
        new_question="",
        model: str = "gpt-4o",
        temperature: float = 0.5,
        max_tokens: int = 1000,
        frequency_penalty: float = 0,
        presence_penalty: float = 0.6,
        seed: int = None,
        max_contexts: int = None,
        json_output: bool = False,
        **kwargs,
    ):
        # build the messages
        has_new_question = new_question != ""
        if has_new_question:
            self.add(content=new_question, role=Role.USER)
        try:
            if max_contexts is not None:
                old_max = self.max_contexts
                self.max_contexts = max_contexts
            completion, finish_reason = run_gpt(
                context=self,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                json_output=json_output,
                **kwargs,
            )
            if max_contexts is not None:
                self.max_contexts = old_max
        except openai.RateLimitError as exc:
            print(
                Fore.RED
                + Style.BRIGHT
                + "You're going too fast! Error: "
                + str(exc)
                + Style.RESET_ALL
            )
            return ""

        if finish_reason == "function_call":
            function_info = completion.choices[0].message.function_call
            response = self.handle_function_call(function_info)
            self.add(content=response, role=Role.FUNCTION, name=function_info["name"])
        elif finish_reason == "length":
            current_response = completion
            self.add(content=current_response, role=Role.ASSISTANT)
            response = f"{current_response}||cutoff {max_tokens}||"
        else:
            response = completion
            self.add(content=response, role=Role.ASSISTANT)
        return response


def run_gpt(
    context: Context,
    temperature: float = 0.5,
    max_tokens: int = 1000,
    frequency_penalty: float = 0,
    presence_penalty: float = 0.6,
    model: str = "gpt-4o",
    seed: int = None,
    api_key: str = None,
    json_output: bool = False,
    **kwargs,
):
    if model in MODEL_MAP:
        model = MODEL_MAP[model]
    if "gpt" in model:
        service = "openai"
    elif "claude" in model:
        service = "anthropic"
    else:
        service = "ollama"
    if json_output is True:
        kwargs["response_format"] = {"type": "json_object"}
    messages = context.context
    if service == "openai":
        api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        Response = client.chat.completions.create
        kwargs["frequency_penalty"] = frequency_penalty
        kwargs["presence_penalty"] = presence_penalty
        kwargs["seed"] = seed
    elif service == "anthropic":
        api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)
        Response = client.messages.create
        has_system = [i for i in range(len(messages)) if messages[i]["role"] == "system"]
        if len(has_system) > 0:
            system_prompt = messages.pop(has_system[0])["content"]
            kwargs["system"] = system_prompt
    else:
        api_key = "ollama"
        client = OpenAI(
            api_key=api_key, base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        )
        Response = client.chat.completions.create

    if "functions" in kwargs and not kwargs["functions"]:
        kwargs.pop("functions")
        kwargs.pop("function_call", None)
    out = Response(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    if service == "openai" or service == "ollama":
        stop_reason = out.choices[0].finish_reason
        content = out.choices[0].message.content
    elif service == "anthropic":
        stop_reason = out.stop_reason
        content = out.content[0].text
    return content, stop_reason


class QuestionAnswer:
    def __init__(
        self,
        instructions,
        temperature: float = 0.5,
        max_tokens: int = 1000,
        frequency_penalty: float = 0,
        presence_penalty: float = 0.6,
        max_contexts: int = 100,
        model: str = "gpt-4o",
        function_path: Path = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_contexts = max_contexts
        self.context = Context(instructions, max_contexts=max_contexts)
        self.model = model
        self.function_path = function_path
        self.functions = None
        if self.function_path is not None:
            self.functions = load_json(function_path / "function_list.json")
            if str(self.function_path) not in sys.path:
                sys.path.append(str(self.function_path))

    def call_method_from_file(self, file_name, function_name, args):
        try:
            module = importlib.import_module(file_name)
            method = getattr(module, function_name)
            result = method(**args)
            if isinstance(result, dict):
                result = json.dumps(result)

            return str(result)
        except Exception as e:
            # Handle any exceptions that occur during the process
            return f"Error running function '{function_name}' from file '{file_name}' with args {args}: {str(e)}"

    def handle_function_call(self, function_info):
        function_name = function_info["name"]
        arguments = function_info["arguments"]
        try:
            arguments = json.loads(arguments)
        except Exception as exc:
            return f"Error calling `{function_name}`. Problem parsing json '{arguments}' {exc}"
        return self.call_method_from_file(
            file_name="functions", function_name=function_name, args=arguments
        )

    def get_messages(self, new_question="", add_to_context: bool = True):
        has_new_question = new_question != ""
        if has_new_question and add_to_context is True:
            self.context.add(content=new_question, role=Role.USER)
            messages = self.context.context
        elif has_new_question and add_to_context is False:
            messages = self.context.context
            messages.append(self.context.make_context_item(content=new_question, role=Role.USER))
        else:
            messages = self.context.context
        return messages

    def get_response(self, new_question, add_to_context: bool = True, **kwargs):
        # build the messages
        messages = self.get_messages(new_question, add_to_context=add_to_context)

        try:
            completion, finish_reason = run_gpt(
                context=self.context,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                functions=self.functions,
                function_call="auto",
                **kwargs,
            )
        except openai.RateLimitError as exc:
            print(Fore.RED + Style.BRIGHT + "RateLimitError! " + str(exc) + Style.RESET_ALL)
            return ""
        except openai.NotFoundError as exc:
            print(Fore.RED + Style.BRIGHT + "NotFoundError! " + str(exc) + Style.RESET_ALL)
            return ""

        if finish_reason == "function_call":
            function_info = completion.choices[0].message.function_call
            response = self.handle_function_call(function_info)
            if add_to_context:
                self.context.add(content=response, role=Role.FUNCTION, name=function_info["name"])
        elif finish_reason == "length":
            print(Fore.RED + "RESPONSE REACHED MAX LENGTH: {self.max_tokens}" + Style.RESET_ALL)
            response = completion
            if add_to_context:
                self.context.add(content=response, role=Role.ASSISTANT)
            # current_response = completion.choices[0].message.content
            # self.context.add(content=current_response, role=Role.ASSISTANT)
            # new_response = self.get_response(new_question=f"Continue your previous response. DO NOT SAY 'Sure, I'll continue' or anything of the sort, just continue from: \"{current_response[-40:]}...\"")
            # response = current_response + new_response
        else:
            response = completion
            if add_to_context:
                self.context.add(content=response, role=Role.ASSISTANT)
        return response

    def to_dict(self):
        return {
            "instructions": self.context.instructions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_contexts": self.max_contexts,
            "context": self.context.to_dict(),
            "model": self.model,
            "function_path": str(self.function_path) if self.function_path else None,
        }

    @classmethod
    def from_dict(cls, data):
        qa = cls(
            instructions=data["instructions"],
            temperature=data["temperature"],
            max_tokens=data["max_tokens"],
            presence_penalty=data["presence_penalty"],
            frequency_penalty=data["frequency_penalty"],
            max_contexts=data["max_contexts"],
            model=data["model"],
            function_path=Path(data["function_path"]) if data["function_path"] else None,
        )
        qa.context = Context.from_dict(data["context"])
        return qa

    def save(self, path):
        save_json(self.to_dict(), path)


def run(
    instructions: str,
    question: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: float = 0.6,
    max_contexts: int = 10,
    model: str = "gpt-4o",
):
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        model=model,
    )
    response = question_answer.get_response(question)
    return response


def save_conversation_history(text, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text + "\n")


def run_iteratively(
    instructions: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: float = 0.6,
    max_contexts: int = 100,
    model: str = "gpt-4o",
    history_path: Path = Path(os.path.expanduser("~/.gpt/history.txt")),
    conversation_name: str = None,
    load_conversation_path: str = None,
):
    if conversation_name is None:
        conversation_name = str(datetime.now()).replace(" ", "_").replace(":", "-").split(".")[0]
        conversation_path = history_path.parent / "conversations" / conversation_name
        conversation_path.parent.mkdir(parents=True, exist_ok=True)
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        model=model,
    )
    if load_conversation_path is not None:
        question_answer.context = Context.load(load_conversation_path)
    question_answer.context.save(conversation_path)

    # keep track of previous questions and answers
    save_conversation_history(
        f"----- New Conversation ({model}) -----" f"\n{instructions}\n----------------------------",
        history_path,
    )
    question_answer.save(conversation_path)
    question_holder_hack = {"question": ""}

    kb = KeyBindings()

    @kb.add("tab", "enter")
    def _(event):
        "When Control+Enter is pressed, accept input."
        question_holder_hack["question"] = event.app.current_buffer.text
        event.app.current_buffer.append_to_history()
        event.app.exit()

    prompt_session = PromptSession(
        ANSI(Fore.GREEN + "Enter your prompt and then press <tab>-<enter>:\n" + Style.RESET_ALL),
        key_bindings=kb,
        history=FileHistory(os.path.expanduser("~/.gpt/input_history.txt")),
        enable_history_search=True,
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        prompt_continuation="",
    )
    while True:
        try:
            prompt_session.prompt()
        except KeyboardInterrupt:
            return
        new_question = question_holder_hack["question"]
        print(Fore.CYAN + "Processing..." + Style.RESET_ALL)
        # ask the user for their question
        # check the question is safe
        save_conversation_history(f">>>>>\n{new_question}", history_path)
        response = question_answer.get_response(new_question)
        save_conversation_history(f"<<<<<\n{response}", history_path)
        print(response)
        question_answer.context.save(conversation_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for controlling ChatGPT")
    parser.add_argument(
        "--load",
        "-l",
        type=str,
        help="Load a conversation",
    )
    parser.add_argument(
        "--instructions",
        "-i",
        type=str,
        default=os.path.expanduser("~/workspace/gpt/prompts/default_prompt.txt"),
        help="Filepath for initial ChatGPT instruction prompt (default ~/.gpt/default_prompt.txt). See https://github.com/f/awesome-chatgpt-prompts for inspiration",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.5,
        help="Temperature value for generating text",
    )
    parser.add_argument(
        "--max_tokens",
        "-n",
        type=int,
        default=2000,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--frequency_penalty",
        "-f",
        type=float,
        default=0,
        help="Frequency penalty value for generating text",
    )
    parser.add_argument(
        "--presence_penalty",
        "-p",
        type=float,
        default=0.6,
        help="Presence penalty value for generating text",
    )
    parser.add_argument(
        "--max_contexts",
        "-c",
        type=int,
        default=10,
        help="Maximum number of questions to include in prompt",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o",
        help="Which chatgpt model to use",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_iteratively(
        instructions=read_instructions(args.instructions),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        max_contexts=args.max_contexts,
        model=args.model,
        load_conversation_path=args.load,
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
