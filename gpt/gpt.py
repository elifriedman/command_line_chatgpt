#!/home/eli/workspace/gpt/venv/bin/python
import sys
import os
import openai
from openai import OpenAI
import argparse
from prompt_toolkit import PromptSession, ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from colorama import Fore, Back, Style
from pathlib import Path
import importlib
import json
from enum import Enum


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
            key, value = line.split("=")
            os.environ[key] = value


base_path = Path(os.path.expanduser("~/.gpt"))
base_path.mkdir(exist_ok=True)
load_dotenv(env_path=base_path / ".env")


def read_instructions(path):
    if not os.path.exists(path):
        return path
    with open(path) as f:
        return f.read()


def read_function_list(path):
    return json.load(path)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Context:
    def __init__(self, instructions, max_contexts: int = 100, context_file: str = None):
        self.instructions = instructions
        self.context_file = context_file
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
        if num_contexts is None:
            num_contexts = 0
        system_prompt = self.make_context_item(content=self.instructions, role=Role.SYSTEM)
        contexts = self._context[-num_contexts:]
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

    def save(self, path):
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path):
        data = load_json(path)
        return cls.from_dict(data)

    def get_response(
        self,
        new_question="",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.5,
        max_tokens: int = 500,
        frequency_penalty: float = 0,
        presence_penalty: float = 0.6,
        seed: int = None,
        max_contexts: int = None
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
            completion = run_gpt(
                context=self,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                **kwargs,
            )
            if max_contexts is not None:
                self.max_contexts = old_max
        except openai.RateLimitError as exc:
            print(Fore.RED + Style.BRIGHT + "You're going too fast! Error: " + str(exc) + Style.RESET_ALL)
            return ""

        if completion.choices[0].finish_reason == "function_call":
            function_info = completion.choices[0].message.function_call
            response = self.handle_function_call(function_info)
            self.add(content=response, role=Role.FUNCTION, name=function_info["name"])
        elif completion.choices[0].finish_reason == "length":
            current_response = completion.choices[0].message.content
            self.add(content=current_response, role=Role.ASSISTANT)
            response = f"{current_response}||cutoff {max_tokens}||"
        else:
            response = completion.choices[0].message.content
            self.add(content=response, role=Role.ASSISTANT)
        return response


def run_gpt(
    context: Context,
    temperature: float = 0.5,
    max_tokens: int = 500,
    frequency_penalty: float = 0,
    presence_penalty: float = 0.6,
    model: str = "gpt-3.5-turbo",
    seed: None
    **kwargs,
):
    if model == "gpt-4":
        model = "gpt-4-1106-preview"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = context.context
    if "functions" in kwargs and not kwargs["functions"]:
        kwargs.pop("functions")
        kwargs.pop("function_call", None)
    out = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        **kwargs,
    )
    return out


class QuestionAnswer:
    def __init__(
        self,
        instructions,
        temperature: float = 0.5,
        max_tokens: int = 500,
        frequency_penalty: float = 0,
        presence_penalty: float = 0.6,
        max_contexts: int = 10,
        context_file: str = None,
        model: str = "gpt-3.5-turbo",
        function_path: Path = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_contexts = max_contexts
        self.context = Context(instructions, max_contexts=max_contexts, context_file=context_file)
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
        return self.call_method_from_file(file_name="functions", function_name=function_name, args=arguments)

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
            completion = run_gpt(
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
            print(Fore.RED + Style.BRIGHT + "You're going too fast! Error: " + str(exc) + Style.RESET_ALL)
            return ""

        if completion.choices[0].finish_reason == "function_call":
            function_info = completion.choices[0].message.function_call
            response = self.handle_function_call(function_info)
            if add_to_context:
                self.context.add(content=response, role=Role.FUNCTION, name=function_info["name"])
        elif completion.choices[0].finish_reason == "length":
            print(Fore.RED + "RESPONSE REACHED MAX LENGTH: {self.max_tokens}" + Style.RESET_ALL)
            response = completion.choices[0].message.content
            if add_to_context:
                self.context.add(content=response, role=Role.ASSISTANT)
            # current_response = completion.choices[0].message.content
            # self.context.add(content=current_response, role=Role.ASSISTANT)
            # new_response = self.get_response(new_question=f"Continue your previous response. DO NOT SAY 'Sure, I'll continue' or anything of the sort, just continue from: \"{current_response[-40:]}...\"")
            # response = current_response + new_response
        else:
            response = completion.choices[0].message.content
            if add_to_context:
                self.context.add(content=response, role=Role.ASSISTANT)
        return response


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """
    return None

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [error for category, error in errors.items() if response.results[0].categories[category]]
        return result
    return None


def run(
    instructions: str,
    question: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: float = 0.6,
    max_contexts: int = 10,
    context_file: str = None,
    model: str = "gpt-3.5-turbo",
):
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        context_file=context_file,
        model=model,
    )
    response = question_answer.get_response(question)
    return response


def save_conversation(text, filepath):
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
    max_contexts: int = 10,
    context_file: str = None,
    model: str = "gpt-3.5-turbo",
    filepath: Path = Path(os.path.expanduser("~/.gpt/history.txt")),
):
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        context_file=context_file,
        model=model,
    )

    # keep track of previous questions and answers
    save_conversation(
        f"----- New Conversation ({model}) -----" f"\n{instructions}\n----------------------------",
        filepath,
    )
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
        errors = get_moderation(new_question)
        if errors:
            print(Fore.RED + Style.BRIGHT + "Sorry, your question didn't pass the moderation check:")
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        save_conversation(f">>>>>\n{new_question}", filepath)
        response = question_answer.get_response(new_question)
        save_conversation(f"<<<<<\n{response}", filepath)
        print(response)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for controlling ChatGPT")
    parser.add_argument(
        "--instructions",
        "-i",
        type=str,
        default=os.path.expanduser("~/.gpt/default_prompt.txt"),
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
        default=500,
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
        default="gpt-3.5-turbo",
        help="Which chatgpt model to use",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = args.model
    if model == "gpt-4":
        model = "gpt-4-1106-preview"
    run_iteratively(
        instructions=read_instructions(args.instructions),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        max_contexts=args.max_contexts,
        context_file=None,
        model=model,
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
