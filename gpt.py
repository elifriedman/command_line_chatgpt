#!/home/eli/workspace/gpt/venv/bin/python
import os
import openai
import argparse
import readline
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


def read_instructions(path):
    if not os.path.exists(path):
        print("Instruction path does not exist")
        return ""
    with open(path) as f:
        return f.read()


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
    ):
        self.instructions = instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_contexts = max_contexts
        self.context_file = context_file
        self.context = []

    def assemble_messages(self, new_question):
        messages = [
            {"role": "system", "content": self.instructions},
        ]
        # add the previous questions and answers
        for question, answer in self.context[-self.max_contexts :]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        # add the new question
        messages.append({"role": "user", "content": new_question})
        return messages

    def get_response(self, new_question):
        # build the messages
        messages = self.assemble_messages(new_question)
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
        except openai.error.RateLimitError as exc:
            print(Fore.RED + Style.BRIGHT + "You're going too fast! Error: " + exc + Style.RESET_ALL)
            return ""
        response = completion.choices[0].message.content
        self.context.append((new_question, response))
        return response


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

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


def get_question():
    full_question = ""
    current_question = ""
    end = "///"
    print(Fore.GREEN + Style.BRIGHT + f"Enter prompt and then {end} to end your question:" + Style.RESET_ALL)
    while end not in current_question:
        current_question = input()
        full_question += f"\n{current_question}"
    full_question = full_question.replace(end, "")
    return full_question

def run(
    instructions: str,
    question: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: float = 0.6,
    max_contexts: int = 10,
    context_file: str = None,
):
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        context_file=context_file,
    )
    response = question_answer.get_response(question)
    print(response)


def run_iteratively(
    instructions: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: float = 0.6,
    max_contexts: int = 10,
    context_file: str = None,
):
    question_answer = QuestionAnswer(
        instructions=instructions,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_contexts=max_contexts,
        context_file=context_file,
    )

    os.system("cls" if os.name == "nt" else "clear")
    # keep track of previous questions and answers
    while True:
        new_question = get_question()
        print(Fore.CYAN + "Processing..." + Style.RESET_ALL)
        # ask the user for their question
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(Fore.RED + Style.BRIGHT + "Sorry, you're question didn't pass the moderation check:")
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = question_answer.get_response(new_question)
        # print the response
        print(Fore.CYAN + Style.BRIGHT + response + Style.RESET_ALL)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for controlling ChatGPT")
    parser.add_argument(
        "--instructions_path",
        type=str,
        default="/home/eli/workspace/gpt/command_line_chatgpt/instructions.txt",
        help="Filepath for initial ChatGPT instruction prompt (default instructions.txt). See https://github.com/f/awesome-chatgpt-prompts for inspiration",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature value for generating text")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum number of tokens to generate")
    parser.add_argument(
        "--frequency_penalty", type=float, default=0, help="Frequency penalty value for generating text"
    )
    parser.add_argument(
        "--presence_penalty", type=float, default=0.6, help="Presence penalty value for generating text"
    )
    parser.add_argument("--max_contexts", type=int, default=10, help="Maximum number of questions to include in prompt")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_iteratively(
        instructions=read_instructions(args.instructions_path),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        max_contexts=args.max_contexts,
        context_file=None,
    )


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        pass
