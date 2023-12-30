import logging

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — PavelGPT, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
            self,
            message_template=DEFAULT_MESSAGE_TEMPLATE,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            start_token_id=2,
            # Bot token may be a list or single int
            bot_token_id=10093,  # yarn_mistral_7b_128k
            # bot_token_id=46787,  # rugpt35_13b
            # int (amount of questions and answers) or None (unlimited)
            history_limit=None,
    ):
        self.logger = logging.getLogger('Conversation')
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.history_limit = history_limit
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "bot",
                "content": "Здравствуйте! Чем могу помочь?"
            }
        ]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_message(self, role, message):
        self.messages.append({
            "role": role,
            "content": message
        })
        self.trim_history()

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_bot_message(self, message):
        self.add_message("assistant", message)

    def trim_history(self):
        if self.history_limit is not None and len(self.messages) > self.history_limit + 2:
            overflow = len(self.messages) - (self.history_limit + 2)
            self.messages = [self.messages[0]] + self.messages[overflow + 2:]  # remove old messages except system

    def get_prompt(self, tokenizer):
        final_text = ""
        # print(self.messages)
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text

        # Bot token id may be an array
        if isinstance(self.bot_token_id, (list, tuple)):
            final_text += tokenizer.decode([self.start_token_id] + self.bot_token_id)
        else:
            final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])

        return final_text.strip()


def generate(model, prompt, messages, generation_config):
    output = model(
        prompt,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        temperature=generation_config.temperature,
        repeat_penalty=generation_config.repetition_penalty,
    )['choices'][0]['text']
    return output.strip()


from llama_cpp import Llama
import os
from pathlib import Path
from huggingface_hub.file_download import http_get
from transformers import GenerationConfig

directory = Path('.').resolve()
model_name = "pavelgpt_7b_128k/ggml-model-Q2_K.gguf"
generation_config = GenerationConfig.from_pretrained("pavelgpt_7b_128k")
final_model_path = str(directory / model_name)

# if not os.path.exists(final_model_path):
#     with open(final_model_path, "wb") as f:
#         http_get(model_url, f)
# os.chmod(final_model_path, 0o777)
# print(f"{final_model_path} files downloaded.")

model = Llama(
    model_path=final_model_path,
    # verbose=True,
    n_gpu_layers=5,
    n_ctx=4096,
    max_length=200,
    echo=True,
)

conversation = Conversation(bot_token_id=7451)
while True:
    user_message = input("User: ")

    # Reset chat command
    if user_message.strip() == "/reset":
        conversation = Conversation(bot_token_id=7451)
        print("History reset completed!")
        continue

    # Skip empty messages from user
    if user_message.strip() == "":
        continue

    conversation.add_user_message(user_message)
    prompt = conversation.get_prompt(model.tokenizer())
    output = generate(
        model=model,
        prompt=prompt,
        generation_config=generation_config,
        messages=conversation.messages
    )

    conversation.add_bot_message(output)
    print("Bot:", output)
    print()
    print("==============================")
    print()
