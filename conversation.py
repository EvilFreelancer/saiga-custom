import logging

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = """
Ты — Saiga 2, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
"""


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
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

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
        if self.history_limit is not None and len(self.messages) > self.history_limit + 1:
            overflow = len(self.messages) - (self.history_limit + 1)
            self.messages = [self.messages[0]] + self.messages[overflow + 1:]  # remove old messages except system

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


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()
