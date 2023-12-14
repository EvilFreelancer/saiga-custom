import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from conversation import Conversation, generate

# MODEL_NAME = "IlyaGusev/gigasaiga_lora"
# MODEL_NAME = "evilfreelancer/ruGPT-3.5-13B-lora"
# MODEL_NAME = "./output"
# MODEL_NAME = "evilfreelancer/saiga_mistral_7b_128k_lora"
MODEL_NAME = "./yarn_mistral_7b_128k_yakovlev_lora"

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = """
Ты — Saiga 2, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.
"""
# DEFAULT_SYSTEM_PROMPT = """
# Ты Иван Яковлевич Яковлев. Просветитель, православный миссионер, российский педагог,
# организатор народных школ, инспектор (руководитель) чувашских школ Казанского учебного
# округа (1875-1903) и созданной им Симбирской чувашской учительской школы (1868-1919),
# создатель нового (современного) чувашского алфавита (1870-е годы) и учебников чувашского
# и русского языков для чувашей, писатель, переводчик, фольклорист. Действительный
# статский советник (1904).
# """


config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=True,
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

template_path = 'internal_prompts/rugpt35.json'
conversation = Conversation(message_template=DEFAULT_MESSAGE_TEMPLATE, system_prompt=DEFAULT_SYSTEM_PROMPT)
while True:
    user_message = input("User: ")

    # Reset chat command
    if user_message.strip() == "/reset":
        conversation = Conversation(message_template=DEFAULT_MESSAGE_TEMPLATE, system_prompt=DEFAULT_SYSTEM_PROMPT)
        print("History reset completed!")
        continue

    # Skip empty messages from user
    if user_message.strip() == "":
        continue

    conversation.add_user_message(user_message)
    prompt = conversation.get_prompt(tokenizer)
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        generation_config=generation_config
    )
    conversation.add_bot_message(output)
    print("Bot:", output)
    print()
    print("==============================")
    print()
