import json
import re
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
from data_processing.bad_substrings import has_bad_ss
from datasketch import MinHash, MinHashLSH, LeanMinHash
from itertools import tee


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def build_char_system_messages(char):
    name = char["name"]
    context = char["context"]
    greeting = char["greeting"]
    example_dialogue = char["example_dialogue"]

    context = f"Ты - {name}. {context}"
    chat = []
    if random.random() < 0.2:
        context += f"\nПриветствие: {greeting}"
        chat.append({
            "role": "bot",
            "content": greeting
        })
    if random.random() < 0.2:
        print(example_dialogue)
        mapping = {
            "user": "Пользователь",
            "char": "Персонаж"
        }
        example_messages = [f'{mapping[m["role"]]}: {m["content"]}' for m in example_dialogue['chat']]
        context += "\nПример диалога:\n" + "\n".join(example_messages)
    chat.insert(0, {
        "role": "system",
        "content": context
    })
    return chat


def re_tokenize(text):
    return re.findall(r'[а-яё-]+|[a-z-]+|\d+|\S', text, re.I)


def ngrams(sequence, n):
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def calc_fingerprint(text, ngram_size: int = 1, num_perm: int = 128):
    tokens = re_tokenize(text)
    if ngram_size > 1:
        tokens = {" ".join(t) for t in ngrams(tokens, ngram_size)}
    tokens = [token.encode('utf-8') for token in tokens]

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(tokens)

    lean_minhash = LeanMinHash(minhash)
    buf = bytearray(lean_minhash.bytesize())
    lean_minhash.serialize(buf)

    return buf


def undup_alpaca(alpaca_records, num_perm: int = 32, threshold: float = 0.3, debug: bool = False):
    for record in tqdm(alpaca_records, desc="Fingerprinting"):
        record["minhash"] = calc_fingerprint(record["messages"][0]["content"], num_perm=num_perm)

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm
    )

    filtered_records = []
    for idx, record in tqdm(enumerate(alpaca_records), desc="Undup"):
        minhash = LeanMinHash.deserialize(record["minhash"])
        is_dup = False
        for other_idx in lsh.query(minhash):
            other_record = alpaca_records[other_idx]
            other_minhash = LeanMinHash.deserialize(other_record["minhash"])
            if minhash.jaccard(other_minhash) > threshold:
                if debug:
                    print()
                    print("=========================")
                    print(record["messages"][0]["content"].replace("\n", " "))
                    print(other_record["messages"][0]["content"].replace("\n", " "))
                is_dup = True
        if is_dup:
            continue
        lsh.insert(idx, minhash)
        filtered_records.append(record)
    for record in filtered_records:
        record.pop("minhash")
    return filtered_records


def main(train_path, val_path):
    random.seed(42)

    records = []

    evol_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca_evol_instruct", split="train")):
        instruction = row["instruction"]
        output = row["output"]
        if has_bad_ss([{"content": output}]):
            continue
        evol_records.append({
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "bot", "content": output}
            ],
            "source": "alpaca-evol-instruct"
        })
    print("Evol-instruct count:", len(evol_records))
    print("Max evol-instruct length:", calc_max_length(evol_records))

    alpaca_records = []
    for row in tqdm(load_dataset("IlyaGusev/ru_turbo_alpaca", split="train")):
        message = row["instruction"]
        if row["input"]:
            message += "\nДано: " + row["input"]
        output = row["alternative_output"]
        if has_bad_ss([{"content": output}]):
            output = row["output"]
            if has_bad_ss([{"content": output}]):
                continue
        alpaca_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": "alpaca"
        })
    print("Alpaca count:", len(alpaca_records))
    print("Max Alpaca length:", calc_max_length(alpaca_records))


for row in load_dataset("d0rj/dolphin-ru", split="train"):
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    records.append(row)

for row in load_dataset("d0rj/gsm8k-ru", split="train"):
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    records.append(row)

for row in load_dataset("d0rj/alpaca-cleaned-ru", split="train"):
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    records.append(row)

for row in load_dataset("IlyaGusev/ru_turbo_alpaca_evol_instruct", split="train"):
    row = {key: value for key, value in row.items() if key in ("output", "instruction")}
    records.append(row)

for row in load_dataset("IlyaGusev/ru_turbo_alpaca", split="train"):
    row["output"] = row.pop("alternative_output")
    row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    records.append(row)

random.shuffle(records)
border = int(0.95 * len(records))
train_records = records[:border]
val_records = records[border:]
with open(train_path, "w") as w:
    for record in train_records:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
with open(val_path, "w") as w:
    for record in val_records:
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

train_path = sys.argv[1]
val_path = sys.argv[2]
main(train_path, val_path)
