import os
import json
import re
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
from itertools import tee
from datasketch import MinHash, MinHashLSH, LeanMinHash
from data_processing.bad_substrings import has_bad_ss
from joblib import Parallel, delayed


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


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
    with Parallel(n_jobs=os.cpu_count()) as parallel:
        fingerprints = parallel(
            delayed(calc_fingerprint)(record["messages"][0]["content"], 1, num_perm)
            for record in tqdm(alpaca_records, desc="Fingerprinting")
        )

    for idx, record in tqdm(enumerate(alpaca_records)):
        record["minhash"] = fingerprints[idx]

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
                break
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

    # collection = "dim/OpenOrca-ru-gpt4"
    # openorca_ru_gpt4_records = []
    # print(f"Loading {collection}")
    # for row in tqdm(load_dataset(collection, split="train")):  # 50k instructions
    #     message = ''
    #     if row["system_prompt"]:
    #         message += row["system_prompt"]
    #     if row["system_prompt"] and row["question"]:
    #         message += "\n"
    #     if row["question"]:
    #         message += row["question"]
    #     output = row["response"]
    #     if has_bad_ss([{"content": output}]):
    #         continue
    #     messages = [
    #         {"role": "user", "content": message},
    #         {"role": "bot", "content": output}
    #     ]
    #     openorca_ru_gpt4_records.append({
    #         "messages": messages,
    #         "source": collection
    #     })
    # print(f"{collection} count:", len(openorca_ru_gpt4_records))
    # print(f"Max {collection} length:", calc_max_length(openorca_ru_gpt4_records), "\n")
    # records += openorca_ru_gpt4_records

    # collection = "d0rj/OpenOrca-ru"
    # openorca_ru_records = []
    # print(f"Loading {collection}")
    # for row in tqdm(load_dataset(collection, split="train[50000:100000]")):  # 50k instructions
    #     message = ''
    #     if row["system_prompt"]:
    #         message += row["system_prompt"]
    #     if row["system_prompt"] and row["question"]:
    #         message += "\n"
    #     if row["question"]:
    #         message += row["question"]
    #     output = row["response"]
    #     if has_bad_ss([{"content": output}]):
    #         continue
    #     messages = [
    #         {"role": "user", "content": message},
    #         {"role": "bot", "content": output}
    #     ]
    #     openorca_ru_records.append({
    #         "messages": messages,
    #         "source": collection
    #     })
    # print(f"{collection} count:", len(openorca_ru_records))
    # print(f"Max {collection} length:", calc_max_length(openorca_ru_records), "\n")
    # records += openorca_ru_records

    # collection = "d0rj/dolphin-ru"
    # dolphin_ru_records = []
    # print(f"Loading {collection}")
    # for row in tqdm(load_dataset(collection, split="train")):  # 50k instructions
    #     message = ''
    #     if row["instruction"]:
    #         message += row["instruction"]
    #     if row["instruction"] and row["input"]:
    #         message += "\n"
    #     if row["input"]:
    #         message += row["input"]
    #     output = row["output"]
    #     if has_bad_ss([{"content": output}]):
    #         continue
    #     messages = [
    #         {"role": "user", "content": message},
    #         {"role": "bot", "content": output}
    #     ]
    #     dolphin_ru_records.append({
    #         "messages": messages,
    #         "source": collection
    #     })
    # print(f"{collection} count:", len(dolphin_ru_records))
    # print(f"Max {collection} length:", calc_max_length(dolphin_ru_records), "\n")
    # records += dolphin_ru_records

    collection = "d0rj/gsm8k-ru"
    gsm8k_ru_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        message = row["question"]
        output = row["answer"]
        if has_bad_ss([{"content": output}]):
            continue
        gsm8k_ru_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": collection
        })
    print(f"{collection} count:", len(gsm8k_ru_records))
    print(f"Max {collection} length:", calc_max_length(gsm8k_ru_records), "\n")
    records += gsm8k_ru_records

    collection = "d0rj/alpaca-cleaned-ru"
    alpaca_cleaned_ru_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        message = ''
        if row["instruction"]:
            message += row["instruction"]
        if row["instruction"] and row["input"]:
            message += "\n"
        if row["input"]:
            message += row["input"]
        output = row["output"]
        if has_bad_ss([{"content": output}]):
            continue
        alpaca_cleaned_ru_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": collection
        })
    print(f"{collection} count:", len(alpaca_cleaned_ru_records))
    print(f"Max {collection} length:", calc_max_length(alpaca_cleaned_ru_records), "\n")
    records += alpaca_cleaned_ru_records

    collection = "IlyaGusev/ru_turbo_alpaca"
    ru_turbo_alpaca_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        if row["label"] == "ok" or not row["label"]:
            continue
        message = ''
        if row["instruction"]:
            message += row["instruction"]
        if row["instruction"] and row["input"]:
            message += "\n"
        if row["input"]:
            message += row["input"]
        output = row["output"]
        if has_bad_ss([{"content": output}]):
            continue
        ru_turbo_alpaca_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": collection
        })
    print(f"{collection} count:", len(ru_turbo_alpaca_records))
    print(f"Max {collection} length:", calc_max_length(ru_turbo_alpaca_records), "\n")
    records += ru_turbo_alpaca_records

    collection = "IlyaGusev/ru_turbo_alpaca_evol_instruct"
    ru_turbo_alpaca_evol_instruct_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        message = row["instruction"]
        output = row["output"]
        if has_bad_ss([{"content": output}]):
            continue
        ru_turbo_alpaca_evol_instruct_records.append({
            "messages": [
                {"role": "user", "content": message},
                {"role": "bot", "content": output}
            ],
            "source": collection
        })
    print(f"{collection} count:", len(ru_turbo_alpaca_evol_instruct_records))
    print(f"Max {collection} length:", calc_max_length(ru_turbo_alpaca_evol_instruct_records), "\n")
    records += ru_turbo_alpaca_evol_instruct_records

    print("All count:", len(records))
    print("All max length:", calc_max_length(records), "\n")

    updup_records = undup_alpaca(records)
    print("Instruct after undup count:", len(updup_records), "\n")

    random.shuffle(updup_records)
    border = int(0.95 * len(updup_records))
    train_records = updup_records[:border]
    val_records = updup_records[border:]
    with open(train_path, "w") as w:
        for record in train_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
    with open(val_path, "w") as w:
        for record in val_records:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    main(train_path, val_path)
