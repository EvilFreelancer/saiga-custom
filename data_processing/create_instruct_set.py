import json
import re
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
from itertools import tee
from datasketch import MinHash, MinHashLSH, LeanMinHash


def calc_max_length(records):
    return max([sum([len(record['input']) + len(record['output']) + len(record['instruction']) for record in records])])


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
        record["minhash"] = calc_fingerprint(
            record["instruction"] + ' ' + record["input"] + ' ' + record["output"],
            num_perm=num_perm
        )

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

    # collection = "d0rj/dolphin-ru"
    # dolphin_ru_records = []
    # print(f"Loading {collection}")
    # for row in tqdm(load_dataset(collection, split="train")):
    #     row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
    #     dolphin_ru_records.append(row)
    # print(f"{collection} count:", len(dolphin_ru_records))
    # print(f"Max {collection} length:", calc_max_length(dolphin_ru_records), "\n")
    # records += dolphin_ru_records

    collection = "d0rj/gsm8k-ru"
    gsm8k_ru_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        result = {"instruction": row['question'], "output": row['answer'], "input": ''}
        gsm8k_ru_records.append(result)
    print(f"{collection} count:", len(gsm8k_ru_records))
    print(f"Max {collection} length:", calc_max_length(gsm8k_ru_records), "\n")
    records += gsm8k_ru_records

    collection = "d0rj/alpaca-cleaned-ru"
    alpaca_cleaned_ru_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
        alpaca_cleaned_ru_records.append(row)
    print(f"{collection} count:", len(alpaca_cleaned_ru_records))
    print(f"Max {collection} length:", calc_max_length(alpaca_cleaned_ru_records), "\n")
    records += alpaca_cleaned_ru_records

    collection = "IlyaGusev/ru_turbo_alpaca"
    ru_turbo_alpaca_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        row['output'] = row.pop('alternative_output')
        row = {key: value for key, value in row.items() if key in ("input", "output", "instruction")}
        ru_turbo_alpaca_records.append(row)
    print(f"{collection} count:", len(ru_turbo_alpaca_records))
    print(f"Max {collection} length:", calc_max_length(ru_turbo_alpaca_records), "\n")
    records += ru_turbo_alpaca_records

    collection = "IlyaGusev/ru_turbo_alpaca_evol_instruct"
    ru_turbo_alpaca_evol_instruct_records = []
    print(f"Loading {collection}")
    for row in tqdm(load_dataset(collection, split="train")):
        result = {"instruction": row['instruction'], "output": row['output'], "input": ''}
        ru_turbo_alpaca_evol_instruct_records.append(result)
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
