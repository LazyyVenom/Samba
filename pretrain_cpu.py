import json
import glob
import os
import sys
from pathlib import Path
from typing import List, Union
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import pyarrow.parquet as pq

#Suppressing Warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Import necessary libraries for Hugging Face datasets
try:
    from datasets import load_dataset
except ImportError as e:
    print('Unable to Import HuggingFace Library. Please try installing the library first to use Hugging Face datasets directly!!!')

def process_jsonl_file(filepath: str, builder: packed_dataset.PackedDatasetBuilder, tokenizer: Tokenizer, data_type: str = "text"):
    if filepath.endswith('.jsonl'):
        import zstandard as zstd
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in tqdm(f):
                data = json.loads(row)
                if data_type == "instruction":
                    text = f"{data['instruction']} {data['input']} {data['output']}"
                elif data_type == "conversation":
                    conversation = []
                    for turn in data['conversation']:
                        if turn['from'] == 'human':
                            conversation.append(f"Human: {turn['value']}")
                        elif turn['from'] == 'gpt':
                            conversation.append(f"GPT: {turn['value']}")
                    text = " ".join(conversation)
                elif data_type == "qa":
                    text = f"Q: {data['question']} A: {data['answer']}"
                else:
                    text = data["text"]
            
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
    
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load the JSON data
            for entry in tqdm(data, desc=f"Processing {os.path.basename(filepath)}"):
                try:
                    if data_type == "instruction":
                        text = f"{entry['instruction']} {entry['input']} {entry['output']}"
                    elif data_type == "conversation":
                        conversation = []
                        for turn in entry['conversation']:
                            if turn['from'] == 'human':
                                conversation.append(f"Human: {turn['value']}")
                            elif turn['from'] == 'gpt':
                                conversation.append(f"GPT: {turn['value']}")
                        text = " ".join(conversation)
                    elif data_type == "qa":
                        text = f"Q: {entry['question']} A: {entry['answer']}"
                    else:
                        text = entry["text"]

                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                except KeyError as e:
                    print(f"Missing key {e} in entry: {entry}")
                except Exception as e:
                    print(f"Unexpected error processing entry {entry}: {e}")


def process_parquet_file(filepath: str, builder: packed_dataset.PackedDatasetBuilder, tokenizer: Tokenizer, data_type: str = "text"):
    table = pq.read_table(filepath)
    
    if data_type == "instruction":
        instruction_column = table.column("instruction").to_pylist()
        input_column = table.column("input").to_pylist()
        output_column = table.column("output").to_pylist()
        texts = [f"{inst} {inp} {out}" for inst, inp, out in zip(instruction_column, input_column, output_column)]
        
    elif data_type == "conversation":
        conversation_column = table.column("conversation").to_pylist()
        texts = []
        for conversation in conversation_column:
            dialogue = []
            for turn in conversation:
                if turn['from'] == 'human':
                    dialogue.append(f"Human: {turn['value']}")
                elif turn['from'] == 'gpt':
                    dialogue.append(f"GPT: {turn['value']}")
            texts.append(" ".join(dialogue))
    
    elif data_type == "qa":
        question_column = table.column("question").to_pylist()
        answer_column = table.column("answer").to_pylist()
        texts = [f"Q: {q} A: {a}" for q, a in zip(question_column, answer_column)]
        
    else:
        text_column = table.column("text").to_pylist()
        texts = text_column
    
    for text in tqdm(texts):
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

def process_hf_dataset(dataset, builder: packed_dataset.PackedDatasetBuilder, tokenizer: Tokenizer, data_format: str = "text"):
    for row in tqdm(dataset):
        if data_format == "instruction":
            text = f"{row['instruction']} {row['input']} {row['output']}"
        elif data_format == "conversation":
            conversation = []
            for turn in row['conversation']:
                if turn['from'] == 'human':
                    conversation.append(f"Human: {turn['value']}")
                elif turn['from'] == 'gpt':
                    conversation.append(f"GPT: {turn['value']}")
            text = " ".join(conversation)
        elif data_format == "qa":
            text = f"Q: {row['question']} A: {row['answer']}"
        else:
            text = row["text"]
        
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

def prepare_full(
    source_path: Union[Path, str],
    destination_path: Path,
    chunk_size: int,
    tokenizer_path: Union[Path, str] = "./",
    split: str = "train",
    data_format: str = "text",
    filenames_subset: List[str] = None,
    process_id: int = 0,
    source_is_hf: bool = False,
) -> None:
    
    if filenames_subset is None:
        filenames_subset = []
        
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(Path(tokenizer_path))

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_dataset_{process_id}",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    if source_is_hf:
        dataset = load_dataset(source_path, split=split)
        process_hf_dataset(dataset, builder, tokenizer, data_format)
    else:
        for filepath in filenames_subset:
            print(f"Processing {filepath}")
            if filepath.endswith(".jsonl") or filepath.endswith(".json"):
                process_jsonl_file(os.path.join(source_path,filepath), builder, tokenizer, data_format)
            elif filepath.endswith(".parquet"):
                process_parquet_file(filepath, builder, tokenizer, data_format)
            else:
                print(f"Unsupported file format: {filepath}")

def prepare(
    source_path: Union[Path, str] = Path("data/input"),
    tokenizer_path: Union[Path, str] = "./",
    destination_path: Path = Path("data/output"),
    chunk_size: int = 2049 * 1024,
    split: str = "train",
    data_format: str = "text",
    percentage: float = 1.0,
) -> None:
    import time

    source_is_hf = False

    if isinstance(source_path, str) and source_path.startswith("HuggingFace"):
        print("HuggingFace H Bawa")
        source_is_hf = True
        source_path = source_path.replace('HuggingFace/', '')

    else:
        print("This Is Source Path:", source_path)
        print("ITS A PATH")
        
        
    if source_is_hf:
        filenames = None
    else:
        import os
        # print("Current working directory:", os.getcwd())
        # print(os.listdir(os.getcwd()))
        filenames = os.listdir(source_path)
        # print(filenames)
        temp = []
        for filename in filenames:
            if filename.endswith(".json") or filename.endswith(".jsonl") or filename.endswith(".parquet"):
                temp.append(filename)
        filenames = temp
        # print(filenames)
        filenames = filenames[:int(len(filenames) * percentage)]
        # print(filenames)

    num_processes = cpu_count()
    chunked_filenames = np.array_split(filenames, num_processes) if filenames else [None] * num_processes

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        if subset is None:  # Skip process creation if there's nothing to process
            continue
        p = Process(
            target=prepare_full,
            args=(source_path, destination_path, chunk_size, tokenizer_path, split, data_format, list(subset), i, source_is_hf),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)