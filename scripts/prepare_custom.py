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

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Import necessary libraries for Hugging Face datasets
try:
    from datasets import load_dataset
except ImportError as e:
    print('Unable to Import HuggingFace Library. Please try installing library first to use hugging face dataset directly!!!')

def process_jsonl_file(filepath: str, builder: packed_dataset.PackedDatasetBuilder, tokenizer: Tokenizer, data_type: str = "text"):
    import zstandard as zstd
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in tqdm(f):
            data = json.loads(row)
            if data_type == "instruction":
                text = f"{data['instruction']} {data['input']} {data['output']}"
            elif data_type == "conversation":
                # Assuming conversation is a list of dictionaries with keys "from" and "value"
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
            for turn in conversation:  # Assuming each conversation is a list of dicts with 'from' and 'value'
                if turn['from'] == 'human':
                    dialogue.append(f"Human: {turn['value']}")
                elif turn['from'] == 'gpt':
                    dialogue.append(f"GPT: {turn['value']}")
            texts.append(" ".join(dialogue))
    
    elif data_type == "qa":
        question_column = table.column("question").to_pylist()
        answer_column = table.column("answer").to_pylist()
        texts = [f"Q: {q} A: {a}" for q, a in zip(question_column, answer_column)]
        
    else:  # Default to using a 'text' column
        text_column = table.column("text").to_pylist()
        texts = text_column
    
    for text in tqdm(texts):
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

def prepare_full(
    source_path: Union[Path, str],
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    data_format: str = "text",  # Added data_format parameter
    filenames_subset: List[str] = None,
    process_id: int = 0,
    source_is_hf: bool = False,  # Flag to indicate if the source is a Hugging Face dataset
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_dataset_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    if source_is_hf:
        # If the source is a Hugging Face dataset, load the dataset and process it
        dataset = load_dataset(source_path, split=split)
        for row in tqdm(dataset):
            text = row["text"]
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))
    else:
        # If the source is local files, process the files accordingly
        for filepath in filenames_subset:
            print(f"Processing {filepath}")
            if filepath.endswith(".jsonl"):
                process_jsonl_file(filepath, builder, tokenizer, data_format)
            elif filepath.endswith(".parquet"):
                process_parquet_file(filepath, builder, tokenizer, data_format)
            else:
                print(f"Unsupported file format: {filepath}")

    # builder.write_reminder()  # As mentioned in the comment, we avoid writing the final corpus to prevent unnecessary tokens

def prepare(
    source_path: Union[Path, str],
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/output"),
    chunk_size: int = 2049 * 1024,
    split: str = "train",
    data_format: str = "text",  # Added data_format parameter
    percentage: float = 1.0,
) -> None:
    import time

    source_is_hf = False

    if isinstance(source_path, str) and source_path.startswith("HuggingFace"):
        source_is_hf = True

    if source_is_hf:
        # If using a Hugging Face dataset, no need to glob filenames
        filenames = None
    else:
        filenames = glob.glob(os.path.join(source_path, f"**/*.{split}.*"), recursive=True)
        filenames = filenames[:int(len(filenames) * percentage)]

    num_processes = cpu_count()
    chunked_filenames = np.array_split(filenames, num_processes) if filenames else [None] * num_processes

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(
            target=prepare_full,
            args=(source_path, tokenizer_path, destination_path, chunk_size, split, data_format, list(subset), i, source_is_hf),
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


# Example Usage: python your_script.py --source_path ./test_data --tokenizer_path ./tokenizer.model --destination_path ./output --data_format conversation
