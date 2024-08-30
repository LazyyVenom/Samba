# Dataset Preparation Script

This script processes JSONL and Parquet files to prepare datasets for machine learning models. It supports different data formats, including **instruction**, **conversation**, **QA**, and **text**. The script also supports multiprocessing for faster processing and can handle Hugging Face datasets.

## Features

- **Data Format Support**: Handles different data formats like `instruction`, `conversation`, `QA`, and `text`.
- **File Format Support**: Processes both JSONL and Parquet file formats.
- **Multiprocessing**: Utilizes multiple CPU cores to speed up dataset preparation.
- **Hugging Face Integration**: Directly processes datasets from the Hugging Face library.

## Prerequisites

- Python 3.7+
- Required Python libraries:
  - `numpy`
  - `tqdm`
  - `pyarrow`
  - `multiprocessing`
  - `jsonargparse`
  - `zstandard`
  - `datasets` (optional, for Hugging Face datasets)

## Installation

```bash
pip install numpy tqdm pyarrow zstandard jsonargparse datasets
```

## Usage

### Basic Usage

```bash
python your_script.py --source_path <source_path> --tokenizer_path <tokenizer_path> --destination_path <destination_path> --data_format <data_format>
```

### Parameters

- `--source_path`: Path to the directory containing your source data files (JSONL or Parquet).
- `--tokenizer_path`: Path to the tokenizer model file.
- `--destination_path`: Path to the output directory where processed data will be saved.
- `--data_format`: Specifies the data format to use. Options are `instruction`, `conversation`, `qa`, or `text`.
- `--chunk_size` (optional): Size of each data chunk in bytes. Default is `2049 * 1024`.
- `--split` (optional): Specifies the data split to process (e.g., `train`, `test`). Default is `train`.
- `--percentage` (optional): Percentage of files to process. Default is `1.0` (100%).

### Example Command

To process data in the **conversation** format:

```bash
python your_script.py --source_path ./test_data --tokenizer_path ./tokenizer.model --destination_path ./output --data_format conversation
```

### Multiprocessing

The script automatically uses all available CPU cores to speed up processing. The filenames are split into chunks, and each chunk is processed in parallel.

## File Formats

### JSONL

Each JSONL file should contain one JSON object per line. The structure of the JSON object depends on the `data_format`:

- **Instruction**:
  ```json
  { "instruction": "Do this", "input": "with this input", "output": "and get this output" }
  ```
- **Conversation**:
  ```json
  { "conversation": [{ "from": "human", "value": "Can you explain?" }, { "from": "gpt", "value": "Sure..." }] }
  ```
- **QA**:
  ```json
  { "question": "What is AI?", "answer": "Artificial Intelligence" }
  ```
- **Text**:
  ```json
  { "text": "Just a plain text example." }
  ```

### Parquet

Parquet files should have columns matching the `data_format`:

- **Instruction**: Columns named `instruction`, `input`, and `output`.
- **Conversation**: A `conversation` column with lists of dictionaries.
- **QA**: Columns named `question` and `answer`.
- **Text**: A `text` column.

## License

This project is licensed under the MIT License and the Apache License 2.0. See the LICENSE files for details.

## Troubleshooting

If you encounter any issues, ensure all required libraries are installed, and the data files are in the correct format. For Hugging Face datasets, ensure the `datasets` library is installed.