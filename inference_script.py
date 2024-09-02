import torch
from pathlib import Path
from lit_gpt.model import GPT, Config
from jsonargparse import CLI
from lit_gpt import Tokenizer

MODEL_NAME = "Samba_421M"
TRAIN_CONFIG = "tsz512x4k_20B"

class InferenceConfig:
    checkpoint_path: Path = Path("./checkpoints/model_checkpoint.pth")
    tokenizer_path: Path = Path("./")
    input_text: str = "This is a sample input text."

def load_model(checkpoint_path: Path) -> GPT:
    config = Config.from_name(MODEL_NAME)
    
    model = GPT(config)
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict['model'])
    
    return model

def preprocess_input(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    tokens = tokenizer.encode(text)
    tokens = tokens.unsqueeze(0)  # Add batch dimension
    return tokens

def infer(model: GPT, input_data: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

def postprocess_output(outputs: torch.Tensor, tokenizer: Tokenizer) -> str:
    # Assuming the model output is a tensor of logits
    predictions = outputs.argmax(dim=-1)
    text = tokenizer.decode(predictions.squeeze(0))  # Squeeze batch dimension before decoding
    return text

def main(cfg: InferenceConfig):
    model = load_model(cfg.checkpoint_path)
    
    tokenizer = Tokenizer(cfg.tokenizer_path)
    
    input_data = preprocess_input(cfg.input_text, tokenizer)
    
    # Move input data to the same device as the model if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    
    outputs = infer(model, input_data)
    
    result = postprocess_output(outputs, tokenizer)
    print("Inference Result:", result)

if __name__ == "__main__":
    CLI(main, config_class=InferenceConfig)
