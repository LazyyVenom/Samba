# Prepare_Custom.py Running

## Setting_Up

Need to install Docker First Along with CUDA TOOLKIT and Nvidia Drivers.
Requirement: Very IMP (This requirement is inherited from Microsoft Samba Codebase)

Make sure All these are installed With:

for Nvidia:
```bash
nvidia-smi
```

for CUDA:
```bash
nvcc --version
```
I will say please prefer 12.1 Cuda version as it may cause problem if other version is 
installed in your system.


for Docker:
```bash
docker --version
```

(I Faced A lot of problems while Handling all these requirements So please check code won't
run without cuda enabled docker Please check it according to your system)


## After Setting is done
Clone my forked repository using:
```bash
git clone https://github.com/LazyyVenom/Samba
```

```bash
docker build -t samba_project .
```
You can change your image name if needed from samba_project to other.

Once the Image is created Successfully you can access it in bash terminal using:
```bash
docker run -it samba_project bash
```
Or whatever way you prefer to use Docker Images. Just incase you use it like this make sure to 
Add the required file directory inside of this.

## Running Script

Preprocessing Dataset (Training)
```bash
python prepare_custom.py --source_path <Input_Path> --destination_path <Output_path> --data_format instruction
```

Preprocessing Dataset (Validation)
```bash
python prepare_custom.py --source_path <Input_Path> --destination_path <Output_path_val> --data_format instruction --split validation
```

--data_format: Possible inputs: (instruction, conversation, qa)
It will pre process it accordingly.


Training Script 
```bash
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=samba-421M --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 pretrain.py --train_data_dir output_val --val_data_dir output_val
```

Inference Script
```bash
python inference_script.py checkpoint_path="path/to/your/checkpoint.pth" tokenizer_path="path/to/your/tokenizer" input_text="This is a new input text."
```

