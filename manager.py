import json
from train import main, parse_args

arg_string = """{
    "model_path": "/path/to/your/model",
    "instance_data_dir": "/path/to/your/images",
    "output_dir": "/path/to/output",
    "gradient_accumulation_steps": "1",
    "mixed_precision": "fp16",
    "seed": "80085",
    "instance_prompt": "xyz person",
    "resolution": "512",
    "train_batch_size": "1",
    "learning_rate": "5e-6",
    "lr_scheduler": "constant",
    "lr_warmup_steps": "0",
    "sample_batch_size": "4",
    "num_train_epochs": "5",
    "save_each_epoch": "1",
    "use_8bit_adam": null
}"""

input_args = []
for key, value in json.loads(arg_string).items():
    input_args.append("--" + key)
    if value:
        input_args.append(value)

args = parse_args(input_args)
main(args)
