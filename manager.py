import yaml
import argparse
from pathlib import Path
from train import train, parse_training_args

def parse_args():
    parser = argparse.ArgumentParser(description="Dreambooth Training Manager")
    parser.add_argument(
        "--config",
        type=str,
        default="./jobs.yml",
        help="Path to a config file with training options",
    )
    args = parser.parse_args()
    return args

def run():
    args = parse_args()
    
    if not Path(args.config).is_file():
        print('Config file not found')
        return

    # Get training configuration from the config file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

        for job_name, job_config in config['jobs'].items():
            # Parse config parameters into array for argparse compatability
            train_config = job_config['train_config']
            training_arg_array = []
            for param_name, param_value in train_config.items():
                training_arg_array.append("--" + param_name)
                if param_value is not None:
                    training_arg_array.append(str(param_value))

            training_args = parse_training_args(training_arg_array)
            
            # Validate some options
            if not Path(training_args.output_dir).is_dir():
                print('output directory path is not valid')
                return

            if training_args.convert_to_ckpt:
                if training_args.output_ckpt_path is None:
                    print('output_ckpt_path must be specified when using convert_to_ckpt')
                    return
                if not Path(training_args.output_ckpt_path).is_dir():
                    print('ckpt output directory path is not valid')
                    return
            
            # Run training
            train(training_args)
            
if __name__ == "__main__":
    run()
