import yaml
import argparse
from pathlib import Path
from scripts.train import train, parse_training_args

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

            # Run training
            training_args = parse_training_args(training_arg_array)
            train(training_args)
            
if __name__ == "__main__":
    run()
