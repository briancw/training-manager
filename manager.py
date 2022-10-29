import yaml
import argparse
import requests
from pathlib import Path
import os
import subprocess
import shutil

# Extra Tools
from scripts.convert_diffusers_to_original_stable_diffusion import convert
from scripts.prune import prune

def parse_manager_args():
    parser = argparse.ArgumentParser(description="Dreambooth Training Manager")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./jobs.yml",
        help="Path to a config file with training options",
    )
    args = parser.parse_args()
    return args

def run():
    args = parse_manager_args()
    
    # Check if config file exists
    if not Path(args.config).is_file():
        print('Config file not found')
        return

    # Check if training script exists
    if not Path('./train_dreambooth.py').is_file():
        print("Training script not found. Downloading now.")
        fname = "train_dreambooth.py"
        url = "https://raw.githubusercontent.com/ShivamShrirao/diffusers/62799dcfbc08771595ea2194a1f6ae52de1e7def/examples/dreambooth/train_dreambooth.py"
        r = requests.get(url)
        open(fname, 'wb').write(r.content)
        print("Training script downloaded.")
    
    # Get training configuration from the config file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

        for job_name, job_config in config['jobs'].items():
            # Parse config parameters into array
            train_config = job_config['train_config']
            training_arg_array = []
            for param_name, param_value in train_config.items():
                training_arg_array.append("--" + param_name)
                if param_value is not None:
                    training_arg_array.append(str(param_value))
            
            # TODO: Update log path when generating paths
            # Build path arguments
            if "paths" in config and "job_paths" in job_config:
                output_dir = os.path.join(config['paths']['diffusers_models_path'], job_config['job_paths']['output_name'])

                training_arg_array.append("--pretrained_model_name_or_path")
                training_arg_array.append(os.path.join(config['paths']['diffusers_models_path'], job_config['job_paths']['training_model']))
                training_arg_array.append("--output_dir")
                training_arg_array.append(output_dir)
                training_arg_array.append("--instance_data_dir")
                training_arg_array.append(os.path.join(config['paths']['training_images_path'], job_config['job_paths']['training_images']))
                
                # Paths for class images
                if ("class_images_path" in config["paths"] and "class_images" in job_config['job_paths']):
                    training_arg_array.append("--class_data_dir")
                    training_arg_array.append(os.path.join(config['paths']['class_images_path'], job_config['job_paths']['class_images']))

            # Run training
            subprocess.run(["python", "train_dreambooth.py", *training_arg_array])

            # TODO path validation
            # Convert and prune created models
            if "post_config" in job_config:
                post_config = job_config['post_config']
                if ("convert_to_ckpt" in post_config) and post_config['convert_to_ckpt']:
                    ckpt_only = ("ckpt_only" in post_config and post_config['ckpt_only'])
                    do_prune = ("prune" in post_config and post_config['prune'])

                    for f in os.scandir(output_dir):
                        if f.is_dir() and f.name != "0":
                            ckpt_name = job_config['job_paths']['output_name'] + "_step" + f.name + ".ckpt"
                            ckpt_path = os.path.join(config['paths']['ckpt_models_path'], ckpt_name)
                            print('Converting: ' + f.path)
                            convert(f.path, ckpt_path)

                            if ckpt_only:
                                # Remove the original
                                print("Removing original model: " + str(f.path))
                                shutil.rmtree(f.path)

                            if do_prune:
                                print('Pruning: ' + str(ckpt_path))
                                prune(ckpt_path, True)
                    
                    # TODO remove parent directory
            
if __name__ == "__main__":
    run()
