import yaml
import argparse
from pathlib import Path
from train import main, parse_args

# Extra Tools
from scripts.convert_diffusers_to_original_stable_diffusion import convert
from scripts.prune import prune

def parse_manager_args():
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
    args = parse_manager_args()
    
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
            
            # TODO: Update log path when generating paths
            # Build path arguments
            if "paths" in config and "job_paths" in job_config:
                input_model_path = config['paths']['diffusers_models_path'] + '/' + job_config['job_paths']['training_model']
                output_model_path = config['paths']['diffusers_models_path'] + '/' + job_config['job_paths']['output_name']
                training_images_path = config['paths']['training_images_path'] + '/' + job_config['job_paths']['training_images']
                
                training_arg_array.append("--pretrained_model_name_or_path")
                training_arg_array.append(input_model_path)
                training_arg_array.append("--output_dir")
                training_arg_array.append(output_model_path)
                training_arg_array.append("--instance_data_dir")
                training_arg_array.append(training_images_path)
                
                # Paths for class images
                if ("class_images_path" in config["paths"] and "class_images" in job_config['job_paths']):
                    class_images_path = config['paths']['class_images_path'] + '/' + job_config['job_paths']['class_images']
                    training_arg_array.append("--class_data_dir")
                    training_arg_array.append(class_images_path)

            training_args = parse_args(training_arg_array)

            # Run training
            main(training_args)

            # TODO path validation
            if "post_config" in job_config:
                post_config = job_config['post_config']
                if ("convert_to_ckpt" in post_config) and post_config['convert_to_ckpt']:
                    ckpt_only = ("ckpt_only" in post_config and post_config['ckpt_only'])
                    do_prune = ("prune" in post_config and post_config['prune'])

                    if training_args.save_every_n_steps:
                        for step in range(training_args.min_save_step, (training_args.max_train_steps + 1), training_args.save_every_n_steps):
                            if step == 0:
                                continue

                            step_path = training_args.output_dir + '_step' + str(step)
                            ckpt_path = config['paths']['ckpt_models_path'] + '/' + job_config['job_paths']['output_name'] + '_step' + str(step) + '.ckpt'
                            print('Converting: ' + step_path)
                            convert(step_path, ckpt_path, False, ckpt_only)
                            if do_prune:
                                prune(ckpt_path, True)
                    else:
                        ckpt_path = config['paths']['ckpt_models_path'] + '/' + job_config['job_paths']['output_name'] + '.ckpt'
                        print('Converting: ' + training_args.output_dir)
                        convert(training_args.output_dir, ckpt_path, False, ckpt_only)
                        if do_prune:
                            prune(ckpt_path, True)
            
if __name__ == "__main__":
    run()
