# Dreambooth Training Manager
This is a simple manager to automate running dreambooth training from yaml config files.

## Installation
```bash
git clone https://github.com/briancw/training-manager
cd training-manager
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Optionally install xformers. It will likely take a while to compile. You may need to install build dependencies first.
```bash
pip install git+https://github.com/facebookresearch/xformers#egg=xformers
```

## Running
```bash
source venv/bin/activate
python manager.py --config my-job.yml
```
Refer to the jobs-example-simple.yml and jobs-example-advanced.yml for project setup and configuration options.

#### training_model
The base model used to train with should be inside your diffusers models folder. This needs to be a diffusers style model, not a .ckpt file. You can convert a ckpt model to a diffusers model using the script from the scripts folder in this repo.

#### output_name
Outputs will generated in your diffusers model folder using the name specified in "output_name".

#### class_images and training_images
These should refer to folders inside your specified training_images_path and class_images_path.

#### train_config
You can get all of the available options for the "train_config" section with:
```
python train.py -h
```

#### save_every_n_steps
This projects adds an option to generate a model file every n steps. Previously epochs were used, but epochs are pretty ambigious so now steps are used instead.

<!-- #### min_save_step -->

It is possible to pass paths directly to train_config, but this setup is designed to automatically generate paths in order to handle automatic conversion and pruning.

#### post_config
- "convert_to_ckpt" will convert models to .ckpt format and place them in "ckpt_models_path"
- "prune" will prune down generated .ckpt models. Typically this results in a 50% file size reduction.
- "ckpt_only" will remove the diffusers style model after converting.

### Rare Token Generator
Use the generate_tokens.py script to find a rare 3 character token to use in your instance prompt when training.<br>
The script will work with diffusers style models and **not** .ckpt files.
```bash
python scripts/generate_tokens.py --model_path /some/path/to/your/model
```

### Substantial memory reduction (Credit ShivamShirao)
Using the flag use_8bit_adam with [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) and [xformers](https://github.com/facebookresearch/xformers) will result in a substantial memory reduction. See [ShivamShrirao's repo](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) for more details.<br>
In order to get use_8bit_adam working on Arch linux, I needed to manually add "/opt/cuda/targets/x86_64-linux/lib/" to my LD_LIBRARY_PATH.<br>

## Credits
- Based on [Victarry's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) Dreambooth implementations
- Huggingface for [diffusers](https://github.com/huggingface/diffusers)
- Memory and speed improvments from [ShivamShirao's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) repo
- (Josh Achiam)[https://github.com/jachiam] for diffusers to ckpt script
- (Harubaru)[https://github.com/harubaru] for pruning script
