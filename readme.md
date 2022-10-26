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

### Model Paths
A diffusers style model is required for training, and will also output diffusers style models into that folder. You can convert a .ckpt model into a diffusers model with the script in this projects scripts folder. You can also automatically convert all generated models into .ckpt files after training.

#### train_config
You can get all of the available options for the "train_config" section with:
```
python train.py -h
```

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
- Based on [Shivam's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) Dreambooth improvements
- [Victarry](https://github.com/Victarry/stable-dreambooth) for the original Dreambooth implementation and token generation script
- [Huggingface](https://github.com/huggingface/diffusers) for diffusers
- [Josh Achiam](https://github.com/jachiam) for diffusers to ckpt script
- [Harubaru](https://github.com/harubaru) for pruning script
