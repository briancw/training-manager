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
```
source venv/bin/activate
python manager.py --config my-job.yml
```

"jobs.yml" will be used by default, or you can use a specific file with --config.<br>

All configuration options can be viewed with:
```
python train.py -h
```

Training is based on Diffuers dreambooth, so you will need to supply a diffuers model. There is a simple script in the scripts folder to convert an existing ckpt model to diffuers.

### Rare Token Generator
Use the generate_tokens.py script to find a rare 3 character token to use in your instance prompt when training.<br>
The script will work with diffusers style models and **not** .ckpt files.
```bash
python scipts/generate_tokens.py --model_path /some/path/to/your/model
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
