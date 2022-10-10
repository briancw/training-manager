# Dreambooth Training Manager
### Install
```bash
git clone https://github.com/briancw/training-manager
cd training-manager
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optionally install xformers
pip install git+https://github.com/facebookresearch/xformers#egg=xformers
```

### Save epochs during training
You can use the flag "--save_each_epoch=n" to save the model at every n epochs.

### Rare Token Generator
Use the generate_tokens.py script to find a rare 3 character token to use in your instance prompt when training.<br>
The script will work with diffusers style models and **not** .ckpt files.
```bash
python generate_tokens.py --model_path /some/path/to/your/model
```

### Substantial memory reduction (Credit ShivamShirao)
Using [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) amd [xformers](https://github.com/facebookresearch/xformers) will result in a substantial memory reduction. See [ShivamShrirao's repo](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) for more details.

### Arch Linux Bitsandbytes
At least for me, when running on Arch, I need to manually specify LD_LIBRARY_PATH for bitsandbytes in my run script
```
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/
```

## Credits
- Based on [Victarry's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) Dreambooth implementations
- Huggingface for [diffusers](https://github.com/huggingface/diffusers)
- Memory and speed improvments from [ShivamShirao's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) repo