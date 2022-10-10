# Dreambooth Training Manager
### Install
```bash
git clone https://github.com/briancw/training-manager
cd training-manager
python -m venv venv
pip install -r requirements.txt
```

### Rare Token Generator
Use the generate_tokens.py script to find a rare 3 character token to use in your instance prompt when training.<br>
The script will work with diffusers style models and **not** .ckpt files.
```bash
python generate_tokens.py --model_path /some/path/to/your/model
```

### Substantial memory reduction (Credit ShivamShirao)
Using [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) amd [xformers](https://github.com/facebookresearch/xformers) will result in a substantial memory reduction. See ShivamShrirao's repo for more details.
```bash
pip install git+https://github.com/facebookresearch/xformers#egg=xformers
pip install bitsandbytes
```

### Arch Linux Bitsandbytes
At least for me, when running on Arch, I need to manually specify LD_LIBRARY_PATH for bitsandbytes in my run script
```
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/
```

## Credits
- Based on [Victarry's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) Dreambooth implementations.
- Huggingface for [diffusers](https://github.com/huggingface/diffusers)
- Enhancents from [ShivamShirao's](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth) Dreambooth repo have been incorporated.