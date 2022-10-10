# Dreambooth Training Manager
Based on [ShivamShrirao's dreambooth script](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)

### Substantial memory reduction
Using [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes) amd [xformers](https://github.com/facebookresearch/xformers) will result in a substantial memory reduction. See ShivamShrirao's repo for more details.
```bash
pip install git+https://github.com/facebookresearch/xformers#egg=xformers
pip install bitsandbytes
```

### Arch Linux Bitsandbytes
At least for me, when running on Arch, I need to manually specify LD_LIBRARY_PATH for bitsandbytes.
```
LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/
```
