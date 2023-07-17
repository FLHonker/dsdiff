
# Dual Sparse Diffusion Models (DSDiff)

This is a project that combines Q-diffusion [1] and Token Merging (ToMe) [2] for compression and acceleration of diffusion models. It considers two aspects of sparsification: sparse parameters and sparse data (tokens). 
Note that the DSDiff algorithm only supports Transformer-based diffusion models, i.e. [stable-diffusion](https://github.com/CompVis/stable-diffusion), nowadays. 

We reimplement the calibration part of q-diffusion but not completely. 

## Installation

Clone this repository, and then create and activate a suitable conda environment named `dsdiff` by using the following command:

```bash
git clone https://github.com/FLHonker/dsdiff.git
cd dsdiff
conda env create -f environment.yml
conda activate dsdiff
```

## Run

1. First download relvant checkpoints following the instructions in the [stable-diffusion](https://github.com/CompVis/stable-diffusion#weights) repos from CompVis. We currently use `sd-v1-4.ckpt` for Stable Diffusion. 

2. Download quantized checkpoints from the Google Drive [[link](https://drive.google.com/drive/folders/1ImRbmAvzCsU6AOaXbIeI7-4Gu2_Scc-X?usp=share_link)]. The checkpoints quantized with 4/8-bit weights-only quantization are the same as the ones with 4/8-bit weights and 8-bit activations quantization. 

3. Then use the following commands to run inference scripts with quantized checkpoints:

```bash
# Stable Diffusion
# 4-bit weights-only, ratio=50%
python scripts/txt2img.py --prompt <prompt. e.g. "a puppet wearing a hat"> --plms --cond --ptq --weight_bit 4 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --ratio 0.5 --outdir <output_path> --cali_ckpt <quantized_ckpt_path> 
# 4-bit weights, 8-bit activations, ratio=50% (with 16-bit for attention matrices after softmax)
python scripts/txt2img.py --prompt <prompt. e.g. "a puppet wearing a hat"> --plms --cond --ptq --weight_bit 4 --quant_mode qdiff --no_grad_ckpt --split --n_samples 5 --quant_act --act_bit 8 --sm_abit 16 --ratio 0.5 --outdir <output_path> --cali_ckpt <quantized_ckpt_path> 
```

After generating images by the above scripts, you can evaluate the FID score using the following command: 

```bash
# calculate the FID score, please modify the image path firstly!
python scripts/eval_fid.py 
```

## Results

We apply the DSDff on stable-diffusion-v1.4. "A(v1.5)/B": ToMe reported the reults (A) on v1.5, and we re-tested as B in the table. Due to the randomly generated samples, the error variation of FID is relatively large. 

| ToMe | Q-diff | quant_act | FID (SD-v1.4) |
| :---: | :---: | :---: | --- |
| 0 | FP32 | × | 33.12(v1.5)/33.77 |
| 50% | FP32 | × | 33.02(v1.5)/33.54 |
| 0 | W4A32 | × | 33.71 |
| 50% | W4A32 | × | 34.06 |
| 0 | W4A8 | ✅ | 33.31 |
| 10% | W4A8 | ✅ | 33.28 |
| 30% | W4A8 | ✅ | 33.90 |
| 50% | W4A8 | ✅ | 34.02 |
| 60% | W4A8 | ✅ | 34.20 |

We do not have the conditions to test the acceleration effect of real quantized models, but theoretically ToMe can further accelerate quantized models by about 2x. It can be seen from the table that the two sparse methods have little impact on performance, and they do not interfere with each other and can complement each other.

## Acknowledgments

Our code was developed based on [q-diffusion](https://github.com/Xiuyu-Li/q-diffusion) and [tomesd](https://github.com/dbolya/tomesd). 
We referred to [BRECQ](https://github.com/yhhhli/BRECQ) for repreducing the calibration part of Q-diffusion. We thank [torch-fidelity](https://github.com/toshas/torch-fidelity) for IS and FID computation. 

## References

[1] Q-Diffusion: Quantizing Diffusion Models, arXiv:2302.04304

[2] Token Merging for Fast Stable Diffusion, CVPR Workshop 2023
