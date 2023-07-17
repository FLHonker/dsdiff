import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import tomesd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from qdiff import *
from qdiff.utils import convert_adaround
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from ldm.util import instantiate_from_config


def get_cali_samples(args):
    class CALI_COCO(Dataset):
        def __init__(self, root):
            self.samples = [os.path.join(root, file) for file in os.listdir(root)]

        def __getitem__(self, index):
            item = torch.load(self.samples[index])
            img, ts, cond = item['img'], item['ts'], item['cond']
            return (img, ts, cond)
        
        def __len__(self):
            return len(self.samples)

    cali_dataset = CALI_COCO(root=args.data_path)
    data_loader = DataLoader(cali_dataset, batch_size=32, num_workers=8, shuffle=True, pin_memory=True)
    cali_data = {'img':[], 'ts':[], 'cond':[]}
    for batch in data_loader:
        cali_data['img'].append(batch[0])
        cali_data['ts'].append(batch[1])
        cali_data['cond'].append(batch[2])
        if len(cali_data['img']) * batch[0].size(0) >= args.num_samples:
            break
    cali_data['img'] = torch.cat(cali_data['img'], dim=0)[:args.num_samples]
    cali_data['ts'] = torch.cat(cali_data['ts'], dim=0)[:args.num_samples]
    cali_data['cond'] = torch.cat(cali_data['cond'], dim=0)[:args.num_samples]
    return (cali_data['img'], cali_data['ts'], cali_data['cond'])

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument("--outdir", type=str, nargs="?", default="outdir/cali_quant_sd", 
                        help="dir to write results to")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt", 
                        help="path to checkpoint of model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='outdir/cali_coco/samples', type=str, help='path to ImageNet data')

    # linear quantization configs
    parser.add_argument("--quant_act", action="store_true", help="if to quantize activations when ptq==True")
    parser.add_argument("--weight_bit", type=int,default=4, help="int bit for weight quantization")
    parser.add_argument("--act_bit", type=int, default=8, help="int bit for activation quantization")
    parser.add_argument("--quant_mode", type=str, default="qdiff", choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use")

    # qdiff specific configs
    parser.add_argument("--cond", action="store_true", help="whether to use conditional guidance")
    parser.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing")
    parser.add_argument("--split", action="store_true", help="use split strategy in skip connection")
    parser.add_argument("--sm_abit",type=int, default=16, help="attn softmax activation bit")
    parser.add_argument("--verbose", action="store_true", help="print out info like quantized model arch")
    parser.add_argument("--ratio", type=float, default=0.5, help="ratio of token merging")

    # weight calibration parameters
    parser.add_argument('--num_samples', default=6400, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    opt = parser.parse_args()

    seed_everything(opt.seed)
    os.makedirs(opt.outdir, exist_ok=True)
    save_path = os.path.join(opt.outdir, 'sd_v1.4_w{}a{}.pth'.format(opt.weight_bit, opt.act_bit))

    # load model
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    # build quantization parameters
    if opt.split:
        setattr(model.model.diffusion_model, "split", True)
    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': opt.quant_act}
    qnn = QuantModel(
        model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff", sm_abit=opt.sm_abit)
    qnn.to(device)
    qnn.eval()
    if opt.no_grad_ckpt:
        print('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)
    
    # ToMe - token sparse
    if opt.ratio > 0:
        model.model.diffusion_model = qnn
        model = tomesd.apply_patch(model, ratio=opt.ratio)
        qnn = model.model.diffusion_model
        print('Applying ToMe on Stable Diffusion.')
    
    # load cali data
    cali_data = get_cali_samples(opt)

    # Initialize weight quantization parameters
    print("Initializing weight quantization parameters")
    qnn.set_quant_state(True, False)
    if not opt.cond:
        _ = qnn(cali_data[0][:4].to(device), cali_data[1][:4].to(device))
    else:
        _ = qnn(cali_data[0][:4].to(device), cali_data[1][:4].to(device), cali_data[2][:4].to(device))
    # change weight quantizer from uniform to adaround
    convert_adaround(qnn)
    
    # for m in qnn.model.modules():
    #     if isinstance(m, AdaRoundQuantizer):
    #         m.zero_point = nn.Parameter(m.zero_point)
    #         m.delta = nn.Parameter(m.delta)

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=opt.iters_w, weight=opt.weight, asym=True, batch_size=opt.batch_size, 
                  b_range=(opt.b_start, opt.b_end), warmup=opt.warmup, act_quant=False, opt_mode='mse')

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start calibration
    print('Start calibration ...')
    recon_model(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    if opt.quant_act:
        # Initialize activation quantization parameters
        print("Initializing act quantization parameters")
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            if not opt.cond:
                _ = qnn(cali_data[0][:4].to(device), cali_data[1][:4].to(device))
            else:
                _ = qnn(cali_data[0][:4].to(device), cali_data[1][:4].to(device), cali_data[2][:4].to(device))
        
        # for m in qnn.model.modules():
        #     if isinstance(m, AdaRoundQuantizer):
        #         m.zero_point = nn.Parameter(m.zero_point)
        #         m.delta = nn.Parameter(m.delta)
        #     elif isinstance(m, UniformAffineQuantizer):
        #         if m.zero_point is not None:
        #             if not torch.is_tensor(m.zero_point):
        #                 m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
        #             else:
        #                 m.zero_point = nn.Parameter(m.zero_point)
        
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, batch_size=opt.batch_size, 
                      iters=opt.iters_a, act_quant=True, opt_mode='mse', lr=opt.lr, p=opt.p)
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
    torch.save(qnn, save_path)
    print('Saved quantized model to ', save_path)
    