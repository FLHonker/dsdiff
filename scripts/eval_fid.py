import os
from PIL import Image
import numpy as np
import torch
from torchvision.datasets import ImageNet
from torchvision.transforms import transforms
import torch_fidelity
from ldm.data.in1k_label import class_names
from ldm.data import lsun

def register_in1k():
    # 定义ImageNet数据集的transform
    transform = transforms.Compose(
        [
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ]
    )

    class ImageNetNoLabel(ImageNet):
        def __init__(self, root, split='val', transform=None, **kwargs):
            super(ImageNetNoLabel, self).__init__(
                root=root,
                split=split,
                transform=transform,
                **kwargs
            )

        def __getitem__(self, index):
            image = super(ImageNetNoLabel, self).__getitem__(index)[0]
            image = (image * 255).to(torch.uint8)
            return image
    
    # 加载ImageNet验证集数据集，只选取每个类别下的前5张图片
    in1k_val = ImageNetNoLabel(
        root='data/imagenet',
        split='val',
        transform=transform
    )

    # 只选取每个类别下的前5张图片
    class_dict = {}
    for i in range(len(in1k_val)):
        _, class_id = in1k_val.samples[i]
        if class_id not in class_dict:
            class_dict[class_id] = []
        if len(class_dict[class_id]) < 5:
            class_dict[class_id].append(i)
            
    in1k_val = torch.utils.data.Subset(in1k_val, [item for sublist in class_dict.values() for item in sublist])
    # torch_fidelity.register_dataset('in1k_val', in1k_val)
    return in1k_val

def gen_prompt(path='ldm/data/in1k_prompt.txt'):
    prompts = []
    for name in class_names.values():
        name = name.split(',')[0]
        prompts.append(f"A high quality photograph of a {name}")
    with open(path, 'w') as f:
        for i in prompts:
            f.write(i + '\n')
    print('Generted prompts and saved to file!')


def register_lsun():
    class LSUNBaseTensor(lsun.LSUNBase):
        def __getitem__(self, i):
            example = dict((k, self.labels[k][i]) for k in self.labels)
            image = Image.open(example["file_path_"])
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

            image = Image.fromarray(img)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=self.interpolation)

            # Since flip_p is set to 0, we do not flip and remove this function call.
            # image = self.flip(image)
            image = np.array(image).astype(np.uint8)
            
            # Tensor with dtype uint8, original shape: (256, 256, 3), output shape: (3, 256, 256)
            example = torch.tensor(image).permute(2, 0, 1)
            return example
        
        def __len__(self):
            # Perform evaluation on subsamples
            # return 500000
            return super().__len__()

    class LSUNBedroomsTensorTrain(LSUNBaseTensor):
        def __init__(self, **kwargs):
            super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


    lsun_beds256_train = lambda root, download: LSUNBedroomsTensorTrain(
                    size=256,
                    interpolation="bicubic",
                    flip_p=0.)
        
    torch_fidelity.register_dataset('lsun_beds256_train', lsun_beds256_train)


if __name__ == "__main__":
    img_path = 'outdir/2023-07-12-14-10-35/samples'
    cache_root = 'outdir/cache/'
    torch.set_grad_enabled(False)
    # gen_prompt()
    val_data = register_in1k()
    # 计算FID得分
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=img_path, # fill this with your generated images path
        input2=val_data,
        batch_size=256, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=False,
        cache_root=cache_root, # fill this with your own path
        cache=True,
        verbose=True,
        samples_find_deep=True,
        samples_find_ext="jpg,jpeg,png,webp"
    )

    print(metrics_dict)
    