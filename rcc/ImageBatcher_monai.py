import os
import sys
import monai
import torch
from pathlib import Path
#from rcc_common import LoadImage, SaveImage, LoadMask, ShowWarnings
from monai.data import Dataset, DataLoader
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    ConcatItemsd,
)

def _batcher(dataloader, batch_size):
    for batch in dataloader:
        image, mask = batch["image"], batch["mask"]

        for i in range(0, image.shape[0], batch_size):
            begin = i
            end = min(begin+batch_size, image.shape[0])
            yield image[begin:end, ...], mask[begin:end, ...].type(torch.long).squeeze(1)

def load_list(path):
    if isinstance(path, str) or isinstance(path, Path):
        with open(path, mode="rt", newline="") as f:
            return [ line.strip() for line in f if len(line.strip()) > 0 ]

    assert hasattr(path, "__iter__")

    return path

def strip_ext(path):
    if path.lower().endswith(".nii.gz"):
        return path[:-7]

    return os.path.splitext(path)[0]

def load_label_map(label_path):
    label_map = dict()

    with open(label_path, mode="rt", newline="") as f:
        for line in f:
            line = line.strip().split("#")[0]

            tokens = [ token for token in line.split(" ") if len(token) > 0 ]

            if len(tokens) < 8:
                continue

            label_name = " ".join(tokens[7:]).lower()
            label = int(tokens[0])

            if label_name.startswith('"'):
                label_name = label_name[1:-1]

            if label_name == "clear label" or label_name.startswith("label"):
                continue

            new_label = -1

            if "kid" in label_name:
                new_label = 1
            elif "cy" in label_name:
                new_label = 2
            elif label_name.startswith("rk") or label_name.startswith("lk"):
                new_label = 3
            elif "un" in label_name:
                new_label = -1
            else:
                print(f"Error: Unnamed label '{label_name}' with label value {label}: {label_path}", flush=True, file=sys.stderr)

            label_map[label] = new_label

    return label_map

def get_roi_1d(size, modulus):
    remainder = size % modulus

    begin = remainder // 2
    end = begin + size - remainder

    return begin, end

def crop_modulo_n(data, modulus):
    xbegin, xend = get_roi_1d(data.shape[-2], modulus)
    ybegin, yend = get_roi_1d(data.shape[-1], modulus)

    return data[..., xbegin:xend, ybegin:yend]

def pad_modulo_n(data, shape, modulus):
    xbegin, xend = get_roi_1d(shape[-2], modulus)
    ybegin, yend = get_roi_1d(shape[-1], modulus)

    newData = torch.zeros(list(data.shape[:-2]) + list(shape[-2:]), dtype=data.dtype, device=data.device)
    newData[..., xbegin:xend, ybegin:yend] = data

    return newData

class LoadMask(LoadImage):
    def __call__(self, data):
        label_path = strip_ext(data) + ".txt"

        data = super().__call__(data)

        assert os.path.exists(label_path)

        mask = data
        new_mask = torch.zeros_like(mask)

        label_map = load_label_map(label_path)

        for label in torch.unique(mask):
            if label > 0 and int(label) not in label_map:
                print(f"Error: Undescribed label with value {label}: {label_path}", flush=True, file=sys.stderr)
                new_mask[mask == label] = -1

        for label, new_label in label_map.items():
            new_mask[mask == label] = new_label

        return monai.data.MetaTensor(new_mask, meta=mask.meta)

class LoadMaskd(LoadImaged):
    def __call__(self, data):
        d = dict(data)
        label_paths = [ strip_ext(d[key]) + ".txt" for key in self.key_iterator(d) ]

        d = super().__call__(d)

        for label_path, key in zip(label_paths, self.key_iterator(d)):
            assert os.path.exists(label_path)

            mask = d[key]
            new_mask = torch.zeros_like(mask)

            label_map = load_label_map(label_path)

            for label in torch.unique(mask):
                if label > 0 and int(label) not in label_map:
                    print(f"Error: Undescribed label with value {label}: {label_path}", flush=True, file=sys.stderr)
                    new_mask[mask == label] = -1

            for label, new_label in label_map.items():
                new_mask[mask == label] = new_label

            d[key] = monai.data.MetaTensor(new_mask, meta=mask.meta)

        return d

class KeepKeys(monai.transforms.MapTransform):
    def __call__(self, data):
        d = dict()

        for key in self.key_iterator(data):
            d[key] = data[key]

        return d

class SliceImage(monai.transforms.Transform):
    def __init__(self, axis=-1):
        self.axis=axis

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]

        slices = [slice(None)]*data.ndim

        return [ data[slices] for slices[self.axis] in range(data.shape[self.axis]) ]

class SliceImaged(monai.transforms.MapTransform):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis=axis

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]

        d = dict(data)

        for key in self.key_iterator(d):
            #print(f"Key: {key}", flush=True)
            slices = [slice(None)]*d[key].ndim
            d[key] = [ d[key][slices] for slices[self.axis] in range(d[key].shape[self.axis]) ]


        dict_list = [ {k: d[k][i] for k in self.key_iterator(d)} for i in range(len(d[key])) ]

        # Copy everything else
        for k in d:
            if k not in dict_list[0]:
                for item in dict_list:
                    item[k] = d[k]

        return dict_list

class SelectAnnotatedHalvesd(monai.transforms.MapTransform):
    small = 7**3

    def __init__(self, label_key="label", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.label_key]
        halfX = mask.shape[-3]//2

        right_mask = mask[..., :halfX, :, :]
        left_mask = mask[..., halfX:, :, :]

        right_side = (right_mask > 0).sum() > self.small
        left_side = (left_mask > 0).sum() > self.small

        assert left_side + right_side > 0

        if right_side + left_side == 2:
            pass # Nothing to do
        elif right_side:
            for key in self.key_iterator(d):
                d[key] = d[key][..., :halfX, :, :]
        else:
            for key in self.key_iterator(d):
                d[key] = d[key][..., halfX:, :, :]

        return d

class ThresholdCystsd(monai.transforms.MapTransform):
    def __init__(self, image_key="image", label_key="label"):
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)

        image = d[self.image_key]
        mask = d[self.label_key].squeeze(dim=0)

        mask[torch.logical_and(image[0, ...] < 253, mask == 1)] = -1

        for c in range(image.shape[0]):
            mask[image[c, ...] < -188] = -1

        d[self.image_key] = image
        d[self.label_key] = mask.unsqueeze(dim=0)

        return d

# inverse() doesn't work!
class CropModuloN(monai.transforms.Transform):
    def __init__(self, modulus):
        self.modulus = modulus

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]

        newData = crop_modulo_n(data, self.modulus)

        return newData

class CropModuloNd(monai.transforms.MapTransform):
    def __init__(self, modulus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulus = modulus

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = crop_modulo_n(d[key], self.modulus)

        return d

class PadModuloN(monai.transforms.Transform):
    def __init__(self, shape, modulus):
        self.modulus = modulus
        self.shape = shape

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]

        newData = pad_modulo_n(data, self.shape, self.modulus)

        return newData

class PadModuloNd(monai.transforms.MapTransform):
    def __init__(self, shape, modulus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulus = modulus
        self.shape = shape

    def __call__(self, data):
        if isinstance(data, list):
            return [ self.__call__(d) for d in data ]
    
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = pad_modulo_n(d[key], self.shape, self.modulus)

        return d

class ImageBatcher:
    def __init__(self, dataRoot, listFile, batchSize, numClasses=4, seed=6112, dilateUnknown=False):
        self.dataRoot = dataRoot
        self.multipleOf = 16
        self.numChannels = 4
        self.batchSize = batchSize
        self.numClasses = numClasses
        self.dilateUnknown = dilateUnknown
        self.patientIds = load_list(listFile)

        if seed is not None:
            print(f"Info: Setting global seed = {seed} ...")
            monai.utils.set_determinism(seed)

        imageDirs = [ "Images" ]*self.numChannels
        maskDirs = [ "Masks" ]

        imageBases = [ f"normalized{i+1}_aligned.nii.gz" if i > 0 else "normalized_aligned.nii.gz" for i in range(self.numChannels) ]
        maskBases = [ "mask_aligned.nii.gz" ]
        imageKeys = [ f"image{i+1}" if i > 0 else "image" for i in range(self.numChannels) ]
        maskKeys = [ "mask" ]

        imageFiles = [ {key: os.path.join(self.dataRoot, directory, patientId, base) for key, base, directory in zip(imageKeys + maskKeys, imageBases + maskBases, imageDirs + maskDirs)} for patientId in self.patientIds ]

        transforms = Compose([
            LoadImaged(keys=imageKeys, ensure_channel_first=True, image_only=True),
            LoadMaskd(keys=maskKeys, ensure_channel_first=True, image_only=True),
            ConcatItemsd(keys=imageKeys, name="image"),
            KeepKeys(keys=["image", "mask"]),
            ThresholdCystsd(image_key="image", label_key="mask"),
            SelectAnnotatedHalvesd(keys=["image", "mask"], label_key="mask"),
            SliceImaged(keys=["image", "mask"]),
            CropModuloNd(keys=["image", "mask"], modulus=self.multipleOf),
        ])

        #collate_fn = lambda batch : pad_list_data_collate(batch, mode="constant", constant_values=(-1,))

        def my_collate(batch):
            if isinstance(batch[0], list):
                batch = sum(batch, [])
            return pad_list_data_collate(batch, mode="constant", constant_values=(-1,))

        dataset = Dataset(data=imageFiles, transform=transforms)
        
        self.dataloader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True, num_workers=min(8, self.batchSize), collate_fn=my_collate)

    def start(self):
        pass

    def stop(self):
        pass

    def __iter__(self):
        return _batcher(self.dataloader, self.batchSize)
        #return self.dataloader.__iter__()

if __name__ == "__main__":
    #dataRoot="/data/AIR/RCC/NiftiCombined"
    dataRoot="/scratch/cluster_scratch/layns/RCC/NiftiNew"
    #dataRoot="/lscratch/38110568/NiftiCombined"
    #listFile=os.path.join(dataRoot, "all_randomSplit1.txt")

    listFile = [ load_list(os.path.join(dataRoot, base)) for base in ("train_isbi2022_easyhard_randomSplit1.txt","test_isbi2022_easyhard_randomSplit1.txt") ]
    listFile = sum(listFile, [])


    batcher = ImageBatcher(dataRoot, listFile, 32, numClasses=4, dilateUnknown=True)
    batcher.start()

    #ShowWarnings(True)

    #batcher.seed(7271)

    for batch in batcher:
        imageBatch, maskBatch = batch

        print(f"{type(imageBatch)}, {type(maskBatch)}")
        print(f"{imageBatch.shape}, {maskBatch.shape}")

    batcher.stop()

