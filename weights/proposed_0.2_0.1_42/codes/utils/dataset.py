import os
import glob
import random
from PIL import Image
import numpy as np
import pandas as pd
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, args, input_type='3d', dataset_type='train'):
        self.root_dir = root_dir
        self.scan_params = args.scan_params
        self.y_range = args.y_range
        self.depth = args.depth
        self.dataset_type = dataset_type

        self.is_recon_only = (args.lambda_c == 0.0 and args.lambda_t == 0.0)
        self.is_test_mode = (self.dataset_type not in ['train', 'val']) or args.repeat_test
        self.sample_5_scans = args.sample_5_scans
        self.repeat_test = args.repeat_test

        if self.dataset_type in os.path.basename(self.root_dir):
            base_root = self.root_dir
        elif self.repeat_test:
            base_root = self.root_dir
        else:
            base_root = os.path.join(self.root_dir, self.dataset_type)

        self.volume_root = os.path.join(base_root, '3d_volume')
        self.label_root = os.path.join(base_root, 'interfero')
        self.pixel_root = os.path.join(base_root, 'pixel_mask')
        self.height_root = os.path.join(base_root, 'height')

        if self.is_test_mode or self.is_recon_only:
            all_files = glob.glob(os.path.join(self.volume_root, '*.npy'))
            self.roi_list = [os.path.basename(f).replace('.npy', '') for f in all_files]
        else:
            all_files = glob.glob(os.path.join(self.volume_root, '*.npy'))
            base_names = set([os.path.basename(f).rsplit('-', 1)[0] for f in all_files])
            self.roi_list = list(base_names)

    def __len__(self):
        return len(self.roi_list)

    def __getitem__(self, idx):
        if self.is_test_mode or self.is_recon_only:
            file_id = self.roi_list[idx]

            # 1. Volume
            v_path = os.path.join(self.volume_root, f"{file_id}.npy")
            vol = np.load(v_path)
            volume_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)

            # 2. Interfero
            l_path = os.path.join(self.label_root, f"{file_id}.csv")
            l_array = np.loadtxt(l_path, delimiter=',', dtype=np.float32)
            pil_img = Image.fromarray(l_array, mode='F')

            if pil_img.size != (232, 232):
                l_array = np.array(pil_img.resize((232, 232), resample=Image.LANCZOS))

            label_tensor = torch.tensor(l_array, dtype=torch.float32).unsqueeze(0)

            # 3. Pixel Mask
            m_path = os.path.join(self.pixel_root, f"{file_id}.npy")
            m_array = np.load(m_path)
            pil_m = Image.fromarray(m_array.astype(np.int32), mode='I')

            if pil_m.size != (232, 232):
                m_array = np.array(pil_m.resize((232, 232), resample=Image.NEAREST))

            mask_tensor = torch.tensor(m_array.astype(np.int32), dtype=torch.long)

            # 4. Height
            height_path = os.path.join(self.height_root, f"{file_id}.csv")
            df = pd.read_csv(height_path)
            df = df[pd.to_numeric(df['idx'], errors='coerce').notnull()]
            height_values = torch.tensor(df['final_h'].astype(float).values, dtype=torch.float32)

            return volume_tensor, label_tensor, mask_tensor, height_values, file_id

        else:
            base_name = self.roi_list[idx]
            volumes, labels, masks = [], [], []

            existing_files = glob.glob(os.path.join(self.volume_root, f"{base_name}-*.npy"))
            existing_suffixes = [os.path.basename(f).replace('.npy', '').rsplit('-', 1)[1] for f in existing_files]
            existing_suffixes.sort(key=int)

            if self.sample_5_scans and len(existing_suffixes) > 5:
                num_range = random.sample(existing_suffixes, 5)
            else:
                num_range = existing_suffixes

            for i in num_range:
                file_id = f"{base_name}-{i}"

                v_path = os.path.join(self.volume_root, f"{file_id}.npy")
                vol = np.load(v_path)
                volumes.append(torch.tensor(vol, dtype=torch.float32).unsqueeze(0))

                l_path = os.path.join(self.label_root, f"{file_id}.csv")
                l_array = np.loadtxt(l_path, delimiter=',', dtype=np.float32)
                pil_img = Image.fromarray(l_array, mode='F')

                if pil_img.size != (232, 232):
                    l_array = np.array(pil_img.resize((232, 232), resample=Image.LANCZOS))

                labels.append(torch.tensor(l_array, dtype=torch.float32).unsqueeze(0))

                m_path = os.path.join(self.pixel_root, f"{file_id}.npy")
                m_array = np.load(m_path)
                pil_m = Image.fromarray(m_array.astype(np.int32), mode='I')

                if pil_m.size != (232, 232):
                    m_array = np.array(pil_m.resize((232, 232), resample=Image.NEAREST))

                masks.append(torch.tensor(m_array.astype(np.int32), dtype=torch.long))

            volume_tensor = torch.stack(volumes, dim=0)
            label_tensor = torch.stack(labels, dim=0)
            mask_tensor = torch.stack(masks, dim=0)

            first_valid_suffix = num_range[0]
            height_path = os.path.join(self.height_root, f"{base_name}-{first_valid_suffix}.csv")
            df = pd.read_csv(height_path)
            df = df[pd.to_numeric(df['idx'], errors='coerce').notnull()]
            height_values = torch.tensor(df['final_h'].astype(float).values, dtype=torch.float32)

            return volume_tensor, label_tensor, mask_tensor, height_values, base_name


def load_datasets(root_dir, args, is_train):
    if is_train:
        trainset = CustomDataset(root_dir, args, dataset_type='train')
        valset = CustomDataset(root_dir, args, dataset_type='val')
        return [trainset, valset]
    else:
        testset = CustomDataset(root_dir, args, dataset_type='test')
        return testset


def make_dataloaders(dataset, bn_size, is_train):
    if is_train:
        train_loader = torch.utils.data.DataLoader(dataset[0], pin_memory=True, batch_size=bn_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(dataset[1], pin_memory=True, batch_size=bn_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, persistent_workers=True)
        return train_loader, val_loader
    else:
        test_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=bn_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, persistent_workers=True)
        return test_loader


def custom_collate_fn(batch):
    volumes, labels, masks, heights, file_names = zip(*batch)

    volumes_stacked = torch.stack(volumes, 0)
    labels_stacked = torch.stack(labels, 0)
    masks_stacked = torch.stack(masks, 0)

    return volumes_stacked, labels_stacked, masks_stacked, heights, file_names


def save_config(config, save_dir):
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")