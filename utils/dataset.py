import os
import glob
import random
from PIL import Image
import numpy as np
import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, args, input_type='3d', dataset_type='train'):
        self.root_dir = root_dir
        self.scan_params = args.scan_params
        self.y_range = args.y_range
        self.depth = args.depth
        self.dataset_type = dataset_type

        self.is_recon_only = (args.lambda_c == 0.0 and args.lambda_t == 0.0)
        self.is_test_mode = self.dataset_type not in ['train', 'val']
        self.sample_5_scans = args.sample_5_scans
        self.repeat_test = args.repeat_test

        if self.dataset_type in os.path.basename(os.path.normpath(self.root_dir)):
            base_root = self.root_dir
        elif self.repeat_test:
            base_root = self.root_dir
        else:
            base_root = os.path.join(self.root_dir, self.dataset_type)

        self.volume_root = os.path.join(base_root, '3d_volume')
        self.label_root = os.path.join(base_root, 'interfero')

        all_files = glob.glob(os.path.join(self.volume_root, '*.npy'))

        if self.is_test_mode or self.is_recon_only:
            self.roi_list = sorted(
                os.path.splitext(os.path.basename(f))[0]
                for f in all_files
            )
        else:
            base_names = {
                os.path.basename(f).rsplit('-', 1)[0]
                for f in all_files
            }
            self.roi_list = sorted(base_names)

    def __len__(self):
        return len(self.roi_list)

    def _load_volume(self, file_id):
        v_path = os.path.join(self.volume_root, f"{file_id}.npy")
        vol = np.load(v_path).astype(np.float32, copy=False)
        return torch.from_numpy(vol).unsqueeze(0)

    def _load_label(self, file_id):
        l_path = os.path.join(self.label_root, f"{file_id}.csv")
        if self.is_test_mode and not os.path.isfile(l_path):
            return torch.full((1, 232, 232), float('nan'))
        l_array = np.loadtxt(l_path, delimiter=',', dtype=np.float32)

        if l_array.shape != (232, 232):
            pil_img = Image.fromarray(l_array, mode='F')
            l_array = np.array(
                pil_img.resize((232, 232), resample=Image.LANCZOS),
                dtype=np.float32
            )

        return torch.from_numpy(l_array).unsqueeze(0)

    def _load_height(self, file_id):
        height_path = os.path.join(self.height_root, f"{file_id}.csv")
        df = pd.read_csv(height_path)
        df = df[pd.to_numeric(df['idx'], errors='coerce').notnull()]

        height_values = df['final_h'].astype(float).values.astype(np.float32)
        return torch.from_numpy(height_values)

    def __getitem__(self, idx):
        if self.is_test_mode or self.is_recon_only:
            file_id = self.roi_list[idx]

            volume_tensor = self._load_volume(file_id)
            label_tensor = self._load_label(file_id)

            return volume_tensor, label_tensor, file_id

        else:
            base_name = self.roi_list[idx]
            volumes, labels = [], []

            existing_files = glob.glob(
                os.path.join(self.volume_root, f"{base_name}-*.npy")
            )
            existing_suffixes = [
                os.path.splitext(os.path.basename(f))[0].rsplit('-', 1)[1]
                for f in existing_files
            ]
            existing_suffixes.sort(key=int)

            if self.sample_5_scans and len(existing_suffixes) > 5:
                num_range = random.sample(existing_suffixes, 5)
            else:
                num_range = existing_suffixes

            for i in num_range:
                file_id = f"{base_name}-{i}"

                volumes.append(self._load_volume(file_id))
                labels.append(self._load_label(file_id))

            volume_tensor = torch.stack(volumes, dim=0)
            label_tensor = torch.stack(labels, dim=0)

            return volume_tensor, label_tensor, base_name


def load_datasets(root_dir, args, is_train):
    if is_train:
        trainset = CustomDataset(root_dir, args, dataset_type='train')
        valset = CustomDataset(root_dir, args, dataset_type='val')
        return [trainset, valset]
    else:
        testset = CustomDataset(root_dir, args, dataset_type='test')
        return testset


def make_dataloaders(dataset, bn_size, is_train):
    loader_kwargs = dict(
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn
    )

    if is_train:
        train_loader = torch.utils.data.DataLoader(
            dataset[0],
            batch_size=bn_size,
            shuffle=True,
            **loader_kwargs
        )
        val_loader = torch.utils.data.DataLoader(
            dataset[1],
            batch_size=bn_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bn_size,
            shuffle=False,
            **loader_kwargs
        )
        return test_loader


def custom_collate_fn(batch):
    volumes, labels, file_names = zip(*batch)
    return torch.stack(volumes, 0), torch.stack(labels, 0), file_names


def save_config(config, save_dir):
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")
