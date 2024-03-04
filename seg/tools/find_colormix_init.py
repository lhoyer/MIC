from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
from torch import nn, optim
import argparse
import wandb
import os
import cv2


def get_options():
    parser = argparse.ArgumentParser(
        description="Options for finding optimal parameters."
    )

    # Add arguments
    parser.add_argument(
        "--experiment", type=str, help="Threshold value.", default="hcp"
    )
    parser.add_argument("--n_epochs", type=int, help="Threshold value.", default=1)
    parser.add_argument("--batch_size", type=int, help="Threshold value.", default=2048)
    parser.add_argument(
        "--learning_rate", type=float, help="Threshold value.", default=1e-3
    )
    parser.add_argument(
        "--auto_bcg",
        type=int,
        default=0,
        help="1 if you want to use algorithm for automatic background removal.",
    )
    parser.add_argument(
        "--work_dir", type=str, help="Workdir for weights and biases.", 
        default="/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


class UnpairedDataset(Dataset):
    def __init__(
        self, path, src_dataset, tgt_dataset, volume_size_src, volume_size_tgt, auto_bcg
    ):
        """
        Initializes the dataset with source and target numpy arrays.
        Args:
        """

        self.auto_bcg = auto_bcg
        if self.auto_bcg:
            print("Using automatic background removal!")

        self.src_data = self.get_val(src_dataset, path, volume_size_src)
        self.tgt_data = self.get_val(tgt_dataset, path, volume_size_tgt)

        # Shuffle the initial data
        self.shuffle_data()

    def get_val(self, dataset, path, volume_size):
        src_val = []
        files = sorted(
            [
                f.strip(".png")
                for f in os.listdir(f"{path}/{dataset}/images/train/")
                if ".png" in f
            ]
        )

        for i in range(volume_size):
            idx = files[i]
            img = Image.open(f"{path}/{dataset}/images/train/{idx}.png").convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0

            if self.auto_bcg:
                masks = self.find_background(img)
            else:
                masks = Image.open(
                    f"{path}/{dataset}/labels/train/{idx}_labelTrainIds.png"
                )
                masks = np.array(masks, dtype=np.int32)

            src_val.append(img[masks != 0].reshape(-1))

        return np.concatenate(src_val)

    def find_background(self, img):
        self.kernel_size=(3,3)
        # img.clamp_(0, 1)
        # img = img.permute(1, 2, 0).cpu().numpy()
        image = (img * 255).astype(np.uint8)

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Apply thresholding to find markers
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal using morphological closing operation
        kernel = np.ones(self.kernel_size, np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Background area determination
        sure_bg = cv2.dilate(closing, kernel, iterations=1)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply the watershed
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]

        means_per_marker = []
        for m in np.unique(markers):
            means_per_marker.append(img[markers == m].mean())

        foreground_id = np.argmax(means_per_marker)
        final_mask = np.zeros_like(markers)
        final_mask[markers == np.unique(markers)[foreground_id]] = 1

        return torch.Tensor(final_mask)

    def shuffle_data(self):
        """Shuffles the data, maintaining pairing between src and tgt."""
        indices_src = np.arange(self.src_data.shape[0])
        indices_tgt = np.arange(self.tgt_data.shape[0])
        np.random.shuffle(indices_src)
        np.random.shuffle(indices_tgt)
        self.src_data = self.src_data[indices_src]
        self.tgt_data = self.tgt_data[indices_tgt]

    def __len__(self):
        """Returns the number of batches in the dataset."""
        return max(self.src_data.shape[0], self.tgt_data.shape[0])

    def __getitem__(self, idx):
        """
        Retrieves a batch at the index `idx`.
        Args:
        - idx (int): Index of the batch to retrieve.

        Returns:
        - batch_src (torch.Tensor): Batch of source data.
        - batch_tgt (torch.Tensor): Batch of target data.
        """
        idx_src = idx % self.src_data.shape[0]
        idx_tgt = idx % self.tgt_data.shape[0]
        return torch.tensor(
            [self.src_data[idx_src]], dtype=torch.float32
        ), torch.tensor([self.tgt_data[idx_tgt]], dtype=torch.float32)


def train():
    args = get_options()
    print(args)

    if args.experiment == "hcp":
        path = "/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/"
        dataset_src = "hcp1"
        dataset_tgt = "hcp2"
        volume_size_src = 256
        volume_size_tgt = 256
    elif args.experiment == "abide":
        path = "/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/"
        dataset_src = "abide_caltech"
        dataset_tgt = "hcp2"
        volume_size_src = 256
        volume_size_tgt = 256
    elif args.experiment == "wmh":
        path = "/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/"
        dataset_src = "umc"
        dataset_tgt = "nuhs"
        volume_size_src = 48
        volume_size_tgt = 48
    elif args.experiment == "spine":
        path = "/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/lumbarspine/"
        dataset_src = "VerSe"
        dataset_tgt = "MRSpineSegV"
        volume_size_src = 120
        volume_size_tgt = 12
    else:
        raise ValueError(f"Unknown experiment {args.experiment}")

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    auto_bcg = bool(args.auto_bcg)

    if auto_bcg:
        tag = '-auto_bcg'
    else:
        tag = ''
    wandb_taks_name = f"{dataset_src}-{dataset_tgt}-{tag}"
    print(wandb_taks_name)
    wandb.init(project="MIC-init", 
               config=args, 
               dir=args.work_dir,
               name=wandb_taks_name)

    # Set random seed for reproducibility
    torch.manual_seed(0)

    criterion = MMDLoss()

    # Initialize the dataset and dataloader
    dataset = UnpairedDataset(
        path, dataset_src, dataset_tgt, volume_size_src, volume_size_tgt, auto_bcg
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    normalization_net = nn.Linear(1, 1)

    # Initialize the optimizer
    optimizer = optim.Adam(normalization_net.parameters(), lr=learning_rate)

    print("Starting training...")
    # Train the model
    loss_val = []
    local_iter = 0
    for epoch in range(n_epochs):
        for batch_src, batch_tgt in dataloader:
            # Zero the gradients
            optimizer.zero_grad()

            img_polished = normalization_net(batch_src)

            loss = criterion(img_polished, batch_tgt)

            loss.backward()
            optimizer.step()

            loss_val.append(loss.item())
            wandb.log({"loss": loss.item()})

            for name, param in normalization_net.named_parameters():
                wandb.log({name: param.data.item()}, step=local_iter + 1)

            local_iter += 1

    # fig = plt.figure()
    # plt.plot(loss_val)
    # plt.savefig("loss.png")
    for name, param in normalization_net.named_parameters():
        print(f"{name}={param.data.item():.4f}")
        wandb.log({f"Final: {name}": param.data.item()})

    wandb.finish()


if __name__ == "__main__":
    train()
