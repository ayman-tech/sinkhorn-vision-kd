"""
Robustness evaluation on the official CIFAR-10-C / CIFAR-100-C benchmarks
(Hendrycks & Dietterich 2019, https://arxiv.org/abs/1903.12261).

Each corruption file on Zenodo is a numpy array of shape (50000, 32, 32, 3)
in uint8 — it concatenates the 10000-image test set five times, one per
severity level. labels.npy gives the matching ground-truth labels (same
across severities). Using the official files means the numbers here are
directly comparable to published CIFAR-10-C / CIFAR-100-C results.

Default corruptions (matching the user's "Gaussian noise / blur / occlusion"
brief, mapped to the closest official CIFAR-C files):

  - gaussian_noise   exact match
  - gaussian_blur    exact match
  - spatter          closest analog to "occlusion" (splotchy droplet artifacts
                     that occlude pixels). Override via --corruptions.

Any of the 19 standard CIFAR-C corruptions can be passed via --corruptions,
e.g. --corruptions defocus_blur frost snow.

Usage:
    python evaluate_robustness.py --dataset cifar10
    python evaluate_robustness.py --dataset cifar100 --severities 1 3 5
    python evaluate_robustness.py --dataset cifar10 \
        --corruptions gaussian_noise motion_blur frost
"""

import argparse
import os
import tarfile
import urllib.request
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import (
    CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD,
)
from utils.metrics import accuracy, AverageMeter


MODEL_FACTORY = {
    "resnet20": resnet20,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "mobilenetv2": mobilenetv2,
}

# All 19 corruptions present on the Zenodo records below.
CIFAR_C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
]

# Default selection for this project (noise / blur / occlusion-like).
DEFAULT_CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "spatter"]

# Zenodo only ships these as tarballs (~2.9 GB each). We download once,
# extract just the requested .npy files, then delete the tar.
ZENODO_TARS = {
    "cifar10":  ("https://zenodo.org/records/2535967/files/CIFAR-10-C.tar",
                 "CIFAR-10-C.tar",  "CIFAR-10-C"),
    "cifar100": ("https://zenodo.org/records/3555552/files/CIFAR-100-C.tar",
                 "CIFAR-100-C.tar", "CIFAR-100-C"),
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── CIFAR-C downloader / loader ─────────────────────────────────────────────

def _download(url: str, dest: str) -> None:
    """Stream-download with a one-line progress indicator."""
    print(f"  downloading {url}")
    print(f"          to {dest}")
    last_pct = [-1]

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(100 * blocks * block_size / total)
        if pct != last_pct[0] and pct % 5 == 0:
            mb = blocks * block_size / 1e6
            tmb = total / 1e6
            print(f"    {pct:3d}%  ({mb:6.1f} / {tmb:6.1f} MB)")
            last_pct[0] = pct

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    os.replace(tmp, dest)


def _extract_npys(tar_path: str, target_dir: str, names: List[str]) -> None:
    """Extract just the requested .npy files from the tar into target_dir."""
    wanted = {f"{n}.npy" for n in names}
    found = set()
    print(f"  extracting {len(wanted)} files from {os.path.basename(tar_path)}")
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in wanted:
                # Strip any leading directory inside the tar (e.g. "CIFAR-10-C/").
                src = tf.extractfile(member)
                if src is None:
                    continue
                out_path = os.path.join(target_dir, base)
                with open(out_path, "wb") as out:
                    out.write(src.read())
                print(f"    + {base}")
                found.add(base)
                if found == wanted:
                    break
    missing = wanted - found
    if missing:
        raise RuntimeError(
            f"Tar missing files: {missing} (got {found}). "
            f"Tar may be corrupted — delete {tar_path} and retry."
        )


def ensure_cifar_c(dataset: str, corruptions: List[str], data_dir: str,
                   keep_tar: bool = False) -> str:
    """Make sure the requested CIFAR-C .npy files exist locally.

    The official Zenodo records ship a single ~2.9 GB tar per dataset.
    We download it once, extract only the corruption files we need plus
    labels.npy, then by default delete the tar to reclaim disk.
    """
    url, tar_name, dirname = ZENODO_TARS[dataset]
    target_dir = os.path.join(data_dir, dirname)
    os.makedirs(target_dir, exist_ok=True)

    needed = list(corruptions) + ["labels"]
    for name in needed:
        if name not in CIFAR_C_CORRUPTIONS and name != "labels":
            raise ValueError(
                f"Unknown CIFAR-C corruption '{name}'. "
                f"Choose from: {', '.join(CIFAR_C_CORRUPTIONS)}"
            )

    missing = [n for n in needed
               if not os.path.exists(os.path.join(target_dir, f"{n}.npy"))]
    if not missing:
        return target_dir

    tar_path = os.path.join(data_dir, tar_name)
    if not os.path.exists(tar_path):
        print(f"\nDownloading official {dirname} from Zenodo (~2.9 GB, one-time)")
        _download(url, tar_path)

    _extract_npys(tar_path, target_dir, missing)

    if not keep_tar:
        os.remove(tar_path)
        print(f"  removed {tar_path} to reclaim ~2.9 GB "
              f"(pass --keep_tar to retain)")
    return target_dir


class CIFARCDataset(Dataset):
    """One severity slice of an official CIFAR-C corruption file.

    The .npy file contains 50000 = 10000 (test set size) * 5 (severities)
    images stored consecutively, severity 1 first. Index range for
    severity s (1-based) is [(s-1)*10000 : s*10000].
    """

    SEVERITY_SIZE = 10000

    def __init__(self, root: str, corruption: str, severity: int,
                 mean, std):
        self.images = np.load(os.path.join(root, f"{corruption}.npy"))
        self.labels = np.load(os.path.join(root, "labels.npy"))
        if self.images.shape[0] != 5 * self.SEVERITY_SIZE:
            raise RuntimeError(
                f"{corruption}.npy has shape {self.images.shape}, "
                f"expected (50000, 32, 32, 3)"
            )
        lo = (severity - 1) * self.SEVERITY_SIZE
        hi = severity * self.SEVERITY_SIZE
        self.images = self.images[lo:hi]
        self.labels = self.labels[lo:hi]
        self.normalize = T.Normalize(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # uint8 HWC -> float32 CHW in [0,1] -> normalize
        img = self.images[idx]
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        x = self.normalize(x)
        return x, int(self.labels[idx])


# Clean test set still comes from torchvision (no corruption applied).
class CleanCIFAR(Dataset):
    def __init__(self, dataset: str, data_dir: str, mean, std):
        import torchvision
        cls = (torchvision.datasets.CIFAR10 if dataset == "cifar10"
               else torchvision.datasets.CIFAR100)
        self.base = cls(root=data_dir, train=False, download=True, transform=None)
        self.transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


def build_loader(ds: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    pin = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin)


# ─── Evaluation ─────────────────────────────────────────────────────────────

def load_model(arch: str, num_classes: int, ckpt_path: str, device):
    if arch not in MODEL_FACTORY:
        raise ValueError(f"Unknown arch: {arch}")
    model = MODEL_FACTORY[arch](num_classes=num_classes).to(device)
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def eval_loader(model: nn.Module, loader: DataLoader, device) -> float:
    top1 = AverageMeter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        acc = accuracy(logits, labels, topk=(1,))[0]
        top1.update(acc, images.size(0))
    return top1.avg


def collect_methods(args, device, num_classes):
    ckpt_dir = args.checkpoint_dir
    specs = [
        (f"Teacher ({args.teacher})", args.teacher,
         os.path.join(ckpt_dir, f"{args.dataset}_{args.teacher}_teacher.pth")),
        ("Student (no KD)", args.student,
         os.path.join(ckpt_dir, f"{args.student}_no_kd_best.pth")),
        ("KL-KD", args.student,
         os.path.join(ckpt_dir, "kl_kd_best.pth")),
        ("Fixed-OT-KD", args.student,
         os.path.join(ckpt_dir, "sinkhorn_kd_best.pth")),
        ("Adaptive-OT-KD", args.student,
         os.path.join(ckpt_dir, "adaptive_sinkhorn_kd_best.pth")),
    ]
    models = {}
    for name, arch, path in specs:
        m = load_model(arch, num_classes, path, device)
        if m is None:
            print(f"[skip] {name}: checkpoint not found at {path}")
        else:
            models[name] = m
    return models


def print_table(results: dict, severities: List[int], corruptions: List[str]):
    methods = list(results.keys())
    print("\n" + "=" * 110)
    print("ROBUSTNESS — CIFAR-C top-1 accuracy (%)")
    print("=" * 110)

    for corr in ["clean"] + corruptions:
        header_sevs = ["clean"] if corr == "clean" else [f"sev{s}" for s in severities]
        header = f"{'Method':<22} | " + " | ".join(f"{h:>7}" for h in header_sevs)
        if corr != "clean":
            header += f" | {'mean':>7} | {'drop':>7}"
        print(f"\n[{corr}]")
        print(header)
        print("-" * len(header))

        for m in methods:
            row = f"{m:<22}"
            if corr == "clean":
                v = results[m].get("clean")
                row += f" | {v:>6.2f}%" if v is not None else f" | {'N/A':>7}"
            else:
                accs = [results[m][corr].get(s) for s in severities]
                cells = [f"{v:>6.2f}%" if v is not None else f"{'N/A':>7}" for v in accs]
                valid = [float(a) for a in accs if a is not None]
                mean_acc = sum(valid) / len(valid) if valid else None
                clean = results[m].get("clean")
                drop = (clean - mean_acc) if (clean is not None and mean_acc is not None) else None
                row += " | " + " | ".join(cells)
                row += f" | {mean_acc:>6.2f}%" if mean_acc is not None else f" | {'N/A':>7}"
                row += f" | {drop:>6.2f}" if drop is not None else f" | {'N/A':>7}"
            print(row)
    print("=" * 110)


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-C robustness evaluation")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Defaults to ./checkpoints/<dataset>")
    p.add_argument("--teacher", type=str, default="resnet110")
    p.add_argument("--student", type=str, default="resnet20")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--corruptions", type=str, nargs="+",
                   default=DEFAULT_CORRUPTIONS,
                   help=f"Any of: {', '.join(CIFAR_C_CORRUPTIONS)}")
    p.add_argument("--keep_tar", action="store_true",
                   help="Keep the downloaded ~2.9 GB CIFAR-C tar after extraction.")
    p.add_argument("--save_csv", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints/{args.dataset}"

    for s in args.severities:
        if not 1 <= s <= 5:
            raise ValueError(f"severities must be in 1..5, got {s}")

    device = get_device()
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"Device: {device} | dataset: {args.dataset} | "
          f"ckpt_dir: {args.checkpoint_dir}")

    # 1. Make sure CIFAR-C files are local (downloads to data_dir if missing).
    print(f"\nChecking CIFAR-C files for {args.dataset} "
          f"(corruptions: {', '.join(args.corruptions)})")
    cifar_c_root = ensure_cifar_c(args.dataset, args.corruptions,
                                  args.data_dir, keep_tar=args.keep_tar)
    print(f"  ok — CIFAR-C in {cifar_c_root}")

    # 2. Load all available checkpoints.
    models = collect_methods(args, device, num_classes)
    if not models:
        print("No checkpoints found — nothing to evaluate.")
        return

    results = {m: {"clean": None, **{c: {} for c in args.corruptions}}
               for m in models}

    # 3. Clean accuracy for reference.
    if args.dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD

    clean_loader = build_loader(
        CleanCIFAR(args.dataset, args.data_dir, mean, std),
        args.batch_size, args.num_workers,
    )
    print("\n[clean]")
    for name, model in models.items():
        acc = eval_loader(model, clean_loader, device)
        results[name]["clean"] = acc
        print(f"  {name:<22} {acc:.2f}%")

    # 4. Corruption sweep — official CIFAR-C images, no on-the-fly noise.
    for corr in args.corruptions:
        for sev in args.severities:
            ds = CIFARCDataset(cifar_c_root, corr, sev, mean, std)
            loader = build_loader(ds, args.batch_size, args.num_workers)
            print(f"\n[{corr} | severity {sev}]")
            for name, model in models.items():
                acc = eval_loader(model, loader, device)
                results[name][corr][sev] = acc
                print(f"  {name:<22} {acc:.2f}%")

    print_table(results, args.severities, args.corruptions)

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "corruption", "severity", "top1_acc"])
            for m, d in results.items():
                if d.get("clean") is not None:
                    w.writerow([m, "clean", 0, d["clean"]])
                for corr in args.corruptions:
                    for sev, acc in d[corr].items():
                        w.writerow([m, corr, sev, acc])
        print(f"\nSaved CSV to {args.save_csv}")


if __name__ == "__main__":
    main()
