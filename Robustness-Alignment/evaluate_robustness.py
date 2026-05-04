"""CIFAR-10-C / CIFAR-100-C robustness evaluation.

Pulls the official Hendrycks & Dietterich (2019) tarball from Zenodo, extracts
just the requested .npy files, and sweeps every loaded checkpoint across
severity levels 1..5.

Each corruption file ships as 50000 images = 5 severities * 10000 test images,
laid out consecutively (severity 1 first). labels.npy is the same labels
repeated five times.
"""
import argparse, os, tarfile, urllib.request
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from utils.metrics import accuracy, AverageMeter


_F={"resnet20":resnet20, "resnet56":resnet56,
    "resnet110":resnet110, "mobilenetv2":mobilenetv2}

ALL_CORRUPTIONS=[
    "brightness","contrast","defocus_blur","elastic_transform","fog",
    "frost","gaussian_blur","gaussian_noise","glass_blur","impulse_noise",
    "jpeg_compression","motion_blur","pixelate","saturate","shot_noise",
    "snow","spatter","speckle_noise","zoom_blur",
]
# spatter is the closest official analog to "occlusion" — droplet artifacts
# that physically cover pixels. Override with --corruptions.
DEFAULT_CORRUPTIONS=["gaussian_noise","gaussian_blur","spatter"]

# Zenodo serves these as ~2.9GB tarballs. We download once, extract only the
# .npy files we need, then drop the tar.
ZENODO={
    "cifar10":  ("https://zenodo.org/records/2535967/files/CIFAR-10-C.tar",
                 "CIFAR-10-C.tar",  "CIFAR-10-C"),
    "cifar100": ("https://zenodo.org/records/3555552/files/CIFAR-100-C.tar",
                 "CIFAR-100-C.tar", "CIFAR-100-C"),
}


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _download(url, dst):
    print(f"  downloading {url}\n          to {dst}")
    last=[-1]
    def _hook(blocks, bs, total):
        if total<=0: return
        pct=int(100*blocks*bs/total)
        if pct!=last[0] and pct%5==0:
            mb=blocks*bs/1e6; tot=total/1e6
            print(f"    {pct:3d}%  ({mb:6.1f} / {tot:6.1f} MB)")
            last[0]=pct
    tmp=dst+".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    os.replace(tmp, dst)


def _extract(tar_path, target_dir, names):
    want={f"{n}.npy" for n in names}; got=set()
    print(f"  extracting {len(want)} files from {os.path.basename(tar_path)}")
    with tarfile.open(tar_path, "r") as tf:
        for mem in tf:
            if not mem.isfile(): continue
            base=os.path.basename(mem.name)
            if base in want:
                src=tf.extractfile(mem)
                if src is None: continue
                with open(os.path.join(target_dir, base), "wb") as out:
                    out.write(src.read())
                print(f"    + {base}")
                got.add(base)
                if got==want:
                    break
    miss=want-got
    if miss:
        raise RuntimeError(f"tar missing: {miss}; got {got}. delete {tar_path} and retry.")


def ensure_cifar_c(dataset, corruptions, data_dir, keep_tar=False):
    url, tar_name, dirname = ZENODO[dataset]
    target=os.path.join(data_dir, dirname)
    os.makedirs(target, exist_ok=True)

    needed=list(corruptions)+["labels"]
    for n in needed:
        if n not in ALL_CORRUPTIONS and n!="labels":
            raise ValueError(f"unknown corruption {n!r}; choose from: {', '.join(ALL_CORRUPTIONS)}")

    miss=[n for n in needed if not os.path.exists(os.path.join(target, f"{n}.npy"))]
    if not miss:
        return target

    tar=os.path.join(data_dir, tar_name)
    if not os.path.exists(tar):
        print(f"\nfetching official {dirname} from Zenodo (~2.9 GB, one-time)")
        _download(url, tar)
    _extract(tar, target, miss)
    if not keep_tar:
        os.remove(tar)
        print(f"  removed {tar} (use --keep_tar to retain ~2.9 GB)")
    return target


class CIFARC(Dataset):
    """One severity slice of an official CIFAR-C corruption file."""
    SLICE=10000

    def __init__(self, root, corruption, severity, mean, std):
        self.imgs=np.load(os.path.join(root, f"{corruption}.npy"))
        self.lab=np.load(os.path.join(root, "labels.npy"))
        if self.imgs.shape[0]!=5*self.SLICE:
            raise RuntimeError(f"{corruption}.npy shape {self.imgs.shape}, expected (50000,32,32,3)")
        lo=(severity-1)*self.SLICE; hi=severity*self.SLICE
        self.imgs=self.imgs[lo:hi]; self.lab=self.lab[lo:hi]
        self.norm=T.Normalize(mean, std)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x=torch.from_numpy(self.imgs[i]).permute(2,0,1).contiguous().float()/255.0
        return self.norm(x), int(self.lab[i])


class CleanCIFAR(Dataset):
    def __init__(self, dataset, data_dir, mean, std):
        import torchvision
        cls=(torchvision.datasets.CIFAR10 if dataset=="cifar10"
             else torchvision.datasets.CIFAR100)
        self.base=cls(root=data_dir, train=False, download=True, transform=None)
        self.tx=T.Compose([T.ToTensor(), T.Normalize(mean,std)])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x,y=self.base[i]
        return self.tx(x), y


def _loader(ds, bs, nw):
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=torch.cuda.is_available())


def _load_model(arch, nc, path, dev):
    if arch not in _F:
        raise ValueError(f"unknown arch {arch!r}")
    m=_F[arch](num_classes=nc).to(dev)
    if not os.path.exists(path):
        return None
    blob=torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def _eval(m, loader, dev):
    a=AverageMeter()
    for x,y in loader:
        x=x.to(dev, non_blocking=True); y=y.to(dev, non_blocking=True)
        a.update(accuracy(m(x), y, topk=(1,))[0], x.size(0))
    return a.avg


def _gather(args, dev, nc):
    cdir=args.checkpoint_dir
    specs=[(f"Teacher ({args.teacher})", args.teacher,
            os.path.join(cdir, f"{args.dataset}_{args.teacher}_teacher.pth")),
           ("Student (no KD)", args.student,
            os.path.join(cdir, f"{args.student}_no_kd_best.pth")),
           ("KL-KD",         args.student, os.path.join(cdir, "kl_kd_best.pth")),
           ("Fixed-OT-KD",   args.student, os.path.join(cdir, "sinkhorn_kd_best.pth")),
           ("Adaptive-OT-KD",args.student, os.path.join(cdir, "adaptive_sinkhorn_kd_best.pth"))]
    out={}
    for name, arch, path in specs:
        m=_load_model(arch, nc, path, dev)
        if m is None:
            print(f"[skip] {name}: no ckpt at {path}")
        else:
            out[name]=m
    return out


def print_table(res, sevs, corruptions):
    methods=list(res.keys())
    print("\n"+"="*110)
    print("ROBUSTNESS — CIFAR-C top-1 (%)")
    print("="*110)

    for corr in ["clean"]+corruptions:
        cols=["clean"] if corr=="clean" else [f"sev{s}" for s in sevs]
        head=f"{'method':<22} | "+" | ".join(f"{h:>7}" for h in cols)
        if corr!="clean":
            head+=f" | {'mean':>7} | {'drop':>7}"
        print(f"\n[{corr}]")
        print(head); print("-"*len(head))

        for m in methods:
            row=f"{m:<22}"
            if corr=="clean":
                v=res[m].get("clean")
                row+=f" | {v:>6.2f}%" if v is not None else f" | {'N/A':>7}"
            else:
                accs=[res[m][corr].get(s) for s in sevs]
                cells=[f"{v:>6.2f}%" if v is not None else f"{'N/A':>7}" for v in accs]
                ok=[float(a) for a in accs if a is not None]
                mu=sum(ok)/len(ok) if ok else None
                cl=res[m].get("clean")
                drop=(cl-mu) if (cl is not None and mu is not None) else None
                row+=" | "+" | ".join(cells)
                row+=f" | {mu:>6.2f}%" if mu is not None else f" | {'N/A':>7}"
                row+=f" | {drop:>6.2f}" if drop is not None else f" | {'N/A':>7}"
            print(row)
    print("="*110)


def parse_args():
    p=argparse.ArgumentParser(description="CIFAR-C robustness eval")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--teacher", default="resnet110")
    p.add_argument("--student", default="resnet20")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--severities", type=int, nargs="+", default=[1,2,3,4,5])
    p.add_argument("--corruptions", type=str, nargs="+", default=DEFAULT_CORRUPTIONS,
                   help=f"any of: {', '.join(ALL_CORRUPTIONS)}")
    p.add_argument("--keep_tar", action="store_true")
    p.add_argument("--save_csv", default=None)
    return p.parse_args()


def main():
    args=parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir=f"./checkpoints/{args.dataset}"
    for s in args.severities:
        if not 1<=s<=5:
            raise ValueError(f"severity must be in 1..5, got {s}")

    dev=_dev()
    nc=10 if args.dataset=="cifar10" else 100
    print(f"dev={dev} | dataset={args.dataset} | ckpts={args.checkpoint_dir}")

    print(f"\nchecking CIFAR-C files for {args.dataset} (corruptions: {', '.join(args.corruptions)})")
    cifar_c=ensure_cifar_c(args.dataset, args.corruptions, args.data_dir,
                           keep_tar=args.keep_tar)
    print(f"  ok — {cifar_c}")

    models=_gather(args, dev, nc)
    if not models:
        print("nothing to evaluate.")
        return

    res={m:{"clean":None, **{c:{} for c in args.corruptions}} for m in models}

    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if args.dataset=="cifar10" else (CIFAR100_MEAN, CIFAR100_STD)
    clean=_loader(CleanCIFAR(args.dataset, args.data_dir, mean, std),
                  args.batch_size, args.num_workers)
    print("\n[clean]")
    for name, m in models.items():
        a=_eval(m, clean, dev)
        res[name]["clean"]=a
        print(f"  {name:<22} {a:.2f}%")

    for corr in args.corruptions:
        for sev in args.severities:
            ds=CIFARC(cifar_c, corr, sev, mean, std)
            ldr=_loader(ds, args.batch_size, args.num_workers)
            print(f"\n[{corr} | severity {sev}]")
            for name, m in models.items():
                a=_eval(m, ldr, dev)
                res[name][corr][sev]=a
                print(f"  {name:<22} {a:.2f}%")

    print_table(res, args.severities, args.corruptions)

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w=csv.writer(f)
            w.writerow(["method","corruption","severity","top1_acc"])
            for m,d in res.items():
                if d.get("clean") is not None:
                    w.writerow([m, "clean", 0, d["clean"]])
                for corr in args.corruptions:
                    for sev, a in d[corr].items():
                        w.writerow([m, corr, sev, a])
        print(f"\ncsv -> {args.save_csv}")


if __name__=="__main__":
    main()
