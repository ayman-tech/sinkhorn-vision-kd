"""Teacher-student alignment metrics on the test set.

per method: KL(p_T||p_S), Sinkhorn W_eps, cosine(z_T, z_S), top-1 / top-5 agree.
Sinkhorn always uses a uniform cost so the number is comparable regardless of
which cost the method actually trained with.
"""
import argparse, os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import get_cifar_loaders
from utils.metrics import AverageMeter
from distillation.sinkhorn_distill import log_sinkhorn, build_cost_matrix


_F={"resnet20":resnet20, "resnet56":resnet56,
    "resnet110":resnet110, "mobilenetv2":mobilenetv2}


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(arch, nc, path, dev):
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
def alignment_metrics(student, teacher, loader, dev, T, C, eps, max_iter):
    kl_m=AverageMeter(); w_m=AverageMeter(); cos_m=AverageMeter()
    a1=AverageMeter(); a5=AverageMeter()
    teacher.eval(); student.eval()

    for x,_ in loader:
        x=x.to(dev, non_blocking=True)
        zT=teacher(x); zS=student(x)

        pT=F.softmax(zT/T, dim=1).clamp(min=1e-8)
        pS=F.softmax(zS/T, dim=1).clamp(min=1e-8)
        lpT=pT.log(); lpS=pS.log()

        # Multiply by T^2 to match the standard Hinton-KD scaling — keeps
        # the number on the same scale as the training loss.
        kl_per=(pT*(lpT-lpS)).sum(dim=1)
        kl_m.update((kl_per.mean()*(T*T)).item(), x.size(0))

        w_cost,_=log_sinkhorn(lpT, lpS, C, epsilon=eps, max_iter=max_iter, threshold=1e-3)
        w_m.update(w_cost.item(), x.size(0))

        cos_m.update(F.cosine_similarity(zT, zS, dim=1).mean().item(), x.size(0))

        pT_arg=zT.argmax(dim=1)
        a1.update((zS.argmax(dim=1)==pT_arg).float().mean().item()*100.0, x.size(0))

        k=min(5, zS.size(1))
        in_top5=(zS.topk(k, dim=1).indices==pT_arg.unsqueeze(1)).any(dim=1)
        a5.update(in_top5.float().mean().item()*100.0, x.size(0))

    return {"kl":kl_m.avg, "wasserstein":w_m.avg, "cosine":cos_m.avg,
            "top1_agree":a1.avg, "top5_agree":a5.avg}


def collect_students(args, dev, nc):
    cdir=args.checkpoint_dir
    specs=[("Student (no KD)",  args.student, os.path.join(cdir, f"{args.student}_no_kd_best.pth")),
           ("KL-KD",            args.student, os.path.join(cdir, "kl_kd_best.pth")),
           ("Fixed-OT-KD",      args.student, os.path.join(cdir, "sinkhorn_kd_best.pth")),
           ("Adaptive-OT-KD",   args.student, os.path.join(cdir, "adaptive_sinkhorn_kd_best.pth"))]
    out={}
    for name, arch, path in specs:
        m=_load(arch, nc, path, dev)
        if m is None:
            print(f"[skip] {name}: no ckpt at {path}")
        else:
            out[name]=m
    return out


def print_table(res):
    print("\n"+"="*92)
    print("TEACHER-STUDENT ALIGNMENT (test set)")
    print("="*92)
    head=(f"{'method':<22} | {'KL(T||S)':>10} | {'W_eps':>10} | "
          f"{'cos(zT,zS)':>13} | {'top-1 agree':>12} | {'top-5 agree':>12}")
    print(head); print("-"*len(head))
    for name, m in res.items():
        print(f"{name:<22} | {m['kl']:>10.4f} | {m['wasserstein']:>10.4f} | "
              f"{m['cosine']:>13.4f} | {m['top1_agree']:>11.2f}% | {m['top5_agree']:>11.2f}%")
    print("="*92)
    print("lower KL/W = closer; higher cosine/agree = closer.")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--teacher", default="resnet110")
    p.add_argument("--student", default="resnet20")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--sinkhorn_iter", type=int, default=50)
    p.add_argument("--save_csv", default=None)
    return p.parse_args()


def main():
    args=parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir=f"./checkpoints/{args.dataset}"
    dev=_dev()
    nc=10 if args.dataset=="cifar10" else 100
    print(f"dev={dev} | dataset={args.dataset} | ckpts={args.checkpoint_dir}")

    tpath=os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.teacher}_teacher.pth")
    teacher=_load(args.teacher, nc, tpath, dev)
    if teacher is None:
        raise FileNotFoundError(f"teacher ckpt missing: {tpath}")

    students=collect_students(args, dev, nc)
    if not students:
        print("no student ckpts found.")
        return

    _,_,te=get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=torch.cuda.is_available())

    C=build_cost_matrix(nc, "uniform", device=dev)

    res: Dict[str, Dict[str, float]] = {}
    for name, m in students.items():
        print(f"\n[evaluating] {name}")
        res[name]=alignment_metrics(m, teacher, te, dev,
                                    T=args.temperature, C=C,
                                    eps=args.epsilon, max_iter=args.sinkhorn_iter)
    print_table(res)

    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w=csv.writer(f)
            w.writerow(["method","kl","wasserstein","cosine","top1_agree","top5_agree"])
            for name, m in res.items():
                w.writerow([name, m["kl"], m["wasserstein"], m["cosine"],
                            m["top1_agree"], m["top5_agree"]])
        print(f"\ncsv -> {args.save_csv}")


if __name__=="__main__":
    main()
