"""Aggregate eval: load every checkpoint we can find, print a table, dump plots.

Also supports --run_seeds to retrain everything across multiple seeds.
"""
import argparse, os, glob
import numpy as np
import torch
import torch.nn as nn

from models import resnet20, resnet56, resnet110, mobilenetv2
from utils.data_loader import get_cifar_loaders, get_class_names
from utils.metrics import accuracy, count_parameters, estimate_flops, AverageMeter
from utils.visualization import (plot_cost_matrix, plot_training_curves,
                                  plot_compression_tradeoff, plot_cost_matrix_evolution)


_FACTORY={"resnet20":resnet20, "resnet56":resnet56,
          "resnet110":resnet110, "mobilenetv2":mobilenetv2}


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(arch, nc, ckpt, dev):
    m=_FACTORY[arch](num_classes=nc).to(dev)
    if not os.path.exists(ckpt):
        return m, None
    blob=torch.load(ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(blob["state_dict"])
    return m, blob


def _eval(m, loader, dev):
    m.eval()
    a=AverageMeter()
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(dev),y.to(dev)
            a.update(accuracy(m(x),y,topk=(1,))[0], x.size(0))
    return a.avg


def _print_table(rows):
    print("\n"+"="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    head=f"{'method':<25} | {'top-1':>10} | {'params':>10} | {'flops':>12}"
    print(head); print("-"*len(head))
    for r in rows:
        acc=f"{r['top1_acc']:.2f}%" if r['top1_acc'] else "N/A"
        pa=f"{r['params']/1e6:.2f}M"
        fl=f"{r['flops']/1e6:.1f}M" if r['flops'] else "N/A"
        print(f"{r['method']:<25} | {acc:>10} | {pa:>10} | {fl:>12}")
    print("="*80)


def collect_results(args):
    dev=_dev()
    nc=10 if args.dataset=="cifar10" else 100
    cdir=args.checkpoint_dir
    _,_,te=get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir,
                             batch_size=args.batch_size, num_workers=args.num_workers)
    rows=[]

    # Slot 1: teacher
    tp=os.path.join(cdir, f"{args.dataset}_{args.teacher}_teacher.pth")
    t, t_ck = _load_model(args.teacher, nc, tp, dev)
    t_acc = t_ck["best_acc"] if t_ck else None
    if t_acc is None and t_ck:
        t_acc=_eval(t, te, dev)
    rows.append({"method":f"Teacher ({args.teacher})", "top1_acc":t_acc,
                 "params":count_parameters(t), "flops":estimate_flops(t),
                 "params_M":count_parameters(t)/1e6})

    # Slot 2..5: students
    todo=[("Student (no KD)", os.path.join(cdir, f"{args.student}_no_kd_best.pth")),
          ("KL-KD",            os.path.join(cdir, "kl_kd_best.pth")),
          ("Fixed-OT-KD",      os.path.join(cdir, "sinkhorn_kd_best.pth")),
          ("Adaptive-OT-KD (Ours)", os.path.join(cdir, "adaptive_sinkhorn_kd_best.pth"))]
    last_ck=None
    for name, path in todo:
        m, ck = _load_model(args.student, nc, path, dev)
        rows.append({"method":name, "top1_acc": (ck["best_acc"] if ck else None),
                     "params":count_parameters(m), "flops":estimate_flops(m),
                     "params_M":count_parameters(m)/1e6})
        if name.startswith("Adaptive-OT-KD"):
            last_ck=ck
    return rows, last_ck


def make_figures(args, rows, adapt_ck):
    cdir=args.checkpoint_dir
    cn=get_class_names(args.dataset)
    figs=os.path.join(cdir, "figures")
    os.makedirs(figs, exist_ok=True)

    if adapt_ck and "cost_matrix" in adapt_ck:
        C=adapt_ck["cost_matrix"]
        if isinstance(C, torch.Tensor):
            C=C.numpy()
        plot_cost_matrix(C, cn,
                         save_path=os.path.join(figs, "learned_cost_matrix.png"),
                         title=f"Learned Cost Matrix ({args.dataset.upper()})")

    curves={}
    for key, label in [("kl_kd","KL-KD"), ("sinkhorn_kd","Fixed-OT-KD"),
                        ("adaptive_sinkhorn_kd","Adaptive-OT-KD")]:
        rp=os.path.join(cdir, f"{key}_results.pth")
        if os.path.exists(rp):
            d=torch.load(rp, map_location="cpu", weights_only=False)
            if "history" in d:
                curves[label]=d["history"]
    if curves:
        plot_training_curves(curves, save_path=os.path.join(figs, "training_curves.png"))

    pts=[{"method":r["method"], "params_M":r["params_M"], "top1_acc":r["top1_acc"]}
         for r in rows if r["top1_acc"] is not None]
    if pts:
        plot_compression_tradeoff(pts, save_path=os.path.join(figs, "compression_tradeoff.png"))

    snaps=sorted(glob.glob(os.path.join(cdir, "adaptive_sinkhorn_kd_epoch*.pth")))
    if snaps:
        Cs=[]; eps=[]
        for path in snaps:
            ck=torch.load(path, map_location="cpu", weights_only=False)
            if "cost_matrix" in ck:
                C=ck["cost_matrix"]
                if isinstance(C, torch.Tensor):
                    C=C.numpy()
                Cs.append(C); eps.append(ck["epoch"]+1)
        if Cs:
            # cap to 5 evenly-spaced snapshots so the figure stays readable
            if len(Cs)>5:
                idx=np.linspace(0, len(Cs)-1, 5, dtype=int)
                Cs=[Cs[i] for i in idx]; eps=[eps[i] for i in idx]
            plot_cost_matrix_evolution(Cs, eps, cn,
                                       save_path=os.path.join(figs, "cost_evolution.png"))
    print(f"\nfigures -> {figs}/")


def run_multi_seed(args):
    from train import train_distillation, train_student_baseline

    seeds=list(range(args.seed, args.seed+args.num_seeds))
    accum={k:[] for k in ["student_baseline","kl_kd","sinkhorn_kd","adaptive_sinkhorn_kd"]}
    for s in seeds:
        print(f"\n{'='*60}\nSEED {s}\n{'='*60}")
        args.seed=s
        args.mode="student_baseline"
        accum["student_baseline"].append(train_student_baseline(args))
        for m in ["kl_kd","sinkhorn_kd","adaptive_sinkhorn_kd"]:
            args.method=m
            _, a = train_distillation(args)
            accum[m].append(a)

    print("\n"+"="*60)
    print(f"MULTI-SEED RESULTS ({args.num_seeds} seeds)")
    print("="*60)
    print(f"{'method':<25} | {'mean':>10} | {'std':>8}")
    print("-"*50)
    for k,v in accum.items():
        print(f"{k:<25} | {np.mean(v):>9.2f}% | {np.std(v):>7.2f}%")


def parse_args():
    p=argparse.ArgumentParser(description="evaluate + compare KD methods")
    p.add_argument("--dataset", default="cifar100", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--checkpoint_dir", default="./checkpoints/cifar100")
    p.add_argument("--teacher", default="resnet110")
    p.add_argument("--student", default="resnet20")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--run_seeds", action="store_true")
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args=parse_args()
    print(f"device={_dev()}")
    if args.run_seeds:
        run_multi_seed(args)
    else:
        rows, adapt_ck = collect_results(args)
        _print_table(rows)
        make_figures(args, rows, adapt_ck)


if __name__=="__main__":
    main()
