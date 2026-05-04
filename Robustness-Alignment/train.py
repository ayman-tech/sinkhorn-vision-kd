"""Training entry point for Sinkhorn OT-KD.

modes: pretrain_teacher | distill | student_baseline
methods (when --mode distill): kl_kd | sinkhorn_kd | adaptive_sinkhorn_kd
"""
import argparse, os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from models import resnet20, resnet56, resnet110, mobilenetv2
from distillation import KLDistillationLoss, SinkhornDistillationLoss, AdaptiveSinkhornKD
from utils.data_loader import get_cifar_loaders, get_class_names
from utils.metrics import accuracy, count_parameters, estimate_flops, AverageMeter
from utils.visualization import plot_cost_matrix, plot_training_curves


_MODELS={"resnet20":resnet20,"resnet56":resnet56,"resnet110":resnet110,"mobilenetv2":mobilenetv2}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # benchmark=False matters more than determinism here — avoids cudnn picking
    # different algos across runs on the same machine.
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(arch, nc):
    if arch not in _MODELS:
        raise ValueError(f"unknown arch {arch!r}, want one of {list(_MODELS)}")
    if arch=="mobilenetv2":
        return _MODELS[arch](num_classes=nc, width_mult=1.0)
    return _MODELS[arch](num_classes=nc)


def _sgd(model, lr, mom, wd):
    return optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)


def _cosine_with_warmup(opt, epochs, warmup=5):
    def fn(e):
        if e<warmup:
            return (e+1)/warmup
        prog=(e-warmup)/max(1, epochs-warmup)
        return 0.5*(1.0+np.cos(np.pi*prog))
    return optim.lr_scheduler.LambdaLR(opt, fn)


def _save(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def evaluate_model(model, loader, device):
    model.eval()
    m=AverageMeter()
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            acc=accuracy(model(x), y)[0]
            m.update(acc, x.size(0))
    return m.avg


def pretrain_teacher(args):
    dev=_device(); set_seed(args.seed)
    nc=10 if args.dataset=="cifar10" else 100
    tr,_,te=get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir,
                              batch_size=args.batch_size, num_workers=args.num_workers)

    net=_build_model(args.teacher, nc).to(dev)
    opt=_sgd(net, lr=0.1, mom=0.9, wd=1e-4)
    sched=_cosine_with_warmup(opt, args.pretrain_epochs, warmup=5)
    crit=nn.CrossEntropyLoss()

    print(f"pretraining {args.teacher} on {args.dataset} | params={count_parameters(net):,}")
    print(f"{'epoch':>6} | {'loss':>10} | {'tr_acc':>7} | {'te_acc':>7} | {'lr':>8}")
    print("-"*55)

    best=0.0
    for ep in range(args.pretrain_epochs):
        net.train()
        loss_m=AverageMeter(); top1=AverageMeter()
        for x,y in tr:
            x,y=x.to(dev),y.to(dev)
            z=net(x); l=crit(z,y)
            opt.zero_grad(); l.backward(); opt.step()
            top1.update(accuracy(z,y)[0], x.size(0))
            loss_m.update(l.item(), x.size(0))
        sched.step()

        te_acc=evaluate_model(net, te, dev)
        cur_lr=opt.param_groups[0]["lr"]
        print(f"{ep+1:>6} | {loss_m.avg:>10.4f} | {top1.avg:>6.2f}% | {te_acc:>6.2f}% | {cur_lr:>8.5f}")

        if te_acc>best:
            best=te_acc
            cdir=args.checkpoint_dir or f"./checkpoints/{args.dataset}"
            _save({"arch":args.teacher, "state_dict":net.state_dict(),
                   "num_classes":nc, "best_acc":best, "epoch":ep},
                  os.path.join(cdir, f"{args.dataset}_{args.teacher}_teacher.pth"))
    print(f"\ndone. best test acc: {best:.2f}%")


def train_distillation(args):
    dev=_device(); set_seed(args.seed)
    nc=10 if args.dataset=="cifar10" else 100
    cdir=args.checkpoint_dir or f"./checkpoints/{args.dataset}"
    os.makedirs(cdir, exist_ok=True)

    needs_val=(args.method=="adaptive_sinkhorn_kd")
    vfrac=args.val_fraction if needs_val else 0.0
    tr, val, te = get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir,
                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                    val_fraction=vfrac, seed=args.seed)
    val_iter=iter(val) if val is not None else None

    teacher=_build_model(args.teacher, nc).to(dev)
    tpath=args.teacher_path or os.path.join(cdir, f"{args.dataset}_{args.teacher}_teacher.pth")
    if os.path.exists(tpath):
        ck=torch.load(tpath, map_location=dev, weights_only=False)
        teacher.load_state_dict(ck["state_dict"])
        print(f"loaded teacher: {tpath} (acc={ck.get('best_acc','?')}%)")
    else:
        print(f"WARNING: no teacher ckpt at {tpath} — running with random teacher (sanity only).")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad=False

    student=_build_model(args.student, nc).to(dev)
    print(f"\nmethod={args.method}  T={args.teacher}({count_parameters(teacher):,})  "
          f"S={args.student}({count_parameters(student):,})  flops_S={estimate_flops(student):,}")

    if args.method=="kl_kd":
        crit=KLDistillationLoss(temperature=args.temperature, alpha=args.alpha)
    elif args.method=="sinkhorn_kd":
        crit=SinkhornDistillationLoss(num_classes=nc, temperature=args.temperature,
                                      lambda_ot=args.lambda_ot, epsilon=args.epsilon,
                                      max_iter=args.sinkhorn_max_iter,
                                      threshold=args.sinkhorn_threshold,
                                      cost_type=args.cost_type).to(dev)
    elif args.method=="adaptive_sinkhorn_kd":
        crit=AdaptiveSinkhornKD(num_classes=nc, temperature=args.temperature,
                                lambda_ot=args.lambda_ot, epsilon=args.epsilon,
                                max_iter=args.sinkhorn_max_iter,
                                threshold=args.sinkhorn_threshold,
                                cost_lr=args.cost_lr, cost_update_freq=args.cost_update_freq,
                                cost_grad_clip=args.cost_grad_clip).to(dev)
    else:
        raise ValueError(f"unknown method {args.method!r}")

    opt=_sgd(student, args.lr, args.momentum, args.weight_decay)
    sched=_cosine_with_warmup(opt, args.epochs, args.warmup_epochs)

    hist={"train_acc":[], "val_acc":[], "train_loss":[],
          "ot_loss":[], "kd_loss":[], "ce_loss":[]}
    best=0.0

    head=f"{'epoch':>6} | {'loss':>8} | {'OT/KD':>8} | {'CE':>8} | {'tr':>7} | {'te':>7} | {'lr':>8}"
    print(f"\n{head}\n{'-'*len(head)}")

    for ep in range(args.epochs):
        student.train()
        L=AverageMeter(); ot=AverageMeter(); ce=AverageMeter(); top1=AverageMeter()

        bar=tqdm(tr, desc=f"ep {ep+1}/{args.epochs}", leave=False)
        for x,y in bar:
            x,y=x.to(dev),y.to(dev)

            # Bilevel outer step: refresh C using a held-out val batch.
            if args.method=="adaptive_sinkhorn_kd" and crit.should_update_cost():
                try:
                    vx,vy=next(val_iter)
                except (StopIteration, TypeError):
                    val_iter=iter(val)
                    vx,vy=next(val_iter)
                vx,vy=vx.to(dev),vy.to(dev)
                crit.step_cost_matrix(student, teacher, vx, vy)

            with torch.no_grad():
                tz=teacher(x)
            sz=student(x)
            r=crit(sz, tz, y)
            l=r["loss"]
            opt.zero_grad(); l.backward(); opt.step()

            if args.method=="adaptive_sinkhorn_kd":
                crit.increment_step()

            top1.update(accuracy(sz,y)[0], x.size(0))
            L.update(l.item(), x.size(0))
            ot_v=r.get("ot_loss", r.get("kd_loss", torch.tensor(0.0)))
            if isinstance(ot_v, torch.Tensor):
                ot_v=ot_v.item()
            ot.update(ot_v, x.size(0))
            ce.update(r["ce_loss"].item(), x.size(0))
            bar.set_postfix(loss=f"{L.avg:.4f}", acc=f"{top1.avg:.1f}%")

        sched.step()
        te_acc=evaluate_model(student, te, dev)
        cur_lr=opt.param_groups[0]["lr"]
        print(f"{ep+1:>6} | {L.avg:>8.4f} | {ot.avg:>8.4f} | {ce.avg:>8.4f} | "
              f"{top1.avg:>6.2f}% | {te_acc:>6.2f}% | {cur_lr:>8.5f}")

        hist["train_acc"].append(top1.avg)
        hist["val_acc"].append(te_acc)
        hist["train_loss"].append(L.avg)
        hist["ot_loss"].append(ot.avg)
        hist["ce_loss"].append(ce.avg)

        if te_acc>best:
            best=te_acc
            st={"method":args.method, "epoch":ep,
                "student_arch":args.student, "teacher_arch":args.teacher,
                "state_dict":student.state_dict(), "best_acc":best, "history":hist}
            if args.method=="adaptive_sinkhorn_kd":
                st["cost_matrix"]=crit.get_cost_matrix_numpy()
            _save(st, os.path.join(cdir, f"{args.method}_best.pth"))

        if (ep+1)%args.save_freq==0:
            st={"method":args.method, "epoch":ep,
                "student_arch":args.student, "state_dict":student.state_dict(),
                "optimizer":opt.state_dict(), "scheduler":sched.state_dict(),
                "best_acc":best, "history":hist}
            if args.method=="adaptive_sinkhorn_kd":
                st["cost_matrix"]=crit.get_cost_matrix_numpy()
            _save(st, os.path.join(cdir, f"{args.method}_epoch{ep+1}.pth"))

    print(f"\ndone. best test acc: {best:.2f}%")
    _save({"history":hist, "best_acc":best, "args":vars(args)},
          os.path.join(cdir, f"{args.method}_results.pth"))

    if args.method=="adaptive_sinkhorn_kd":
        C=crit.get_cost_matrix_numpy()
        plot_cost_matrix(C, get_class_names(args.dataset),
                         save_path=os.path.join(cdir, "learned_cost_matrix.png"),
                         title=f"Learned Cost Matrix ({args.dataset.upper()})")
    return hist, best


def train_student_baseline(args):
    dev=_device(); set_seed(args.seed)
    nc=10 if args.dataset=="cifar10" else 100
    cdir=args.checkpoint_dir or f"./checkpoints/{args.dataset}"

    tr,_,te=get_cifar_loaders(dataset=args.dataset, data_dir=args.data_dir,
                              batch_size=args.batch_size, num_workers=args.num_workers)
    s=_build_model(args.student, nc).to(dev)
    opt=_sgd(s, args.lr, args.momentum, args.weight_decay)
    sched=_cosine_with_warmup(opt, args.epochs, args.warmup_epochs)
    crit=nn.CrossEntropyLoss()

    print(f"baseline {args.student} (no KD) | params={count_parameters(s):,}")
    best=0.0
    for ep in range(args.epochs):
        s.train()
        L=AverageMeter(); top1=AverageMeter()
        for x,y in tr:
            x,y=x.to(dev),y.to(dev)
            z=s(x); l=crit(z,y)
            opt.zero_grad(); l.backward(); opt.step()
            top1.update(accuracy(z,y)[0], x.size(0))
            L.update(l.item(), x.size(0))
        sched.step()
        te_acc=evaluate_model(s, te, dev)
        if (ep+1)%10==0:
            print(f"ep {ep+1:>4} | loss={L.avg:.4f} | tr={top1.avg:.2f}% | te={te_acc:.2f}%")
        if te_acc>best:
            best=te_acc
            _save({"arch":args.student, "state_dict":s.state_dict(), "best_acc":best},
                  os.path.join(cdir, f"{args.student}_no_kd_best.pth"))
    print(f"baseline done. best: {best:.2f}%")
    return best


def parse_args():
    p=argparse.ArgumentParser(description="Sinkhorn OT-KD")
    p.add_argument("--mode", default="distill",
                   choices=["pretrain_teacher","distill","student_baseline"])
    p.add_argument("--method", default="adaptive_sinkhorn_kd",
                   choices=["kl_kd","sinkhorn_kd","adaptive_sinkhorn_kd"])
    p.add_argument("--teacher", default="resnet110", choices=["resnet56","resnet110"])
    p.add_argument("--student", default="resnet20",
                   choices=["resnet20","resnet56","mobilenetv2"])
    p.add_argument("--dataset", default="cifar100", choices=["cifar10","cifar100"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9, help="KL-KD weight")
    p.add_argument("--lambda_ot", type=float, default=0.5, help="OT loss weight")
    p.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn entropy reg")
    p.add_argument("--sinkhorn_max_iter", type=int, default=50)
    p.add_argument("--sinkhorn_threshold", type=float, default=1e-3)
    p.add_argument("--cost_type", default="uniform",
                   choices=["uniform","label_distance","random"])
    p.add_argument("--cost_lr", type=float, default=0.01)
    p.add_argument("--cost_update_freq", type=int, default=10)
    p.add_argument("--cost_grad_clip", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--teacher_path", default=None)
    p.add_argument("--save_freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None, help="YAML config (overrides CLI)")
    return p.parse_args()


def load_config(args):
    if args.config is None:
        return args
    with open(args.config, "r") as f:
        cfg=yaml.safe_load(f)
    flat={}
    for sec in cfg.values():
        if isinstance(sec, dict):
            flat.update(sec)
    for k,v in flat.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def main():
    args=load_config(parse_args())
    print("="*60)
    print("Sinkhorn OT-KD for vision")
    print("="*60)
    print(f"device={_device()}  seed={args.seed}\n")

    if args.mode=="pretrain_teacher":
        pretrain_teacher(args)
    elif args.mode=="student_baseline":
        train_student_baseline(args)
    else:
        train_distillation(args)


if __name__=="__main__":
    main()
