# fine_tune_xlmr_pseudo.py
# pip install: torch transformers pandas scikit-learn numpy
import os, json, random, argparse, warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding
try:
    from transformers import get_cosine_schedule_with_warmup as _get_sched
except Exception:
    from transformers import get_linear_schedule_with_warmup as _get_sched

from torch.amp import GradScaler, autocast

# -------------------- helpers --------------------
TEXT_CANDIDATES = ["text", "review", "content", "item_name", "title", "name", "description", "desc"]
LABEL_CANDIDATES = ["label", "category", "target", "class"]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    t = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
    y = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if not t or not y:
        raise ValueError(f"Не нашёл колонки текста/метки. Есть: {list(df.columns)}")
    return t, y

def load_categories(path: Optional[str]) -> Optional[List[str]]:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cats = [line.strip() for line in f if line.strip()]
        return cats
    return None

def normalize_text(s: str) -> str:
    return " ".join(str(s).split())

# -------------------- dataset --------------------
class ClsDS(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tok, max_len: int):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = int(self.labels[i])
        return enc

# -------------------- model utils --------------------
def freeze_bottom_layers(model, n_layers: int = 0):
    if n_layers <= 0: return
    encoder = model.roberta.encoder
    total = len(encoder.layer)
    n = min(n_layers, total)
    for i in range(n):
        for p in encoder.layer[i].parameters():
            p.requires_grad = False

def set_base_trainable(model, flag: bool, freeze_bottom: int = 0):
    # toggle all base params
    for p in model.roberta.parameters():
        p.requires_grad = flag
    # optionally keep bottom frozen
    if flag and freeze_bottom > 0:
        freeze_bottom_layers(model, freeze_bottom)

def init_classifier_bias_with_priors(model, counts: np.ndarray):
    priors = counts.astype(np.float32)
    priors = priors / max(priors.sum(), 1.0)
    priors = np.clip(priors, 1e-9, 1.0)
    b = torch.log(torch.tensor(priors))
    with torch.no_grad():
        head = model.classifier
        if hasattr(head, "out_proj") and hasattr(head.out_proj, "bias") and head.out_proj.bias is not None:
            head.out_proj.bias.copy_(b.to(head.out_proj.bias.device))

# ---- class weights helpers ----
def weights_inv_freq(counts: np.ndarray, alpha: float = 1.0, cap: float = 5.0):
    w = 1.0 / np.clip(counts, 1.0, None) ** alpha
    # normalize mean to 1
    avg = float((w * counts).sum() / max(counts.sum(), 1.0))
    w = w / max(avg, 1e-12)
    # cap extreme
    w = np.minimum(w, cap)
    return w

def weights_cb(counts: np.ndarray, beta: float = 0.99, cap: float = 5.0):
    eff = 1.0 - np.power(beta, counts.astype(np.float64))
    w = (1.0 - beta) / np.clip(eff, 1e-12, None)
    avg = float((w * counts).sum() / max(counts.sum(), 1.0))
    w = w / max(avg, 1e-12)
    w = np.minimum(w, cap)
    return w

def build_class_weights(mode: str, counts: np.ndarray, alpha: float, beta: float, cap: float):
    if mode == "none":
        return np.ones_like(counts, dtype=np.float64)
    if mode == "inv":
        return weights_inv_freq(counts, alpha=alpha, cap=cap)
    if mode == "inv_sqrt":
        return weights_inv_freq(counts, alpha=0.5, cap=cap)
    if mode == "cb":
        return weights_cb(counts, beta=beta, cap=cap)
    raise ValueError(f"Unknown class_weight_mode: {mode}")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="train_pseudo.csv")
    ap.add_argument("--categories_path", type=str, default="/mnt/data/categories.txt")
    ap.add_argument("--model_name", type=str, default="xlm-roberta-base")
    ap.add_argument("--output_dir", type=str, default="./outputs_xlmr_min")

    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5, help="LR for encoder (base)")
    ap.add_argument("--head_lr", type=float, default=5e-5, help="LR for classifier head")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--freeze_bottom_layers", type=int, default=8)
    ap.add_argument("--head_init_epochs", type=int, default=1, help="epochs to train only the head")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=4)

    ap.add_argument("--class_weight_mode", type=str, default="inv_sqrt", choices=["none","inv","inv_sqrt","cb"])
    ap.add_argument("--class_weight_beta", type=float, default=0.99, help="for mode=cb")
    ap.add_argument("--class_weight_cap", type=float, default=3.0)
    ap.add_argument("--use_sampler", action="store_true", help="WeightedRandomSampler by inverse sqrt freq")

    ap.add_argument("--print_pred_dist", action="store_true", help="print prediction distribution")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ---- load data
    df = pd.read_csv(args.input_csv)
    t_col, y_col = detect_columns(df)
    df = df[[t_col, y_col]].dropna().copy()
    df[t_col] = df[t_col].astype(str).map(normalize_text)
    df[y_col] = df[y_col].astype(str).map(str.strip)

    # labels mapping
    cats = load_categories(args.categories_path)
    labels_sorted = cats if cats else sorted(df[y_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    # extend mapping if unseen labels
    miss = sorted(set(df[y_col]) - set(label2id.keys()))
    for lab in miss:
        idx = len(label2id)
        label2id[lab] = idx; id2label[idx] = lab
    num_labels = len(label2id)

    df["label_id"] = df[y_col].map(label2id)

    # split (stratify only if every class has >=2 samples)
    counts_global = df["label_id"].value_counts()
    can_stratify = (counts_global.min() >= 2)
    if not can_stratify:
        warnings.warn("Некоторые классы имеют <2 объектов — stratify отключён (простой holdout).")

    tr, va = train_test_split(
        df, test_size=args.val_size, random_state=args.seed,
        stratify=df["label_id"] if can_stratify else None
    )
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)

    print("\n[train] class distribution:")
    print(tr["label_id"].value_counts().sort_index())
    print("\n[valid] class distribution:")
    print(va["label_id"].value_counts().sort_index())

    # ---- tokenizer, datasets, loaders
    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ClsDS(tr[t_col].tolist(), tr["label_id"].tolist(), tok, args.max_length)
    valid_ds = ClsDS(va[t_col].tolist(), va["label_id"].tolist(), tok, args.max_length)
    collator = DataCollatorWithPadding(tok)

    # sampler (optional)
    train_sampler = None
    if args.use_sampler:
        counts_train = tr["label_id"].value_counts().reindex(range(num_labels), fill_value=0).values.astype(float)
        inv_sqrt = 1.0 / np.sqrt(np.clip(counts_train, 1.0, None))
        sample_w_per_class = inv_sqrt / inv_sqrt.mean()
        sample_w = [sample_w_per_class[y] for y in tr["label_id"].tolist()]
        train_sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.train_bs,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=2, collate_fn=collator)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_bs, shuffle=False, num_workers=2, collate_fn=collator)

    # ---- model
    cfg = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=cfg)

    # init classifier bias to priors (train)
    cls_counts_train = tr["label_id"].value_counts().reindex(range(num_labels), fill_value=0).values
    init_classifier_bias_with_priors(model, cls_counts_train)

    # start: train only head
    set_base_trainable(model, False)
    # later we'll unfreeze with freeze_bottom_layers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- class weights
    cw = build_class_weights(args.class_weight_mode, cls_counts_train.astype(float),
                             alpha=1.0, beta=args.class_weight_beta, cap=args.class_weight_cap)
    print(f"Class weights ({args.class_weight_mode}): min={cw.min():.3f} max={cw.max():.3f}")
    print("Weights by class id:", {i: float(w) for i, w in enumerate(cw)})
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # ---- optimizer with separate LRs for base/head
    decay_terms = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    base_decay, base_nodecay, head_decay, head_nodecay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith("classifier.")
        is_decay = (not any(nd in n for nd in decay_terms))
        if is_head:
            (head_decay if is_decay else head_nodecay).append(p)
        else:
            (base_decay if is_decay else base_nodecay).append(p)

    param_groups = []
    if base_decay:    param_groups.append({"params": base_decay, "lr": args.lr, "weight_decay": args.weight_decay})
    if base_nodecay:  param_groups.append({"params": base_nodecay, "lr": args.lr, "weight_decay": 0.0})
    if head_decay:    param_groups.append({"params": head_decay, "lr": args.head_lr, "weight_decay": args.weight_decay})
    if head_nodecay:  param_groups.append({"params": head_nodecay, "lr": args.head_lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = _get_sched(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    # ---- training loop
    best_f1 = -1.0
    patience_left = args.patience
    all_label_ids = list(range(num_labels))
    all_target_names = [id2label[i] for i in all_label_ids]

    for epoch in range(1, args.epochs + 1):
        # unfreeze base after head_init_epochs
        if epoch == args.head_init_epochs + 1:
            set_base_trainable(model, True, freeze_bottom=args.freeze_bottom_layers)
            # need to rebuild optimizer with base params now trainable
            decay_terms = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
            base_decay, base_nodecay, head_decay, head_nodecay = [], [], [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                is_head = n.startswith("classifier.")
                is_decay = (not any(nd in n for nd in decay_terms))
                if is_head:
                    (head_decay if is_decay else head_nodecay).append(p)
                else:
                    (base_decay if is_decay else base_nodecay).append(p)
            param_groups = []
            if base_decay:    param_groups.append({"params": base_decay, "lr": args.lr, "weight_decay": args.weight_decay})
            if base_nodecay:  param_groups.append({"params": base_nodecay, "lr": args.lr, "weight_decay": 0.0})
            if head_decay:    param_groups.append({"params": head_decay, "lr": args.head_lr, "weight_decay": args.weight_decay})
            if head_nodecay:  param_groups.append({"params": head_nodecay, "lr": args.head_lr, "weight_decay": 0.0})
            optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            scheduler = _get_sched(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

        model.train()
        running = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda"):
                    out = model(**batch); logits = out.logits
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                out = model(**batch); logits = out.logits
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            running += float(loss.item())

        # ---- eval
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                if use_amp:
                    with autocast("cuda"):
                        logits = model(**batch).logits
                else:
                    logits = model(**batch).logits
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_true.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
        all_true  = np.concatenate(all_true)  if all_true  else np.array([], dtype=int)

        # метрики
        wf1 = f1_score(all_true, all_preds, average="weighted") if all_true.size else 0.0
        acc = accuracy_score(all_true, all_preds) if all_true.size else 0.0

        # печать распределения предсказаний (по запросу)
        if args.print_pred_dist and all_preds.size:
            unique, cnts = np.unique(all_preds, return_counts=True)
            pred_dist = {int(k): int(v) for k, v in zip(unique, cnts)}
            print(f"[Epoch {epoch}] pred_dist={pred_dist}")

        print(f"[Epoch {epoch}] train_loss={running/max(1,len(train_loader)):.4f}  val_weighted_f1={wf1:.4f}  val_acc={acc:.4f}")

        # early stopping & save best
        if wf1 > best_f1:
            best_f1 = wf1
            patience_left = args.patience

            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)

            rep = classification_report(
                all_true, all_preds,
                labels=list(range(num_labels)),
                target_names=[id2label[i] for i in range(num_labels)],
                digits=4,
                zero_division=0
            )
            with open(os.path.join(save_dir, "val_report.txt"), "w", encoding="utf-8") as f:
                f.write(rep)
            with open(os.path.join(save_dir, "label_maps.json"), "w", encoding="utf-8") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # итог
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_weighted_f1": float(best_f1), "epochs_run": epoch}, f, indent=2, ensure_ascii=False)

    # сохранить распределения классов
    tr_dist = tr["label_id"].value_counts().sort_index().to_dict()
    va_dist = va["label_id"].value_counts().sort_index().to_dict()
    with open(os.path.join(args.output_dir, "splits_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_class_counts": tr_dist,
            "valid_class_counts": va_dist,
            "num_labels": num_labels,
            "id2label": id2label
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
