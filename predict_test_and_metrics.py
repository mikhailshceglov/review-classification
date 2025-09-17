# predict_test_and_metrics.py
# pip install: torch transformers pandas scikit-learn numpy
import os, json, argparse, time
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

TEXT_CANDS = ["text","review","content","title","description","name","item_name","desc"]
LABEL_CANDS = ["label","category","target","class"]
ID_CANDS    = ["id","Id","ID","row_id","index"]

def detect_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def load_label_maps(model_dir: str, categories_path: Optional[str] = None):
    """Возвращает (label2id, id2label).
    Порядок приоритетов:
      1) model_dir/label_maps.json
      2) config.json -> id2label/label2id
      3) categories.txt (если передан путь)
    """
    # 1) label_maps.json
    lm_path = os.path.join(model_dir, "label_maps.json")
    if os.path.isfile(lm_path):
        with open(lm_path, "r", encoding="utf-8") as f:
            maps = json.load(f)
        id2label = {int(k): v for k, v in maps["id2label"].items()}
        label2id = {k: int(v) for k, v in maps["label2id"].items()}
        return label2id, id2label

    # 2) config.json
    try:
        cfg = AutoConfig.from_pretrained(model_dir)
        if hasattr(cfg, "id2label") and cfg.id2label:
            id2label = {int(k): str(v) for k, v in cfg.id2label.items()}
            label2id = {v: k for k, v in id2label.items()}
            return label2id, id2label
    except Exception:
        pass

    # 3) categories.txt
    if categories_path and os.path.isfile(categories_path):
        cats = [line.strip() for line in open(categories_path, "r", encoding="utf-8") if line.strip()]
        id2label = {i: lab for i, lab in enumerate(cats)}
        label2id = {lab: i for i, lab in id2label.items()}
        return label2id, id2label

    raise FileNotFoundError(
        "Не нашёл карты меток: ни label_maps.json, ни id2label в config.json, ни categories_path. "
        "Либо укажи правильный --model_dir, либо передай --categories_path."
    )

@torch.no_grad()
def predict_df(model_dir: str, df: pd.DataFrame, text_col: str,
               max_length: int = 256, batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    texts = df[text_col].fillna("").astype(str).tolist()
    preds, probs_list = [], []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(probs.argmax(axis=1).tolist())
        probs_list.append(probs)
    avg = (time.time() - t0) / max(len(texts), 1)
    probs = np.vstack(probs_list) if probs_list else np.zeros((0, model.config.num_labels), dtype=np.float32)
    return np.array(preds, dtype=int), probs, avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Путь к директории best чекпойнта (или к самой модели)")
    ap.add_argument("--input_csv", required=True, help="CSV для инференса (test.csv или валидация)")
    ap.add_argument("--output_csv", default="./submission.csv", help="Куда сохранить предсказания")
    ap.add_argument("--text_col", default=None, help="Имя текстовой колонки (если авто-детект не подходит)")
    ap.add_argument("--label_col", default=None, help="Имя колонки истинных меток (если есть)")
    ap.add_argument("--id_col",    default=None, help="ID колонка (если есть)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--save_probs", action="store_true", help="Сохранить вероятности по классам (proba_<label>)")
    ap.add_argument("--categories_path", default=None, help="Путь к categories.txt как запасной вариант карт меток")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    df = pd.read_csv(args.input_csv)
    text_col = args.text_col or detect_col(df, TEXT_CANDS)
    if text_col is None:
        raise ValueError(f"Не удалось найти текстовую колонку. Есть: {list(df.columns)}")
    id_col = args.id_col or detect_col(df, ID_CANDS)
    label_col = args.label_col or detect_col(df, LABEL_CANDS)
    print(f"[cols] text={text_col}  id={id_col}  label={label_col}")

    label2id, id2label = load_label_maps(args.model_dir, args.categories_path)
    num_labels = len(id2label)
    print(f"[labels] num_labels={num_labels} | {id2label}")

    pred_ids, probs, avg = predict_df(args.model_dir, df, text_col, args.max_length, args.batch_size)
    pred_labels = [id2label[int(i)] for i in pred_ids]
    print(f"[speed] avg_sec_per_example = {avg:.4f}")

    out = pd.DataFrame()
    if id_col: out[id_col] = df[id_col]
    out[text_col] = df[text_col]
    out["label"] = pred_labels

    if args.save_probs:
        for j in range(probs.shape[1]):
            out[f"proba_{id2label[j]}"] = probs[:, j]

    out.to_csv(args.output_csv, index=False)
    print(f"[saved] predictions → {args.output_csv}")

    # Метрики, если есть истина
    if label_col is not None:
        y_true_names = df[label_col].astype(str).str.strip().tolist()
        mask = [y in label2id for y in y_true_names]
        n_ok = sum(mask); n_total = len(y_true_names)
        if n_ok == 0:
            print("[metrics] Ни одна истинная метка не совпала с картой — метрики не посчитаны.")
            return
        y_true = np.array([label2id[y] for y, m in zip(y_true_names, mask) if m], dtype=int)
        y_pred = np.array([pid for pid, m in zip(pred_ids, mask) if m], dtype=int)

        acc  = accuracy_score(y_true, y_pred)
        wf1  = f1_score(y_true, y_pred, average="weighted")
        print(f"[metrics] accuracy={acc:.4f}  weighted_f1={wf1:.4f}  (использовано {n_ok}/{n_total} строк)")

        rep = classification_report(
            y_true, y_pred,
            labels=list(range(num_labels)),
            target_names=[id2label[i] for i in range(num_labels)],
            digits=4, zero_division=0
        )
        with open(os.path.splitext(args.output_csv)[0] + "_report.txt", "w", encoding="utf-8") as f:
            f.write(rep)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
        cm_df = pd.DataFrame(cm, index=[f"true_{id2label[i]}" for i in range(num_labels)],
                                columns=[f"pred_{id2label[i]}" for i in range(num_labels)])
        cm_df.to_csv(os.path.splitext(args.output_csv)[0] + "_confusion_matrix.csv")

        with open(os.path.splitext(args.output_csv)[0] + "_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"accuracy": float(acc), "weighted_f1": float(wf1),
                       "used_rows": int(n_ok), "total_rows": int(n_total)}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
