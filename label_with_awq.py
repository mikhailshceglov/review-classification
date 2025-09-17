# файл: label_with_awq.py
import argparse, json, re, time, os, math
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer
try:
    from awq import AutoAWQForCausalLM
except Exception:
    from autoawq.modeling.autoawq import AutoAWQForCausalLM  # из пакета autoawq

JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

def read_categories(path:str) -> List[str]:
    cats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                cats.append(s)
    # Убираем дубли, сохраняем порядок
    seen, uniq = set(), []
    for c in cats:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def build_messages(review: str, categories: List[str]) -> List[Dict[str,str]]:
    cats = ", ".join(categories)
    system = (
        "Ты коротко и строго классифицируешь отзывы покупателей маркетплейса.\n"
        "Всегда выбирай ровно ОДНУ категорию из списка и отвечай ТОЛЬКО JSON.\n"
        "Если отзыв не про товар/пустой/непонятный — используй категорию «нет товара»."
    )
    user = (
        f"Категории: [{cats}]\n\n"
        "Формат ответа строго:\n"
        '{"label": "<одна категория>"}\n\n'
        f"Отзыв:\n{review}\n\n"
        "Верни только JSON без комментариев, без пояснений."
    )
    return [{"role":"system","content":system}, {"role":"user","content":user}]

def extract_label(text:str, categories:List[str]) -> str:
    """
    Достаём первый JSON-блок и валидируем поле label.
    При сбое возвращаем 'нет товара', если такая категория есть.
    """
    match = JSON_RE.search(text)
    fallback = "нет товара" if "нет товара" in categories else categories[-1]
    if not match:
        return fallback
    try:
        obj = json.loads(match.group(0))
        label = str(obj.get("label","")).strip()
        # Жёсткая валидация по множеству категорий
        return label if label in categories else fallback
    except Exception:
        return fallback

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TheBloke/Qwen2.5-7B-Instruct-AWQ",
                    help="HF ID квантованной модели AWQ")
    ap.add_argument("--cats", required=True, help="Путь до categories.txt")
    ap.add_argument("--input", required=True, help="CSV с отзывами")
    ap.add_argument("--text-col", default="text", help="Имя текстовой колонки")
    ap.add_argument("--id-col", default=None, help="(опц.) ID-колонка")
    ap.add_argument("--out", required=True, help="Куда сохранить разметку (CSV)")
    ap.add_argument("--batch", type=int, default=8, help="Размер батча (8 ок для 8 ГБ VRAM)")
    ap.add_argument("--max-new", type=int, default=48, help="max_new_tokens")
    ap.add_argument("--max-inp", type=int, default=768, help="макс. длина входа (токенов)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    categories = read_categories(args.cats)
    print(f"[info] categories: {categories}")

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoAWQForCausalLM.from_quantized(
        args.model,
        trust_remote_code=True,
        safetensors=True,
        device="cuda:0",        # жёстко CUDA
        use_ipex=False,         # не лезем в IPEX
        use_flash_attn=False,   # выключить FlashAttention
        fuse_layers=False,      # выключить fused-ядра AWQ (тоже могут звать FA)
        attn_implementation="sdpa"  # стандартное SDPA из PyTorch
    )
    model.eval()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise ValueError(f"Колонка '{args.text_col}' не найдена в {args.input}")
    texts = df[args.text_col].fillna("").astype(str).tolist()
    ids = df[args.id_col].tolist() if (args.id_col and args.id_col in df.columns) else list(range(len(texts)))

    total = len(texts)
    labels = []
    t0_all = time.time()


    for batch_idx, batch_texts in enumerate(chunked(texts, args.batch), start=1):
        messages = [build_messages(t, categories) for t in batch_texts]
        prompts = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_inp
        ).to(device)

        t0 = time.time()
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new,
            do_sample=False,            # детерминированно
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]  # только сгенерированная часть
        decoded = tok.batch_decode(gen, skip_special_tokens=True)

        batch_labels = [extract_label(txt, categories) for txt in decoded]
        labels.extend(batch_labels)

        dt = time.time() - t0
        done = min(batch_idx*args.batch, total)
        avg = (time.time() - t0_all) / done
        print(f"[batch {batch_idx}] {len(batch_texts)} примеров, {dt:.2f}s; "
              f"готово {done}/{total} | ср. {avg:.3f}s/пример")

    out_df = pd.DataFrame({"id": ids, args.text_col: texts, "label": labels})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    avg_total = (time.time() - t0_all) / max(total,1)
    print(f"[done] сохранено в: {args.out} | среднее {avg_total:.3f}s/пример")

if __name__ == "__main__":
    main()
