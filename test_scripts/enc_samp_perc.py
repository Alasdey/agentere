#!/usr/bin/env python3
"""
encoder_learning_curve_continuous.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Experiment: Train on x% of data to demonstrate data efficiency.
Features:
 - Continuous logging (JSONL) to prevent data loss on crash.
 - Configurable Early Stopping (Patience).
 - dataset shuffling for subsets.
"""

import os
import json
import math
import random
import datetime
import argparse
import copy
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerModel, LongformerTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    DATASET_NAME = 'MAVEN-ERE-Causal-Events'
    DATASET = f'Nofing/{DATASET_NAME}'
    DATASET_TRAIN_SPLIT = "train"
    DATASET_DEV_SPLIT = "dev"  # Needed for rollback
    DATASET_TEST_SPLIT = "test"
    
    KEEP_RELATIONS: list = ['PRECONDITION', 'CAUSE']
    ANNOTATED_PAIRS_ONLY = False
    EXCLUDED_RELS = ["NoRel"]

    # Experiment: Learning Curve
    # Default percentages if not provided via CLI
    TRAIN_PERCENTAGES = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

    # Tokenizer / Encoder
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_SEQ_LENGTH = 4096
    ENCODER_TRAINED_LAYERS = [-1, -2, -3, -4, -5, -6]
    GLOBAL_MENTION = True

    # Training
    BATCH_SIZE = 12
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1  # 10% of steps used for warmup
    BATCH_SHUFFLE = True
    
    # Validation / Patience
    EARLY_STOPPING_PATIENCE = 5  # "5 generation rollback"

    # Aggregation
    AGGREG = "mean"

    # ASL Loss
    GAMMA_NEG = 4
    GAMMA_POS = 1
    CLIP = 0.05
    EPS = 1e-8

    # Eval
    THRESH_FLOOR = 3
    THRESH_STEPS = 100

    # Logging
    TIME_START = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    LOG_DIR = f"./logs/{DATASET_NAME}_{TIME_START}/"
    LOG_FILE = "results_curve.jsonl"

    # Derived
    LABEL_LIST: list = None
    NUM_LABELS: int = None
    REL_TYPE_IDX: list = None


# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOGGING UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def append_log_to_jsonl(filepath, data_dict):
    """Appends a dictionary as a single JSON line to a file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class HFSpanDataset:
    def __init__(self, hf_path: str, keep_relations: list = None):
        self.ds = load_dataset(hf_path)
        self.modes = list(self.ds.keys())

        # Discover relation types
        first_split = self.ds[self.modes[0]]
        all_rel_types = set()
        for row in first_split:
            all_rel_types.update(row["relations"].keys())

        if keep_relations:
            self.ere_types = sorted([r for r in all_rel_types if r in keep_relations])
        else:
            self.ere_types = sorted(all_rel_types)

        self.split_data = None

    def set_dataset_split(self, mode: str):
        if mode not in self.ds:
            return False
        self.split_data = self.ds[mode]
        return True

    def subset_current_split(self, percentage):
        """
        Subsets the loaded split_data by selecting a random percentage.
        Uses a fixed seed to ensure specific percentages are comparable across runs.
        """
        if percentage >= 1.0:
            return # Use full dataset
        
        full_len = len(self.split_data)
        all_indices = list(range(full_len))
        
        # Deterministic shuffle
        rng = random.Random(Config.RANDOM_SEED)
        rng.shuffle(all_indices)
        
        cutoff = int(full_len * percentage)
        # Ensure at least one sample if percentage > 0
        if cutoff == 0 and percentage > 0: cutoff = 1

        selected_indices = all_indices[:cutoff]
        
        print(f"   >>> Subsetting {full_len} -> {len(selected_indices)} docs ({percentage*100:.1f}%)")
        self.split_data = self.split_data.select(selected_indices)

    def mention_info(self) -> list:
        result = []
        for sample in self.split_data:
            tokens = sample["tokens"]
            doc_info = []
            for mid, span in zip(sample["mentions"], sample["spans"]):
                text = " ".join(tokens[i] for i in sorted(span) if 0 <= i < len(tokens))
                doc_info.append((mid, text, frozenset(span)))
            result.append(doc_info)
        return result

    def _sample_to_wsa(self, sample, annotated_only):
        n_labels = len(self.ere_types)
        fsets = [frozenset(sp) for sp in sample["spans"]]
        wsa = {}
        if not annotated_only:
            for i, fs1 in enumerate(fsets):
                for j, fs2 in enumerate(fsets):
                    if i != j:
                        wsa[(fs1, fs2)] = [0] * n_labels
        for rel_type, pairs in sample["relations"].items():
            if rel_type not in self.ere_types: continue
            ridx = self.ere_types.index(rel_type)
            for src, tgt in pairs:
                key = (fsets[src], fsets[tgt])
                if key not in wsa: wsa[key] = [0] * n_labels
                wsa[key][ridx] = 1
        return wsa

    def word_list(self):
        return [sample["tokens"] for sample in self.split_data]

    def word_set_annotation(self, annotated_pairs_only=False):
        return [self._sample_to_wsa(s, annotated_pairs_only) for s in self.split_data]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class LongformerPairClassifier(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained(config.MODEL_NAME)
        for p in self.encoder.parameters(): p.requires_grad = False
        for i in config.ENCODER_TRAINED_LAYERS:
            for p in self.encoder.encoder.layer[i].parameters(): p.requires_grad = True

        h = self.encoder.config.hidden_size
        self.ffnn = nn.Sequential(
            nn.Linear(2 * h, 3 * h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(3 * h, 2 * h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, num_labels),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask, pair_indices, doc_indices, pair_labels):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        hidden = outputs.last_hidden_state
        pair_embeddings, all_indices, all_labels = [], [], []
        for b in range(input_ids.size(0)):
            h_doc = hidden[b]
            for k, (i, j) in enumerate(pair_indices[b]):
                pair_embeddings.append(torch.cat([h_doc[i], h_doc[j]], dim=-1))
                all_indices.append((doc_indices[b], b, i, j))
                all_labels.append(pair_labels[b][k])
        
        if not pair_embeddings: return None, None, None

        pair_embeddings = torch.stack(pair_embeddings)
        all_labels = torch.tensor(all_labels, device=pair_embeddings.device)
        return self.ffnn(pair_embeddings), all_indices, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents
    def __len__(self):
        return len(self.documents["tokens"]["input_ids"])
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.documents["tokens"].items()}
        item["pair_indices"] = self.documents["pair_indices"][idx]
        item["pair_labels"] = self.documents["pair_labels"][idx]
        item["doc_indices"] = idx
        return item

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "global_attention_mask": torch.stack([b["global_attention_mask"] for b in batch]) if "global_attention_mask" in batch[0] else None,
        "pair_indices": [b["pair_indices"] for b in batch],
        "pair_labels": [b["pair_labels"] for b in batch],
        "doc_indices": [b["doc_indices"] for b in batch],
    }

def _word_to_token(word_lists, token_offsets):
    result = []
    for words, offsets in zip(word_lists, token_offsets):
        word_offset, char_pos = [], 0
        for w in words:
            word_offset.append([char_pos, char_pos + len(w)])
            char_pos += len(w) + 1
        doc_map = []
        t_id = 0
        for w_off in word_offset:
            indices = set()
            while t_id < len(offsets) and offsets[t_id][0] < w_off[1]:
                if w_off[0] < offsets[t_id][1]: indices.add(t_id)
                t_id += 1
            doc_map.append(indices)
        result.append(doc_map)
    return result

def _set_word_to_tok(word_set, w2t):
    return frozenset(t for w_idx in word_set for t in w2t[w_idx])

def _tok_clust_pair_rel(annots, w2t):
    result = []
    for doc_idx, doc_annot in enumerate(annots):
        doc = {}
        for clust_pair, gold in doc_annot.items():
            key = (_set_word_to_tok(clust_pair[0], w2t[doc_idx]), _set_word_to_tok(clust_pair[1], w2t[doc_idx]))
            doc[key] = gold
        result.append(doc)
    return result

def _tok_pair_annot(tok_set_annot):
    pair_indices, pair_labels = [], []
    for doc_annot in tok_set_annot:
        pi, pl = [], []
        for pair, gold in doc_annot.items():
            for i in pair[0]:
                for j in pair[1]:
                    pi.append((i, j))
                    pl.append(gold)
        pair_indices.append(pi)
        pair_labels.append(pl)
    return pair_indices, pair_labels

def _create_global_attention_mask(pair_indices_list, attention_mask):
    gam = torch.zeros_like(attention_mask, dtype=torch.long)
    for doc_idx, doc_pairs in enumerate(pair_indices_list):
        mention_tokens = set(t for i, j in doc_pairs for t in (i, j))
        gam[doc_idx, 0], gam[doc_idx, -1] = 1, 1
        for t in mention_tokens:
            if 0 <= t < gam.size(1): gam[doc_idx, t] = 1
    return gam

def prepare_data(hf_dataset: HFSpanDataset, config: Config):
    wsa = hf_dataset.word_set_annotation(config.ANNOTATED_PAIRS_ONLY)
    word_lists = hf_dataset.word_list()
    mention_info = hf_dataset.mention_info()
    tokenizer = LongformerTokenizerFast.from_pretrained(config.MODEL_NAME, add_prefix_space=True)
    tokens = tokenizer([" ".join(wl) for wl in word_lists], is_split_into_words=False, return_offsets_mapping=True, padding="longest", truncation=True, max_length=config.MAX_SEQ_LENGTH, return_tensors="pt")
    w2t = _word_to_token(word_lists, tokens["offset_mapping"])
    tok_set_annot = _tok_clust_pair_rel(wsa, w2t)
    pair_indices, pair_labels = _tok_pair_annot(tok_set_annot)
    if config.GLOBAL_MENTION:
        tokens["global_attention_mask"] = _create_global_attention_mask(pair_indices, tokens["attention_mask"])
    return PairDataset({"tokens": tokens, "pair_indices": pair_indices, "pair_labels": pair_labels, "doc_indices": list(range(len(pair_indices)))}), tok_set_annot

# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_rel(tok_set_annot, indices, logits, tlabels, config):
    if logits is None: return None, None, None
    doc_pair_map = defaultdict(dict)
    for idx, (doc_idx, _batch, ti, tj) in enumerate(indices):
        doc_pair_map[doc_idx][(ti, tj)] = idx

    preds, golds, inds = [], [], []
    for doc_idx, pred_pairs in doc_pair_map.items():
        for set_pair, label_list in tok_set_annot[doc_idx].items():
            pair_logits = []
            for i in set_pair[0]:
                for j in set_pair[1]:
                    if (i, j) in pred_pairs: pair_logits.append(logits[pred_pairs[(i, j)]])
            
            if not pair_logits: continue
            stacked = torch.stack(pair_logits)
            p = stacked.mean(dim=0) if config.AGGREG == "mean" else (stacked.max(dim=0)[0] if config.AGGREG == "max" else torch.logsumexp(stacked, dim=0))
            preds.append(p)
            golds.append(torch.tensor(label_list))
            inds.append((doc_idx, set_pair))
    if not preds: return None, None, None
    return torch.stack(preds), torch.stack(golds), inds

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip > 0: xs_neg = (xs_neg + self.clip).clamp(max=1)
        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = xs_pos * y + xs_neg * (1 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                loss *= (1 - pt).pow(gamma)
        return -loss

def find_best_threshold(preds_np, golds_np, eval_indices, floor=3, steps=100):
    best_score, best_thresh = 0.0, 0.5
    bot, alpha = math.exp(-floor), abs(1 - math.exp(-floor)) / steps
    f1_avg = "binary" if preds_np.shape[1] == 1 else "micro"
    for step in range(steps):
        t = bot + alpha * step
        bp = (preds_np[:, eval_indices] > t).astype(int)
        score = f1_score(golds_np[:, eval_indices], bp, average=f1_avg, zero_division=0.0)
        if score >= best_score: best_score, best_thresh = score, t
    return best_score, best_thresh

# ═══════════════════════════════════════════════════════════════════════════════
# UPDATE: TRAIN EPOCH WITH GRADIENT ACCUMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, tok_set_annot, optimizer, scheduler, loss_fn, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"].to(config.DEVICE) if batch["global_attention_mask"] is not None else None
        
        logits, indices, tlabels = model(input_ids, attn, gam, batch["pair_indices"], batch["doc_indices"], batch["pair_labels"])
        
        if logits is None: 
            # If batch is empty/skipped, we still need to manage steps? 
            # Ideally avoid empty batches, but here just continue.
            continue

        agg_logits, golds, _ = aggregate_rel(tok_set_annot, indices, logits.cpu(), tlabels.cpu(), config)
        if agg_logits is None: continue

        loss = loss_fn(agg_logits.to(config.DEVICE), golds.float().to(config.DEVICE)).mean()
        
        # Divide loss by gradient accumulation steps
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle remaining gradients if last batch wasn't divisible
    if (len(dataloader) % config.GRADIENT_ACCUMULATION_STEPS) != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(len(dataloader), 1)

@torch.no_grad()
def evaluate(model, dataloader, tok_set_annot, config):
    model.eval()
    all_preds, all_golds, all_inds = [], [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"].to(config.DEVICE) if batch["global_attention_mask"] is not None else None
        
        logits, indices, tlabels = model(input_ids, attn, gam, batch["pair_indices"], batch["doc_indices"], 
                                         [[[0]*config.NUM_LABELS]*len(p) for p in batch["pair_indices"]])
        
        agg_logits, golds, inds = aggregate_rel(tok_set_annot, indices, logits.cpu(), tlabels.cpu(), config)
        if agg_logits is not None:
            all_preds.append(torch.sigmoid(agg_logits).numpy())
            all_golds.append(golds.numpy())
            all_inds.extend(inds)

    if not all_preds: return np.array([]), np.array([]), []
    return np.concatenate(all_preds), np.concatenate(all_golds), all_inds

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_fraction(percentage, config, dev_loader, test_loader, dev_tsa, test_tsa, hf_ds_original):
    print(f"\n██████ STARTING RUN: {percentage*100:.0f}% Training Data ██████")
    
    # 1. Reset Model
    model = LongformerPairClassifier(config.NUM_LABELS, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    loss_fn = AsymmetricLoss(gamma_neg=config.GAMMA_NEG, gamma_pos=config.GAMMA_POS, clip=config.CLIP)

    # 2. Subset Training Data (Use copy to avoid filtering original)
    hf_subset = copy.deepcopy(hf_ds_original)
    if not hf_subset.set_dataset_split(config.DATASET_TRAIN_SPLIT):
        raise ValueError(f"Train split {config.DATASET_TRAIN_SPLIT} not found.")
    
    # Apply subsampling
    hf_subset.subset_current_split(percentage)
    
    sub_ds, sub_tsa = prepare_data(hf_subset, config)
    sub_loader = DataLoader(sub_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    
    if len(sub_ds) == 0:
        print("   > Skipping: 0 samples.")
        return None

    # 3. Optimizer & Scheduler (New: Weight Decay + Linear Warmup)
    total_steps = math.ceil(len(sub_loader) / config.GRADIENT_ACCUMULATION_STEPS) * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=config.LEARNING_RATE, 
                                  weight_decay=config.WEIGHT_DECAY)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps)

    loss_fn = AsymmetricLoss(gamma_neg=config.GAMMA_NEG, gamma_pos=config.GAMMA_POS, clip=config.CLIP)

    # 4. Training Loop
    best_dev_f1 = 0.0
    best_thresh = 0.5
    best_state = None
    patience_counter = 0

    print(f"   > Total Steps: {total_steps}, Warmup: {warmup_steps}")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, sub_loader, sub_tsa, optimizer, scheduler, loss_fn, config)
        
        # Validation
        dev_prob, dev_gold, _ = evaluate(model, dev_loader, dev_tsa, config)
        if len(dev_prob) == 0: continue
            
        f1, thresh = find_best_threshold(dev_prob, dev_gold, config.REL_TYPE_IDX, floor=3)

        if f1 > best_dev_f1:
            best_dev_f1 = f1
            best_thresh = thresh
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0 
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"   > Epoch {epoch+1} (Pat {patience_counter}/{config.EARLY_STOPPING_PATIENCE}): loss={train_loss:.4f}, dev_f1={f1:.4f}")
                print(f"   > Early stopping triggered.")
                break
        
        print(f"   > Epoch {epoch+1} (Pat {patience_counter}/{config.EARLY_STOPPING_PATIENCE}): loss={train_loss:.4f}, dev_f1={f1:.4f}")

    # 4. Final Evaluation using Best Dev Model
    if best_state: model.load_state_dict(best_state)
    test_prob, test_gold, _ = evaluate(model, test_loader, test_tsa, config)
    
    if len(test_prob) == 0: return None
    
    # Metrics
    binary_preds = (test_prob[:, config.REL_TYPE_IDX] > best_thresh).astype(int)
    binary_golds = test_gold[:, config.REL_TYPE_IDX].astype(int)
    
    precision, recall, micro_f1, _ = precision_recall_fscore_support(binary_golds, binary_preds, average='micro', zero_division=0.0)
    macro_f1 = f1_score(binary_golds, binary_preds, average='macro', zero_division=0.0)
    
    metrics = {
        "percentage": percentage,
        "n_samples": len(sub_ds),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "dev_best_f1": best_dev_f1,
        "best_threshold": best_thresh,
        "patience_setting": config.EARLY_STOPPING_PATIENCE
    }
    
    print(f"   >>> RESULT {percentage*100:.0f}%: Test Micro-F1: {micro_f1:.4f}")
    return metrics

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentages", nargs="+", type=float, default=Config.TRAIN_PERCENTAGES)
    parser.add_argument("--patience", type=int, default=Config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--dataset", default=Config.DATASET)
    args = parser.parse_args()
    
    config = Config()
    config.TRAIN_PERCENTAGES = sorted(args.percentages)
    config.EARLY_STOPPING_PATIENCE = args.patience
    config.DATASET = args.dataset
    
    os.makedirs(config.LOG_DIR, exist_ok=True)
    full_log_path = os.path.join(config.LOG_DIR, config.LOG_FILE)
    
    set_seed(config.RANDOM_SEED)

    print(f"Dataset: {config.DATASET}")
    print(f"Patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"Logging to: {full_log_path}")

    # Load Data Interface
    hf_ds = HFSpanDataset(config.DATASET, keep_relations=config.KEEP_RELATIONS)
    config.LABEL_LIST = hf_ds.ere_types
    config.NUM_LABELS = len(config.LABEL_LIST)
    
    excluded_lower = {e.lower() for e in config.EXCLUDED_RELS}
    config.REL_TYPE_IDX = [i for i, lab in enumerate(config.LABEL_LIST) if lab.lower() not in excluded_lower]

    # Load Validation & Test (Static)
    if not hf_ds.set_dataset_split(config.DATASET_DEV_SPLIT):
        print("!! Validation split missing. Using TEST for validation.")
        hf_ds.set_dataset_split(config.DATASET_TEST_SPLIT)
        
    dev_ds, dev_tsa = prepare_data(hf_ds, config)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    hf_ds.set_dataset_split(config.DATASET_TEST_SPLIT)
    test_ds, test_tsa = prepare_data(hf_ds, config)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    # Loop over percentages
    for pct in config.TRAIN_PERCENTAGES:
        # Pass the original hf_ds object, the function will deepcopy it
        results = run_experiment_fraction(pct, config, dev_loader, test_loader, dev_tsa, test_tsa, hf_ds)
        
        if results:
            append_log_to_jsonl(full_log_path, results)

    print(f"\nExperiment Complete. Data in {full_log_path}")

if __name__ == "__main__":
    main()