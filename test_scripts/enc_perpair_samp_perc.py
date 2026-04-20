#!/usr/bin/env python3
"""
encoder_learning_curve_continuous.py (Refactored)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Refactored to tokenize lazily and cache validation data.
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
from transformers import LongformerModel, LongformerTokenizerFast
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
    DATASET_DEV_SPLIT = "dev"  
    DATASET_TEST_SPLIT = "test"
    
    KEEP_RELATIONS: list = ['PRECONDITION', 'CAUSE']
    ANNOTATED_PAIRS_ONLY = False
    EXCLUDED_RELS = ["NoRel"]

    # Experiment: Learning Curve
    TRAIN_PERCENTAGES = [0.01, 0.025, 0.05, 0.1] 

    # Tokenizer / Encoder
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_SEQ_LENGTH = 4000
    ENCODER_TRAINED_LAYERS = [-1, -2, -3, -4, -5, -6]
    
    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5
    
    # Validation / Patience
    EARLY_STOPPING_PATIENCE = 5 

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
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def append_log_to_jsonl(filepath, data_dict):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict) + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class HFSpanDataset:
    def __init__(self, hf_path: str, keep_relations: list = None):
        self.ds = load_dataset(hf_path)
        self.modes = list(self.ds.keys())

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
        if percentage >= 1.0: return 
        full_len = len(self.split_data)
        all_indices = list(range(full_len))
        rng = random.Random(Config.RANDOM_SEED)
        rng.shuffle(all_indices)
        cutoff = int(full_len * percentage)
        if cutoff == 0 and percentage > 0: cutoff = 1
        selected_indices = all_indices[:cutoff]
        self.split_data = self.split_data.select(selected_indices)

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
    
    def spans(self):
        return [sample["spans"] for sample in self.split_data]

    def word_set_annotation(self, annotated_pairs_only=False):
        return [self._sample_to_wsa(s, annotated_pairs_only) for s in self.split_data]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class LongformerPairClassifier(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super().__init__()
        self.config = config
        self.encoder = LongformerModel.from_pretrained(config.MODEL_NAME)
        
        for p in self.encoder.parameters(): p.requires_grad = False
        for i in config.ENCODER_TRAINED_LAYERS:
            for p in self.encoder.encoder.layer[i].parameters(): p.requires_grad = True

        h = self.encoder.config.hidden_size
        
        self.ffnn = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, num_labels),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        
        # Extract embedding at the  token position
        # We rely on global_attention_mask to identify the mask token 
        # (since we set it in the dataset)
        # Longformer requires global attention on the mask token for it to attend to the whole doc.
        # We assume 1 mask per sequence.
        
        mask_token_mask = (global_attention_mask == 1)
        # handle the CLS token also having global attention usually. 
        # We need the specific token. Let's check input_ids for mask_token_id.
        # But input_ids are passed in. 
        # Safer: Find where global_attention_mask is 1, excluding index 0 (CLS).
        
        # Heuristic: The mask token is usually the last token with global attention, 
        # or we can search input_ids if we pass mask_id.
        # Let's search input_ids for the mask token ID.
        # Note: We need mask_token_id passed here. 
        # For cleaner code, let's assume the Dataset logic handles it.
        
        pass # Logic moved to using input_ids to find mask
        
    def forward(self, input_ids, attention_mask, global_attention_mask, mask_token_id):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        mask_mask = (input_ids == mask_token_id)
        
        # If no mask found (shouldn't happen), fallback to CLS
        if not mask_mask.any():
             mask_indices = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        else:
             mask_indices = mask_mask.float().argmax(dim=1)
        
        batch_size = input_ids.shape[0]
        mask_embeddings = last_hidden_state[torch.arange(batch_size), mask_indices]
        
        logits = self.ffnn(mask_embeddings)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY DATASET & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def get_span_text(tokens, span_indices):
    ts = tokens[span_indices[0]:span_indices[1]]
    return " ".join(ts)

class LazyPromptDataset(Dataset):
    def __init__(self, texts, labels, pair_info, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.pair_info = pair_info
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize on the fly
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            padding=False # No padding here, collator handles it
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Setup Global Attention: CLS (0) and MASK token
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1 # CLS
        
        # Find Mask Token
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            global_attention_mask[mask_indices[0]] = 1
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "labels": torch.tensor(label, dtype=torch.float),
            "pair_info_idx": idx
        }

def prompt_collate_fn(batch, tokenizer):
    # Pad dynamically
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch], 
        batch_first=True, 
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    global_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['global_attention_mask'] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    
    labels = torch.stack([item['labels'] for item in batch])
    pair_info_idx = [item['pair_info_idx'] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
        "labels": labels,
        "pair_info_idx": pair_info_idx
    }

def prepare_raw_data(hf_dataset: HFSpanDataset, config: Config):
    """
    Transforms dataset into raw lists (texts, labels, pair_info).
    No tokenization happens here.
    """
    wsa = hf_dataset.word_set_annotation(config.ANNOTATED_PAIRS_ONLY)
    word_lists = hf_dataset.word_list()
    
    all_texts = []
    all_labels = []
    all_pair_info = []
    
    for doc_idx, (doc_wsa, doc_tokens) in enumerate(zip(wsa, word_lists)):
        context_str = " ".join(doc_tokens)
        
        for (fs_src, fs_tgt), label_vec in doc_wsa.items():
            def fs_to_text(fs):
                idx_list = sorted(list(fs))
                if not idx_list: return "unknown"
                return " ".join([doc_tokens[i] for i in idx_list])

            m1_text = fs_to_text(fs_src)
            m2_text = fs_to_text(fs_tgt)
            
            # Prompt construction
            prompt = f"{context_str} ; {m1_text} and {m2_text} are in a relation {tokenizer.mask_token}"
            
            all_texts.append(prompt)
            all_labels.append(label_vec)
            all_pair_info.append((doc_idx, (fs_src, fs_tgt)))
            
    return all_texts, all_labels, all_pair_info


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & EVAL
# ═══════════════════════════════════════════════════════════════════════════════

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
    if preds_np.shape[0] == 0: return 0.0, 0.5

    for step in range(steps):
        t = bot + alpha * step
        bp = (preds_np[:, eval_indices] > t).astype(int)
        score = f1_score(golds_np[:, eval_indices], bp, average=f1_avg, zero_division=0.0)
        if score >= best_score: best_score, best_thresh = score, t
    return best_score, best_thresh

def train_epoch(model, dataloader, optimizer, loss_fn, config, mask_token_id):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)
        
        optimizer.zero_grad()
        logits = model(input_ids, attn, gam, mask_token_id)
        
        loss = loss_fn(logits, labels).mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)

@torch.no_grad()
def evaluate(model, dataloader, config, mask_token_id):
    model.eval()
    all_preds = []
    all_golds = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"].to(config.DEVICE)
        
        logits = model(input_ids, attn, gam, mask_token_id)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        golds = batch["labels"].cpu().numpy()
        
        all_preds.append(preds)
        all_golds.append(golds)

    if not all_preds: return np.array([]), np.array([])
    return np.concatenate(all_preds), np.concatenate(all_golds)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment_fraction(percentage, config, hf_ds_original, mask_token_id, tokenizer, dev_loader, test_loader):
    print(f"\n██████ STARTING RUN: {percentage*100:.0f}% Training Data ██████")
    
    # 1. Reset Model
    model = LongformerPairClassifier(config.NUM_LABELS, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    loss_fn = AsymmetricLoss(gamma_neg=config.GAMMA_NEG, gamma_pos=config.GAMMA_POS, clip=config.CLIP)

    # 2. Subset Training Data
    hf_subset = copy.deepcopy(hf_ds_original)
    if not hf_subset.set_dataset_split(config.DATASET_TRAIN_SPLIT):
        raise ValueError(f"Train split {config.DATASET_TRAIN_SPLIT} not found.")
    
    hf_subset.subset_current_split(percentage)
    
    # Prepare only raw text for the subset
    train_texts, train_labels, train_pair_info = prepare_raw_data(hf_subset, config)
    
    if len(train_texts) == 0:
        print("   > Skipping: 0 samples.")
        return None

    # Create lazy dataset for training
    train_ds = LazyPromptDataset(train_texts, train_labels, train_pair_info, tokenizer, config.MAX_SEQ_LENGTH)
    # Use functools.partial to pass tokenizer to collate_fn
    from functools import partial
    collator = partial(prompt_collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=collator, shuffle=True)

    # 3. Training Loop
    best_dev_f1 = 0.0
    best_thresh = 0.5
    best_state = None
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config, mask_token_id)
        
        # Validation (using passed-in dev_loader)
        dev_prob, dev_gold = evaluate(model, dev_loader, config, mask_token_id)
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
                print(f"   > Epoch {epoch+1}: loss={train_loss:.4f}, dev_f1={f1:.4f} [Early Stop]")
                break
        
        print(f"   > Epoch {epoch+1}: loss={train_loss:.4f}, dev_f1={f1:.4f}")

    # 4. Final Evaluation
    if best_state: model.load_state_dict(best_state)
    test_prob, test_gold = evaluate(model, test_loader, config, mask_token_id)
    
    if len(test_prob) == 0: return None
    
    binary_preds = (test_prob[:, config.REL_TYPE_IDX] > best_thresh).astype(int)
    binary_golds = test_gold[:, config.REL_TYPE_IDX].astype(int)
    
    precision, recall, micro_f1, _ = precision_recall_fscore_support(binary_golds, binary_preds, average='micro', zero_division=0.0)
    macro_f1 = f1_score(binary_golds, binary_preds, average='macro', zero_division=0.0)
    
    metrics = {
        "percentage": percentage,
        "n_samples": len(train_ds),
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
    print(f"Logging to: {full_log_path}")

    # Load Data Interface
    hf_ds = HFSpanDataset(config.DATASET, keep_relations=config.KEEP_RELATIONS)
    config.LABEL_LIST = hf_ds.ere_types
    config.NUM_LABELS = len(config.LABEL_LIST)
    
    excluded_lower = {e.lower() for e in config.EXCLUDED_RELS}
    config.REL_TYPE_IDX = [i for i, lab in enumerate(config.LABEL_LIST) if lab.lower() not in excluded_lower]

    # Initialize Tokenizer
    global tokenizer # Make global or pass around
    tokenizer = LongformerTokenizerFast.from_pretrained(config.MODEL_NAME)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask token defined!")

    # Prepare Validation & Test ONLY ONCE
    possible_devs = ["validation", "dev", "val"]
    dev_split = config.DATASET_DEV_SPLIT
    if dev_split not in hf_ds.ds:
        for p in possible_devs:
            if p in hf_ds.ds:
                dev_split = p
                break
    config.DATASET_DEV_SPLIT = dev_split
    
    print("Preparing static Dev set...")
    hf_ds.set_dataset_split(config.DATASET_DEV_SPLIT)
    dev_texts, dev_labels, dev_pairs = prepare_raw_data(hf_ds, config)
    dev_ds = LazyPromptDataset(dev_texts, dev_labels, dev_pairs, tokenizer, config.MAX_SEQ_LENGTH)
    
    print("Preparing static Test set...")
    hf_ds.set_dataset_split(config.DATASET_TEST_SPLIT)
    test_texts, test_labels, test_pairs = prepare_raw_data(hf_ds, config)
    test_ds = LazyPromptDataset(test_texts, test_labels, test_pairs, tokenizer, config.MAX_SEQ_LENGTH)
    
    # Collator
    from functools import partial
    collator = partial(prompt_collate_fn, tokenizer=tokenizer)

    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, collate_fn=collator, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=collator, shuffle=False)
    
    # Loop over percentages
    for pct in config.TRAIN_PERCENTAGES:
        # Pass static loaders and tokenizer
        results = run_experiment_fraction(pct, config, hf_ds, mask_token_id, tokenizer, dev_loader, test_loader)
        
        if results:
            append_log_to_jsonl(full_log_path, results)

    print(f"\nExperiment Complete. Data in {full_log_path}")

if __name__ == "__main__":
    main()