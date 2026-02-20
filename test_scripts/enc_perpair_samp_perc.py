#!/usr/bin/env python3
"""
encoder_learning_curve_continuous.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Experiment: Train on x% of data to demonstrate data efficiency.
Features:
 - Continuous logging (JSONL) to prevent data loss on crash.
 - Configurable Early Stopping (Patience).
 - Prompt-based input: "context; m1 and m2 are in a relation [MASK]"
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
    MAX_SEQ_LENGTH = 4000 # Reduced slightly as we are adding prompt text
    ENCODER_TRAINED_LAYERS = [-1, -2, -3, -4, -5, -6]
    
    # Training
    BATCH_SIZE = 8 # Reduced batch size likely needed as we explode docs into pairs
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5
    
    # Validation / Patience
    EARLY_STOPPING_PATIENCE = 5 

    # Aggregation
    AGGREG = "max" # Changed default to max for cleaner signal in prompt setup

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
        # Use simple span sets for hashing
        fsets = [frozenset(sp) for sp in sample["spans"]]
        wsa = {}
        
        # Pre-fill negatives if we aren't restricted to annotated pairs only
        if not annotated_only:
            for i, fs1 in enumerate(fsets):
                for j, fs2 in enumerate(fsets):
                    if i != j:
                        wsa[(fs1, fs2)] = [0] * n_labels
        
        # Fill positives
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
        
        # Freeze params first
        for p in self.encoder.parameters(): p.requires_grad = False
        # Unfreeze specific layers
        for i in config.ENCODER_TRAINED_LAYERS:
            for p in self.encoder.encoder.layer[i].parameters(): p.requires_grad = True

        h = self.encoder.config.hidden_size
        
        # Input is just 'h' now, because it's the [MASK] embedding
        self.ffnn = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, num_labels),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask, mask_token_id):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        last_hidden_state = outputs.last_hidden_state # (Batch, Seq, Hidden)
        
        # Extract embedding at the [MASK] token position
        # mask_token_id is usually passed in, or we find it in input_ids
        mask_mask = (input_ids == mask_token_id)
        
        # Check if we found exactly one mask per sequence (mostly for sanity, or handle edge cases)
        # Note: argmax finds the first occurrence.
        mask_indices = mask_mask.float().argmax(dim=1) 
        
        batch_size = input_ids.shape[0]
        # Gather the hidden state at the mask index
        # shape: (Batch, Hidden)
        mask_embeddings = last_hidden_state[torch.arange(batch_size), mask_indices]
        
        logits = self.ffnn(mask_embeddings)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-PROCESSING (PROMPT BASED)
# ═══════════════════════════════════════════════════════════════════════════════

class PromptPairDataset(Dataset):
    def __init__(self, encodings, labels, pair_info):
        self.encodings = encodings
        self.labels = labels
        self.pair_info = pair_info # List of (doc_idx, (span_src, span_tgt))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        # Store metadata to reconstruct predictions later
        item['pair_info_idx'] = idx 
        return item

def prompt_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "global_attention_mask": torch.stack([b["global_attention_mask"] for b in batch]) if "global_attention_mask" in batch[0] else None,
        "labels": torch.stack([b["labels"] for b in batch]),
        "pair_info_idx": [b["pair_info_idx"] for b in batch]
    }

def get_span_text(tokens, span_indices):
    # span_indices is (start, end)
    # Simple join, assuming whitespace separated (imperfect but standard for this logic)
    ts = tokens[span_indices[0]:span_indices[1]]
    return " ".join(ts)

def prepare_data(hf_dataset: HFSpanDataset, config: Config):
    """
    Transforms the dataset into a pairwise prompt format:
    Input: "Context; {m1} and {m2} are in a relation <mask>"
    """
    wsa = hf_dataset.word_set_annotation(config.ANNOTATED_PAIRS_ONLY)
    word_lists = hf_dataset.word_list()
    # Need original tokens to reconstruct text for prompts
    
    tokenizer = LongformerTokenizerFast.from_pretrained(config.MODEL_NAME)
    
    # We need to know what the mask token is for the model
    # Longformer uses <mask> usually
    mask_token = tokenizer.mask_token if tokenizer.mask_token is not None else "<mask>"
    
    all_texts = []
    all_labels = []
    all_pair_info = [] # (doc_idx, frozenset_src, frozenset_tgt)
    
    print("Preparing prompt-based data...")
    for doc_idx, (doc_wsa, doc_tokens) in enumerate(zip(wsa, word_lists)):
        context_str = " ".join(doc_tokens)
        
        # doc_wsa keys are (frozenset_src, frozenset_tgt)
        # We need to convert frozensets back to text
        # Since frozenset doesn't preserve order, we rely on the fact that span indices 
        # usually map to the word list. 
        # However, `wsa` abstracted spans. We need to find *a* span for the frozenset.
        # This is tricky with multiple coreferent mentions. 
        # Strategy: Pick the first mention occurrence in the cluster (earliest start index).
        
        for (fs_src, fs_tgt), label_vec in doc_wsa.items():
            
            # Helper to retrieve text from frozenset indices
            def fs_to_text(fs):
                idx_list = sorted(list(fs))
                if not idx_list: return "unknown"
                # Assume contiguous for text generation or just join
                return " ".join([doc_tokens[i] for i in idx_list])

            m1_text = fs_to_text(fs_src)
            m2_text = fs_to_text(fs_tgt)
            
            # Construct Prompt
            # "{context}; {m1} and {m2} are in a relation {mask}"
            prompt = f"{context_str} ; {m1_text} and {m2_text} are in a relation {mask_token}"
            
            all_texts.append(prompt)
            all_labels.append(label_vec)
            all_pair_info.append((doc_idx, (fs_src, fs_tgt)))

    # Batch tokenize
    print(f"Tokenizing {len(all_texts)} pairs...")
    encodings = tokenizer(all_texts, padding=True, truncation=True, max_length=config.MAX_SEQ_LENGTH, return_tensors="pt")
    
    # Create Global Attention Mask
    # Attention on [cls] (0) and [mask] token
    input_ids = encodings["input_ids"]
    mask_token_id = tokenizer.mask_token_id
    
    gam = torch.zeros_like(input_ids)
    gam[:, 0] = 1 # Global on CLS
    
    # Find mask index and set global attention
    mask_mask = (input_ids == mask_token_id)
    gam[mask_mask] = 1 # Global on MASK
    
    encodings["global_attention_mask"] = gam
    
    dataset = PromptPairDataset(encodings, all_labels, all_pair_info)
    
    # We return the dataset and the pair_info list (to map back during eval)
    return dataset, all_pair_info, tokenizer.mask_token_id


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_predictions(logits, pair_info_indices, all_pair_info, config):
    """
    Since input is now exploded (1 input = 1 pair), this function organizes
    predictions back into the structure required for document-level evaluation.
    """
    # Map: doc_idx -> (src, tgt) -> list_of_predictions
    results_map = defaultdict(lambda: defaultdict(list))
    
    sigmoid_preds = torch.sigmoid(logits).cpu().numpy()
    
    for i, pred in enumerate(sigmoid_preds):
        p_idx = pair_info_indices[i] # Index into the global pairwise list
        doc_idx, pair_key = all_pair_info[p_idx]
        results_map[doc_idx][pair_key].append(pred)

    return results_map

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
    
    # optimization: check if empty
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
        gam = batch["global_attention_mask"].to(config.DEVICE) if batch["global_attention_mask"] is not None else None
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
    
    # We don't need complex aggregation here because the dataset is already 1-to-1 with pairs.
    # If the original dataset had duplicate pairs (coreference clusters), we might want to aggregate,
    # but for simplicity in this prompt-based modification, we treat every prompt instance as a prediction point.
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"].to(config.DEVICE) if batch["global_attention_mask"] is not None else None
        
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

def run_experiment_fraction(percentage, config, hf_ds_original, mask_token_id):
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
    
    train_ds, _, _ = prepare_data(hf_subset, config)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, collate_fn=prompt_collate_fn, shuffle=True)
    
    if len(train_ds) == 0:
        print("   > Skipping: 0 samples.")
        return None

    # Prepare Dev/Test lazily inside loop to ensure consistent tokenization if needed, 
    # but ideally should be outside. For structure, we create them here or pass them in.
    # To keep func signature clean, we rebuild dev loader here quickly (overhead is low for small data).
    hf_ds_original.set_dataset_split(config.DATASET_DEV_SPLIT)
    dev_ds, _, _ = prepare_data(hf_ds_original, config)
    dev_loader = DataLoader(dev_ds, batch_size=config.BATCH_SIZE, collate_fn=prompt_collate_fn, shuffle=False)

    hf_ds_original.set_dataset_split(config.DATASET_TEST_SPLIT)
    test_ds, _, _ = prepare_data(hf_ds_original, config)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, collate_fn=prompt_collate_fn, shuffle=False)

    # 3. Training Loop with Patience
    best_dev_f1 = 0.0
    best_thresh = 0.5
    best_state = None
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config, mask_token_id)
        
        # Validation
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

    # 4. Final Evaluation using Best Dev Model
    if best_state: model.load_state_dict(best_state)
    test_prob, test_gold = evaluate(model, test_loader, config, mask_token_id)
    
    if len(test_prob) == 0: return None
    
    # Metrics
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

    # Initialize Tokenizer to get Mask Token ID immediately
    tokenizer = LongformerTokenizerFast.from_pretrained(config.MODEL_NAME)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask token defined!")

    # Load Validation & Test (Static) - Check splits exist
    possible_devs = ["validation", "dev", "val"]
    dev_split = config.DATASET_DEV_SPLIT
    if dev_split not in hf_ds.ds:
        for p in possible_devs:
            if p in hf_ds.ds:
                dev_split = p
                break
    config.DATASET_DEV_SPLIT = dev_split
    print(f"Using dev split: {config.DATASET_DEV_SPLIT}")
    
    # Loop over percentages
    for pct in config.TRAIN_PERCENTAGES:
        # Pass hf_ds and mask_token_id
        results = run_experiment_fraction(pct, config, hf_ds, mask_token_id)
        
        if results:
            append_log_to_jsonl(full_log_path, results)

    print(f"\nExperiment Complete. Data in {full_log_path}")

if __name__ == "__main__":
    main()