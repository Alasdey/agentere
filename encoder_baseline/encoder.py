#!/usr/bin/env python3
"""
encoder_baseline_simple.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Simplified Longformer pair classifier for Event Relation Extraction.
Produces a JSON results file keyed by HF sample ID.
"""

import os
import json
import math
import time
import random
import datetime
import argparse
import itertools
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # 'Maven-ERE-span', 'MECI-v0.1-public-span', 'Hievents-span', 
    # 'EventStoryLine-1.5-span', 'MAVEN-ERE-Causal-Events',
    # 'EventStoryLine-1.5-Causal'
    DATASET_NAME = 'MAVEN-ERE-Causal-Events'
    DATASET = f'Nofing/{DATASET_NAME}'
    DATASET_TRAIN_SPLIT = "train"
    DATASET_TEST_SPLIT = "test"
    # If non-empty, only these relation types are used (rest filtered out).
    # Empty list = use all relation types found in the dataset.
    # KEEP_RELATIONS: list = ['CauseEffect', 'EffectCause']
    KEEP_RELATIONS: list = ['PRECONDITION', 'CAUSE']
    # If True, only pairs that appear in annotations are used (MECI style).
    # If False, all (m_i, m_j) pairs with i≠j are enumerated.
    ANNOTATED_PAIRS_ONLY = False
    # Labels to exclude from loss masking and macro metrics.
    EXCLUDED_RELS = ["NoRel"]

    # Tokenizer / Encoder
    MODEL_NAME = "allenai/longformer-base-4096"
    MAX_SEQ_LENGTH = 4096
    ENCODER_TRAINED_LAYERS = [-1, -2, -3, -4, -5, -6]
    GLOBAL_MENTION = True

    # Training
    BATCH_SIZE = 12
    NUM_EPOCHS = 20
    LEARNING_RATE = 2e-5
    BATCH_SHUFFLE = True

    # Aggregation of token-level pair predictions → mention-level
    AGGREG = "mean"  # "mean", "max", "lse"

    # ASL Loss
    GAMMA_NEG = 4
    GAMMA_POS = 1
    CLIP = 0.05
    EPS = 1e-8

    # Eval
    THRESHOLD = 0.5
    THRESH_FLOOR = 3
    THRESH_STEPS = 100

    # Logging / Output
    TIME_START = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    LOG_DIR = f"./logs/{DATASET_NAME}_{DATASET_TEST_SPLIT}_{TIME_START}/"

    # Derived (set at runtime)
    LABEL_LIST: list = None
    NUM_LABELS: int = None
    REL_TYPE_MASK: list = None  # 1.0 per label, 0.0 for NoRel
    REL_TYPE_IDX: list = None   # indices of non-NoRel labels


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET INTERFACE  (loads the unified HF span format)
# ═══════════════════════════════════════════════════════════════════════════════

class HFSpanDataset:
    """
    Wraps a HuggingFace dataset in the {tokens, mentions, spans, relations} schema.
    Produces word_list and word_set_annotation (wsa) for the encoder pipeline.
    """

    def __init__(self, hf_path: str, keep_relations: list = None):
        self.ds = load_dataset(hf_path)
        self.modes = list(self.ds.keys())

        # Discover relation types from the first available split
        first_split = self.ds[self.modes[0]]
        all_rel_types = set()
        for row in first_split:
            all_rel_types.update(row["relations"].keys())

        if keep_relations:
            self.ere_types = sorted(
                [r for r in all_rel_types if r in keep_relations]
            )
        else:
            self.ere_types = sorted(all_rel_types)

        self.split_data = None  # set by set_dataset()

    def set_dataset(self, mode: str):
        if mode not in self.modes:
            raise ValueError(
                f"Split '{mode}' not in dataset. Available: {self.modes}"
            )
        self.split_data = self.ds[mode]

    def __len__(self):
        return len(self.split_data)

    
    # ── Build mentions info for one sample ─────────────────────────────────
    
    def mention_info(self) -> list:
        """
        Per-sample list of (mention_id, mention_text, frozenset_of_word_indices).
        This is the bridge between human-readable IDs and internal frozensets.
        """
        result = []
        for sample in self.split_data:
            tokens = sample["tokens"]
            doc_info = []
            for mid, span in zip(sample["mentions"], sample["spans"]):
                text = " ".join(
                    tokens[i] for i in sorted(span) if 0 <= i < len(tokens)
                )
                doc_info.append((mid, text, frozenset(span)))
            result.append(doc_info)
        return result

    # ── Build WSA for one sample ──────────────────────────────────────────

    def _sample_to_wsa(
        self, sample: dict, annotated_pairs_only: bool
    ) -> dict:
        """
        Returns {(frozenset_span_a, frozenset_span_b): [0,1,0,...]} for every
        relevant pair in the sample.
        """
        n_labels = len(self.ere_types)
        fsets = [frozenset(sp) for sp in sample["spans"]]

        wsa = {}

        if not annotated_pairs_only:
            # Enumerate ALL pairs i≠j (EventStoryLine, HiEvents, MAVEN-ERE style)
            for i, fs1 in enumerate(fsets):
                for j, fs2 in enumerate(fsets):
                    if i != j:
                        wsa[(fs1, fs2)] = [0] * n_labels
        # else: we only create entries for pairs that appear in relations

        for rel_type, pairs in sample["relations"].items():
            if rel_type not in self.ere_types:
                continue
            idx = self.ere_types.index(rel_type)
            for src_idx, tgt_idx in pairs:
                key = (fsets[src_idx], fsets[tgt_idx])
                if key not in wsa:
                    wsa[key] = [0] * n_labels
                wsa[key][idx] = 1

        return wsa

    # ── Bulk accessors used by the training pipeline ──────────────────────

    def word_list(self) -> list:
        return [sample["tokens"] for sample in self.split_data]

    def word_set_annotation(self, annotated_pairs_only: bool = False) -> list:
        return [
            self._sample_to_wsa(s, annotated_pairs_only)
            for s in self.split_data
        ]

    def ids(self) -> list:
        return [sample["id"] for sample in self.split_data]



# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class LongformerPairClassifier(nn.Module):
    def __init__(self, num_labels: int, config: Config):
        super().__init__()
        self.encoder = LongformerModel.from_pretrained(config.MODEL_NAME)

        # Freeze all, unfreeze last N layers
        for p in self.encoder.parameters():
            p.requires_grad = False
        for i in config.ENCODER_TRAINED_LAYERS:
            for p in self.encoder.encoder.layer[i].parameters():
                p.requires_grad = True

        h = self.encoder.config.hidden_size
        self.ffnn = nn.Sequential(
            nn.Linear(2 * h, 3 * h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(3 * h, 2 * h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        global_attention_mask,
        pair_indices,
        doc_indices,
        pair_labels,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        hidden = outputs.last_hidden_state  # (B, seq, h)

        pair_embeddings, all_indices, all_labels = [], [], []

        for b in range(input_ids.size(0)):
            h_doc = hidden[b]
            for k, (i, j) in enumerate(pair_indices[b]):
                pair_embeddings.append(torch.cat([h_doc[i], h_doc[j]], dim=-1))
                all_indices.append((doc_indices[b], b, i, j))
                all_labels.append(pair_labels[b][k])

        pair_embeddings = torch.stack(pair_embeddings)
        all_labels = torch.tensor(all_labels, device=pair_embeddings.device)
        logits = self.ffnn(pair_embeddings)

        return logits, all_indices, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET / DATALOADER
# ═══════════════════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    def __init__(self, documents: dict, max_length: int):
        self.documents = documents
        self.max_length = max_length

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
        "global_attention_mask": torch.stack(
            [b["global_attention_mask"] for b in batch]
        )
        if "global_attention_mask" in batch[0]
        else None,
        "pair_indices": [b["pair_indices"] for b in batch],
        "pair_labels": [b["pair_labels"] for b in batch],
        "doc_indices": [b["doc_indices"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _word_to_token(word_lists, token_offsets):
    """Maps each word to its corresponding subword token indices."""
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
                if w_off[0] < offsets[t_id][1]:
                    indices.add(t_id)
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
            key = (
                _set_word_to_tok(clust_pair[0], w2t[doc_idx]),
                _set_word_to_tok(clust_pair[1], w2t[doc_idx]),
            )
            doc[key] = gold
        result.append(doc)
    return result


def _tok_pair_annot(tok_set_annot):
    pair_indices, pair_labels = [], []
    for doc_annot in tok_set_annot:
        pi, pl = [], []
        for tok_set_pair, gold in doc_annot.items():
            for i in tok_set_pair[0]:
                for j in tok_set_pair[1]:
                    pi.append((i, j))
                    pl.append(gold)
        pair_indices.append(pi)
        pair_labels.append(pl)
    return pair_indices, pair_labels


def _create_global_attention_mask(pair_indices_list, attention_mask):
    gam = torch.zeros_like(attention_mask, dtype=torch.long)
    for doc_idx, doc_pairs in enumerate(pair_indices_list):
        mention_tokens = set(t for i, j in doc_pairs for t in (i, j))
        gam[doc_idx, 0] = 1
        gam[doc_idx, -1] = 1
        for t in mention_tokens:
            if 0 <= t < gam.size(1):
                gam[doc_idx, t] = 1
    return gam


def prepare_data(
    hf_dataset: HFSpanDataset, config: Config
) -> tuple:
    """
    Returns (PairDataset, tok_set_annot, tok_mention_maps).
    tok_mention_maps[doc_idx] = {token_frozenset: (mention_id, mention_text)}
    """
    wsa = hf_dataset.word_set_annotation(
        annotated_pairs_only=config.ANNOTATED_PAIRS_ONLY
    )
    word_lists = hf_dataset.word_list()
    mention_info = hf_dataset.mention_info()          # ← NEW
    words_joined = [" ".join(wl) for wl in word_lists]

    tokenizer = LongformerTokenizerFast.from_pretrained(
        config.MODEL_NAME, add_prefix_space=True
    )
    tokens = tokenizer(
        words_joined,
        is_split_into_words=False,
        return_offsets_mapping=True,
        padding="longest",
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    w2t = _word_to_token(word_lists, tokens["offset_mapping"])
    tok_set_annot = _tok_clust_pair_rel(wsa, w2t)
    pair_indices, pair_labels = _tok_pair_annot(tok_set_annot)

    # ── Build token-level mention maps ────────────────────────────────
    tok_mention_maps = []
    for doc_idx, doc_info in enumerate(mention_info):
        doc_map = {}
        for mid, mtext, word_fset in doc_info:
            tok_fset = _set_word_to_tok(word_fset, w2t[doc_idx])
            doc_map[tok_fset] = (mid, mtext)
        tok_mention_maps.append(doc_map)
    # ──────────────────────────────────────────────────────────────────

    if config.GLOBAL_MENTION:
        tokens["global_attention_mask"] = _create_global_attention_mask(
            pair_indices, tokens["attention_mask"]
        )

    documents = {
        "tokens": tokens,
        "pair_indices": pair_indices,
        "pair_labels": pair_labels,
    }
    ds = PairDataset(documents, config.MAX_SEQ_LENGTH)
    return ds, tok_set_annot, tok_mention_maps


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION  (token-pair preds → mention-cluster-pair preds)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_rel(tok_set_annot, indices, logits, tlabels, config):
    """
    Aggregates per-token-pair logits into per-mention-pair logits
    using mean/max/logsumexp pooling.
    """
    # Build reverse map: doc_idx → {(tok_i, tok_j) → flat_idx}
    doc_pair_map = defaultdict(dict)
    for idx, (doc_idx, _batch, ti, tj) in enumerate(indices):
        doc_pair_map[doc_idx][(ti, tj)] = idx

    preds, golds, inds = [], [], []

    for doc_idx, pred_pairs in doc_pair_map.items():
        for set_pair, label_list in tok_set_annot[doc_idx].items():
            label = torch.tensor(label_list)
            pair_logits = []
            for i in set_pair[0]:
                for j in set_pair[1]:
                    flat_idx = pred_pairs.get((i, j))
                    if flat_idx is not None:
                        pair_logits.append(logits[flat_idx])

            if not pair_logits:
                continue

            stacked = torch.stack(pair_logits)
            if config.AGGREG == "mean":
                preds.append(stacked.mean(dim=0))
            elif config.AGGREG == "max":
                preds.append(stacked.max(dim=0)[0])
            elif config.AGGREG == "lse":
                preds.append(torch.logsumexp(stacked, dim=0))

            golds.append(label)
            inds.append((doc_idx, set_pair))

    return torch.stack(preds), torch.stack(golds), inds


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS (Asymmetric Loss only)
# ═══════════════════════════════════════════════════════════════════════════════

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + (1 - y) * torch.log(
            xs_neg.clamp(min=self.eps)
        )

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = xs_pos * y + xs_neg * (1 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                weight = (1 - pt).pow(gamma)
            loss *= weight

        return -loss


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(
    preds_np: np.ndarray,
    golds_np: np.ndarray,
    eval_indices: list,
    floor: int = 3,
    steps: int = 100,
) -> tuple:
    """Searches for the threshold maximising micro-F1 on the eval labels."""
    best_score, best_thresh = 0.0, 0.5
    bot = math.exp(-floor)
    alpha = abs(1 - bot) / steps

    f1_avg = "binary" if preds_np.shape[1] == 1 else "micro"

    for step in range(steps):
        t = bot + alpha * step
        bp = (preds_np[:, eval_indices] > t).astype(int)
        score = f1_score(
            golds_np[:, eval_indices], bp, average=f1_avg, zero_division=0.0
        )
        if score >= best_score:
            best_score, best_thresh = score, t

    return best_score, best_thresh


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOPS
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, dataloader, tok_set_annot, optimizer, loss_fn, config):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="  train"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"]
        if gam is not None:
            gam = gam.to(config.DEVICE)

        optimizer.zero_grad()

        logits, indices, tlabels = model(
            input_ids, attn, gam,
            batch["pair_indices"],
            batch["doc_indices"],
            batch["pair_labels"],
        )

        agg_logits, golds, inds = aggregate_rel(
            tok_set_annot, indices, logits.cpu(), tlabels.cpu(), config
        )

        loss = loss_fn(agg_logits.to(config.DEVICE), golds.float().to(config.DEVICE))
        mask = torch.tensor(config.REL_TYPE_MASK, device=config.DEVICE)
        loss = (loss * mask).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model, dataloader, tok_set_annot, config
) -> tuple:
    """
    Returns (all_preds_np, all_golds_np, all_inds).
    all_inds maps each row to (doc_idx_in_split, (frozenset_a, frozenset_b)).
    """
    model.eval()
    all_preds, all_golds, all_inds = [], [], []

    for batch in tqdm(dataloader, desc="  eval"):
        input_ids = batch["input_ids"].to(config.DEVICE)
        attn = batch["attention_mask"].to(config.DEVICE)
        gam = batch["global_attention_mask"]
        if gam is not None:
            gam = gam.to(config.DEVICE)

        dummy_labels = [
            [[0] * config.NUM_LABELS for _ in pairs]
            for pairs in batch["pair_indices"]
        ]

        logits, indices, tlabels = model(
            input_ids, attn, gam,
            batch["pair_indices"],
            batch["doc_indices"],
            dummy_labels,
        )

        agg_logits, golds, inds = aggregate_rel(
            tok_set_annot, indices, logits.cpu(), tlabels.cpu(), config
        )

        preds = torch.sigmoid(agg_logits)
        all_preds.append(preds.numpy())
        all_golds.append(golds.numpy())
        all_inds.extend(inds)

    return np.concatenate(all_preds), np.concatenate(all_golds), all_inds


# ═══════════════════════════════════════════════════════════════════════════════
# JSON EXPORT  (link predictions to HF sample IDs)
# ═══════════════════════════════════════════════════════════════════════════════

def export_predictions_json(
    preds_np: np.ndarray,
    golds_np: np.ndarray,
    all_inds: list,
    threshold: float,
    sample_ids: list,
    label_list: list,
    eval_indices: list,
    tok_mention_maps: list,          # ← NEW parameter
    output_path: str,
    config: Config,
):
    """
    Writes a JSON file with per-sample, per-pair predictions keyed by HF ID.
    Source/target are identified by mention ID and surface text.
    """
    binary_preds = (preds_np[:, eval_indices] > threshold).astype(int)
    binary_golds = golds_np[:, eval_indices].astype(int)

    eval_labels = [label_list[i] for i in eval_indices]
    report_dict = classification_report(
        binary_golds, binary_preds,
        target_names=eval_labels,
        zero_division=0.0,
        digits=4,
        output_dict=True,
    )

    # Group predictions by document
    doc_predictions = defaultdict(list)

    for row_idx, (doc_idx, (fset_a, fset_b)) in enumerate(all_inds):
        hf_id = sample_ids[doc_idx]
        gold_vec = golds_np[row_idx].tolist()
        pred_vec = preds_np[row_idx].tolist()

        # Resolve frozensets → (mention_id, mention_text)
        src_id, src_text = tok_mention_maps[doc_idx].get(fset_a, ("?", "?"))
        tgt_id, tgt_text = tok_mention_maps[doc_idx].get(fset_b, ("?", "?"))

        # Best single label
        eval_probs = [pred_vec[i] for i in eval_indices]
        best_eval_idx = int(np.argmax(eval_probs))
        if eval_probs[best_eval_idx] > threshold:
            pred_label = label_list[eval_indices[best_eval_idx]]
        else:
            pred_label = "NoRel"

        # Gold single label (for readability)
        gold_eval = [gold_vec[i] for i in eval_indices]
        active_gold = [label_list[eval_indices[i]] for i, v in enumerate(gold_eval) if v > 0.5]
        gold_label = active_gold[0] if len(active_gold) == 1 else (
            "MultiLabel" if active_gold else "NoRel"
        )

        doc_predictions[hf_id].append(
            {
                "src": src_id,
                "src_text": src_text,
                "tgt": tgt_id,
                "tgt_text": tgt_text,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "gold_vec": gold_vec,
                "pred_prob": [round(p, 4) for p in pred_vec],
            }
        )

    payload = {
        "config": {
            "dataset": config.DATASET,
            "split": config.DATASET_TEST_SPLIT,
            "model": config.MODEL_NAME,
            "epochs": config.NUM_EPOCHS,
            "lr": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "aggregation": config.AGGREG,
            "annotated_pairs_only": config.ANNOTATED_PAIRS_ONLY,
        },
        "threshold": threshold,
        "label_list": label_list,
        "eval_label_indices": eval_indices,
        "global_metrics": report_dict,
        "samples": dict(doc_predictions),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Predictions exported to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=Config.DATASET)
    parser.add_argument("--train_split", default=Config.DATASET_TRAIN_SPLIT)
    parser.add_argument("--test_split", default=Config.DATASET_TEST_SPLIT)
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument(
        "--keep_relations", nargs="*", default=[],
        help="Only keep these relation labels (empty = all).",
    )
    parser.add_argument(
        "--annotated_pairs_only", default=Config.ANNOTATED_PAIRS_ONLY,
        help="If set, only annotated pairs are used (MECI mode).",
    )
    parser.add_argument(
        "--excluded_rels", nargs="*", default=Config.EXCLUDED_RELS,
        help="Labels to exclude from loss masking and macro metrics.",
    )
    args = parser.parse_args()

    config = Config()
    config.DATASET = args.dataset
    config.DATASET_TRAIN_SPLIT = args.train_split
    config.DATASET_TEST_SPLIT = args.test_split
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.KEEP_RELATIONS = args.keep_relations
    config.ANNOTATED_PAIRS_ONLY = args.annotated_pairs_only
    config.EXCLUDED_RELS = args.excluded_rels

    os.makedirs(config.LOG_DIR, exist_ok=True)
    set_seed(config.RANDOM_SEED)

    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET}")
    print(f"Annotated-pairs-only: {config.ANNOTATED_PAIRS_ONLY}")

    # ── 1. Load dataset interface ─────────────────────────────────────────
    hf_ds = HFSpanDataset(config.DATASET, keep_relations=config.KEEP_RELATIONS)
    config.LABEL_LIST = hf_ds.ere_types
    config.NUM_LABELS = len(config.LABEL_LIST)

    # Build mask: 0 for excluded labels, 1 otherwise
    excluded = {e.lower() for e in config.EXCLUDED_RELS}
    config.REL_TYPE_MASK = [
        0.0 if lab.lower() in excluded else 1.0 for lab in config.LABEL_LIST
    ]
    config.REL_TYPE_IDX = [
        i for i, lab in enumerate(config.LABEL_LIST) if lab.lower() not in excluded
    ]

    print(f"Labels: {config.LABEL_LIST}")
    print(f"Excluded: {config.EXCLUDED_RELS}")
    print(f"Eval indices: {config.REL_TYPE_IDX}")

    # ── 2. Prepare train data ────────────────────────────────────────────
    hf_ds.set_dataset(config.DATASET_TRAIN_SPLIT)
    train_ds, train_tsa, _ = prepare_data(hf_ds, config)      # ignore maps for train    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=config.BATCH_SHUFFLE,
    )
    print(f"Train: {len(train_ds)} docs")

    # ── 3. Prepare test data ─────────────────────────────────────────────
    hf_ds.set_dataset(config.DATASET_TEST_SPLIT)
    test_ids = hf_ds.ids()
    test_ds, test_tsa, test_mention_maps = prepare_data(hf_ds, config)  # keep maps
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )
    print(f"Test:  {len(test_ds)} docs")

    # ── 4. Model, optimizer, loss ─────────────────────────────────────────
    model = LongformerPairClassifier(config.NUM_LABELS, config).to(config.DEVICE)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE
    )
    loss_fn = AsymmetricLoss(
        gamma_neg=config.GAMMA_NEG,
        gamma_pos=config.GAMMA_POS,
        clip=config.CLIP,
        eps=config.EPS,
    )

    # ── 5. Training loop ─────────────────────────────────────────────────
    best_f1, best_thresh, best_epoch = 0.0, config.THRESHOLD, -1
    best_state = None

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n═══ Epoch {epoch}/{config.NUM_EPOCHS - 1} ═══")

        avg_loss = train_epoch(
            model, train_loader, train_tsa, optimizer, loss_fn, config
        )
        print(f"  avg train loss: {avg_loss:.4f}")

        # Evaluate
        preds_np, golds_np, inds = evaluate(model, test_loader, test_tsa, config)
        f1, thresh = find_best_threshold(
            preds_np, golds_np, config.REL_TYPE_IDX,
            config.THRESH_FLOOR, config.THRESH_STEPS,
        )
        print(f"  best micro-F1={f1:.4f} @ threshold={thresh:.4f}")

        # Full report at fixed threshold
        bp = (preds_np > thresh).astype(int)
        report = classification_report(
            golds_np, bp,
            target_names=config.LABEL_LIST,
            zero_division=0.0,
            digits=4,
        )
        report_path = os.path.join(config.LOG_DIR, f"report_epoch_{epoch:02d}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(report)

        if f1 > best_f1:
            best_f1, best_thresh, best_epoch = f1, thresh, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ★ new best (epoch {epoch})")

    print(f"\nBest: epoch={best_epoch}, F1={best_f1:.4f}, thresh={best_thresh:.4f}")

    # ── 6. Final evaluation with best model ──────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(config.DEVICE)

    preds_np, golds_np, inds = evaluate(model, test_loader, test_tsa, config)

    # ── 7. Export JSON ───────────────────────────────────────────────────
    json_path = os.path.join(config.LOG_DIR, "predictions.json")
    export_predictions_json(
        preds_np, golds_np, inds,
        best_thresh, test_ids,
        config.LABEL_LIST, config.REL_TYPE_IDX,
        test_mention_maps,                                     # ← pass maps
        json_path, config,
    )

    # Also save model weights
    torch.save(best_state, os.path.join(config.LOG_DIR, "best_model.pt"))

    # Save config
    config_path = os.path.join(config.LOG_DIR, "config.json")
    cfg_dict = {
        k: v for k, v in vars(config).items()
        if not k.startswith("_") and not callable(v)
    }
    cfg_dict["DEVICE"] = str(config.DEVICE)
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print(f"\nAll outputs in: {config.LOG_DIR}")


if __name__ == "__main__":
    main()