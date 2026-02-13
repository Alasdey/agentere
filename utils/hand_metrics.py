import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def dict_load(json_path):
    """
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def keyy(data):
    """
    """
    seen_keys = set()
    res = set()

    def walk(obj, prefix="/"):

        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in seen_keys:
                    res.add(prefix + k)
                    seen_keys.add(k)
                    # print(f"{prefix}- {k}")
                walk(v, prefix + k + "/")

        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)

    walk(data)

    return res


def keyy_ff(json_path):
    """
    """
    data = dict_load(json_path)
    res = keyy(data)

    return res


def binary_metrics(y_true, y_pred, pos_label='Causal'):
    """
    Helper to calculate micro precision, recall, and F1 for a specific positive label.
    """
    # We use average='binary' here because we have mapped everything to 
    # a binary problem (Causal vs NoRel).
    # If the set is empty (e.g., no predictions for a specific type), handle gracefully.
    try:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0
        )
    except ValueError:
        return 0.0, 0.0, 0.0
        
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4)
    }


def evaluate_whole(df, pred_name: str="unified_pred", gold_name: str="unified_gold"):
    """
    """
    res = {}
    
    res = binary_metrics(
        df[gold_name], 
        df[pred_name]
    )
    
    return res


def evaluate_per_lang(df, pred_name: str="unified_pred", gold_name: str="unified_gold"):
    """
    Per Language Unified Scores
    """
    res = {}

    # Lising the language in the data
    languages = df['lang'].unique()
    
    for lang in languages:
        lang_df = df[df['lang'] == lang]
        res[lang] = binary_metrics(
            lang_df[gold_name], 
            lang_df[pred_name]
        )
    return res


def reverse_pair_key(pair_str):
    """
    Flips 'T10,T11' to 'T11,T10'.
    """
    parts = pair_str.split(',')
    if len(parts) != 2: raise ValueError(f"{pair_str} is not a reversible pair")
    return f"{parts[1]},{parts[0]}"


def flatten_pairs_solo(df, key_name: str="gold"):
    """
    """
    res = set()
    
    for _, row in df.iterrows():
        res.add((row['id'], row['pair'], row[key_name]))
    
    return res


def causal_simple(df_part, causal_list=["EffectCause", "CauseEffect"]):
    """
    """
    res = df_part.apply(lambda x: 'Causal' if x in causal_list else 'NoRel')
    return res


# NoClip not implemented
def symmetrize(df, rel_name="CauseEffect", key_name="gold", no_clip=False, default_rel='NoRel'):
    """
    """
    flat = flatten_pairs_solo(df, key_name=key_name)
    # print(flat)
    flat_exist = set(line[:2] for line in flat)
    # print("flat[0]", flat_exist[0])
    
    def symmetrize_row(row):
        rev_pair = reverse_pair_key(row['pair'])
        if not (row['id'], rev_pair) in flat_exist and no_clip:
            # if row['id']=="conflict-week4-isik-2981956_chunk_4.ann":
            #     print(row['id'], row['pair'], (row['id'], rev_pair) in flat_exist)
            return default_rel
        if row[key_name] == rel_name:
            return rel_name
        if (row['id'], rev_pair, rel_name) in flat:
            # print(row['id'], rev_pair, rel_name)
            # print("uwu")
            return rel_name
        return default_rel
    
    return df.apply(symmetrize_row, axis=1)


def evaluate(df, pred_name: str="unified_pred", gold_name: str="unified_gold"):
    """
    """
    res = {}
    res['per_lang'] = evaluate_per_lang(df, pred_name=pred_name, gold_name=gold_name)
    res['overall'] = evaluate_whole(df, pred_name=pred_name, gold_name=gold_name)
    return res


def data_to_df(data):
    """
    """
    predictions = data['results']['per_pair_predictions']
    # Dataframe
    df = pd.DataFrame(predictions)
    return df


def analyse(data):
    # Get individual predictions from the file
    res = {}
    gold_name = 'unified_gold'
    binary_pred = 'binary_pred'
    all_rel = ['CauseEffect', 'EffectCause']
    ec_rel = ['EffectCause']
    
    
    predictions = data['results']['per_pair_predictions']
    # Dataframe
    df = pd.DataFrame(predictions)
    # Unify the gold
    df[gold_name] = causal_simple(df['gold'], all_rel)
    # Binarize the preds 
    df[binary_pred] = causal_simple(df['pred'], all_rel)
    # Binarize the preds from EffectCause only
    df["ec_only"] = causal_simple(df['pred'], ec_rel)
    
    df["clipped_sym_ec"] = symmetrize(df, rel_name="Causal", key_name="ec_only", no_clip=True, default_rel='NoRel')
    df["sym_ec"] = symmetrize(df, rel_name="Causal", key_name="ec_only", no_clip=False, default_rel='NoRel')

    # return df[:50]
    
    res["direct_binary"] = evaluate(df, pred_name=binary_pred, gold_name=gold_name)
    res["clipped_sym_ec"] = evaluate(df, pred_name="clipped_sym_ec", gold_name=gold_name)
    res["sym_ec"] = evaluate(df, pred_name="sym_ec", gold_name=gold_name)
    
    return res


def analyse_ff(json_path):
    """
    """
    data = dict_load(json_path)
    res = analyse(data)

    return res
