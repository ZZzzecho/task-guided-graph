"""Local North Medical records; preserve eight independent diagnosis labels."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

LABEL_NAMES = (
    '前交叉韧带断裂', '后交叉韧带损伤', '内侧半月板损伤', '外侧半月板损伤',
    '内侧副韧带损伤', '外侧副韧带损伤', '髌骨脱位', '髌股关节软骨损伤',
)


def normalize_diagnosis(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    y = np.asarray(value)
    if y.shape != (8,) or not np.isin(y, [0, 1]).all():
        raise ValueError('diagnosis must be an eight-dimensional binary multi-hot vector')
    return y.astype(np.int64).tolist()


def record_id(text):
    return hashlib.sha256(str(text).strip().encode('utf-8')).hexdigest()


def split_training(rows, *, seed=11, reward_fraction=.1):
    """Keep duplicate descriptions in one split; never use validation/test labels."""
    if not 0 < reward_fraction < 1:
        raise ValueError('reward_fraction must lie between zero and one')
    groups = sorted({record_id(r['description']) for r in rows})
    n = max(1, int(round(len(groups) * reward_fraction)))
    if n >= len(groups):
        raise ValueError('need nonempty train and reward groups')
    rng = np.random.default_rng(seed)
    all_positive = np.asarray([normalize_diagnosis(r['diagnosis']) for r in rows]).sum(0) > 0
    for _ in range(100):
        reward_ids = set(rng.permutation(groups)[:n])
        train = [r for r in rows if record_id(r['description']) not in reward_ids]
        reward = [r for r in rows if record_id(r['description']) in reward_ids]
        supported = not all_positive.any() or all(
            np.asarray([normalize_diagnosis(r['diagnosis']) for r in part]).sum(0)[all_positive].min() > 0
            for part in (train, reward))
        if supported:
            return train, reward
    raise ValueError('could not retain positive support for every present label in both splits')


def prepare_north_data(raw_dir, concepts_csv, output, *, seed=11, reward_fraction=0.):
    import csv
    import pandas as pd
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = Path(raw_dir)
    # Intentionally never load test_data.npy.
    paths = [raw_dir/n for n in ('train_data.npy', 'data0.npy', 'val_data.npy')]
    loaded = [np.load(p, allow_pickle=True).tolist() for p in paths]
    for rows in loaded:
        for r in rows:
            if not isinstance(r, dict) or not str(r.get('description', '')).strip():
                raise ValueError('invalid clinical description')
            r['diagnosis'] = normalize_diagnosis(r['diagnosis'])
    merged = loaded[0] + loaded[1]
    val_ids = {record_id(r['description']) for r in loaded[2]}
    if val_ids & {record_id(r['description']) for r in merged}:
        raise ValueError('train/validation description overlap')
    # Default explicitly uses the full merged training set for both fitting and reward.
    if reward_fraction == 0:
        train, reward = merged, merged
    else:
        train, reward = split_training(merged, seed=seed, reward_fraction=reward_fraction)
    for name, rows in [('train', train), ('reward', reward), ('val', loaded[2])]:
        with (output/f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps({'id': record_id(r['description']), 'text': r['description'],
                                    'labels': r['diagnosis']}, ensure_ascii=False)+'\n')
    pd.DataFrame({'sample_id': [f'{record_id(r["description"])}-{i}' for i,r in enumerate(train)],
                  'text': [r['description'] for r in train]}).to_csv(output/'graph_train.csv.gz', index=False)
    with Path(concepts_csv).open(encoding='utf-8-sig', newline='') as f:
        terms = [row[1].strip() for row in list(csv.reader(f))[1:] if len(row)>1]
    terms = list(dict.fromkeys(t for t in terms if t not in ('', '-', '0')))
    concepts = [{'concept_id': f'north-{i:04d}', 'concept_text': t} for i,t in enumerate(terms)]
    (output/'concepts.json').write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = {'dataset': 'North Medical ch_data', 'label_names': LABEL_NAMES, 'seed': seed,
             'raw_train': len(loaded[0]), 'data0_training': len(loaded[1]), 'merged_train': len(merged),
             'task_train': len(train), 'reward': len(reward), 'validation': len(loaded[2]),
             'concepts': len(concepts), 'test_split_touched': False,
             'reward_mode': 'full_training' if reward_fraction == 0 else 'grouped_training_holdout',
             'split_unit': 'original train/validation files; patient IDs unavailable',
             'input_semantics': 'existing description, unchanged; legacy includes history/examination/MRI',
             'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(concepts_csv)]},
             'positive_counts': {k: np.asarray([r['diagnosis'] for r in v]).sum(0).tolist()
                                 for k,v in [('train',train),('reward',reward),('val',loaded[2])]},
             'concept_source': 'existing medical keywords CSV; no new extraction from validation/test'}
    (output/'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit
