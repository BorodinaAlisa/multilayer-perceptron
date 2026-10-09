import argparse
import os
import pickle
import numpy as np


def load_csv(data):
    labels, feats = [], []
    with open(data) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 32:
                continue
            labels.append(1 if p[1] == 'M' else 0)
            feats.append([float(v) for v in p[2:]])
    return np.array(feats), np.array(labels)


def split_dataset(data, seed, valid_size, out_train, out_valid):
    with open(data) as f:
        rows = [l for l in f.read().splitlines() if l.strip()]
    labels = np.array([r.split(',')[1] for r in rows])
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_val = int(round(len(idx) * valid_size))
        va += idx[:n_val].tolist()
        tr += idx[n_val:].tolist()
    tr, va = rng.permutation(tr), rng.permutation(va)
    for name, ix in ((out_train, tr), (out_valid, va)):
        os.makedirs(os.path.dirname(name) or '.', exist_ok=True)
        with open(name, 'w') as f:
            f.write('\n'.join(rows[i] for i in ix) + '\n')
    print(f'train: {len(tr)} -> {out_train} | valid: {len(va)} -> {out_valid} (seed={seed})')


def fit_scaler(X):
    return X.mean(axis=0), X.std(axis=0) + 1e-12


def standardize(X, mean, std):
    return (X - mean) / std


def load_scaler(data):
    with open(data, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description='Разделение data.csv и расчёт параметров нормировки')
    ap.add_argument('--dataset', default='data.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--valid_size', type=float, default=0.2)
    ap.add_argument('--out_train', default=os.path.join('data', 'train.csv'))
    ap.add_argument('--out_valid', default=os.path.join('data', 'valid.csv'))
    ap.add_argument('--scaler', default=os.path.join('data', 'scaler.pkl'))
    r = ap.parse_args()
    split_dataset(r.dataset, r.seed, r.valid_size, r.out_train, r.out_valid)
    X, _ = load_csv(r.out_train)
    mean, std = fit_scaler(X)
    with open(r.scaler, 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    print(f'> параметры нормировки (по train) сохранены: {r.scaler}')


if __name__ == '__main__':
    main()