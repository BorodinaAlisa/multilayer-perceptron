import argparse
import os
import numpy as np
from mlp import MLP, binary_cross_entropy
from split import load_csv, standardize


def confusion_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=(tp + tn) / len(y), precision=prec, recall=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    ap = argparse.ArgumentParser(description='Прогноз и оценка модели')
    ap.add_argument('--dataset', default=os.path.join('data', 'valid.csv'))
    ap.add_argument('--model', default=os.path.join('models', 'model.pkl'))
    ap.add_argument('--metrics', default=os.path.join('results', 'metrics.txt'))
    r = ap.parse_args()

    net, mean, std = MLP.load(r.model)
    X, y = load_csv(r.dataset)
    P = net.predict_proba(standardize(X, mean, std))
    m = confusion_metrics(y, P.argmax(axis=1))
    lines = [f'BCE loss: {binary_cross_entropy(np.eye(2)[y], P):.4f}',
             f"accuracy: {m['acc']:.4f} | precision: {m['precision']:.4f} | recall: {m['recall']:.4f} | F1: {m['f1']:.4f}",
             f"confusion (M=positive): TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}"]
    print('\n'.join(lines))
    os.makedirs(os.path.dirname(r.metrics) or '.', exist_ok=True)
    with open(r.metrics, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()