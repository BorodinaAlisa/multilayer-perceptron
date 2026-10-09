import argparse
import os
import numpy as np
from mlp import MLP, ACTIVATIONS, binary_cross_entropy
from split import load_csv, standardize, load_scaler


def accuracy(P, y):
    return float((P.argmax(axis=1) == y).mean())


def fit(net, Xt, yt, Xv, yv, epochs, lr, batch_size, momentum=0.0, patience=0, seed=0, log=print):
    rng = np.random.RandomState(seed)
    Yt, Yv = np.eye(2)[yt], np.eye(2)[yv]
    hist = {k: [] for k in ('loss', 'val_loss', 'acc', 'val_acc')}
    best, best_w, wait = np.inf, None, 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xt))
        for s in range(0, len(Xt), batch_size):
            ix = order[s:s + batch_size]
            net.train_step(Xt[ix], Yt[ix], lr, momentum)
        Pt, Pv = net.predict_proba(Xt), net.predict_proba(Xv)
        hist['loss'].append(binary_cross_entropy(Yt, Pt))
        hist['val_loss'].append(binary_cross_entropy(Yv, Pv))
        hist['acc'].append(accuracy(Pt, yt))
        hist['val_acc'].append(accuracy(Pv, yv))
        log(f"epoch {ep:03d}/{epochs} | train_loss={hist['loss'][-1]:.4f} val_loss={hist['val_loss'][-1]:.4f}"
            f" | train_acc={hist['acc'][-1]:.4f} val_acc={hist['val_acc'][-1]:.4f}")
        if patience:                                         # early stopping по val_loss
            if hist['val_loss'][-1] < best - 1e-6:
                best, wait, best_w = hist['val_loss'][-1], 0, net.get_weights()
            else:
                wait += 1
                if wait >= patience:
                    log(f'early stopping на эпохе {ep}, возвращаем веса лучшей эпохи')
                    break
    if best_w:
        net.set_weights(best_w)
    return hist


def plot_history(h, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for key, val_key, title, ylabel, name in (('loss', 'val_loss', 'Loss', 'loss', 'loss.png'),
                                              ('acc', 'val_acc', 'Accuracy', 'accuracy', 'accuracy.png')):
        plt.figure()
        plt.plot(h[key], label='train')
        plt.plot(h[val_key], label='val')
        plt.title(title); plt.xlabel('epoch'); plt.ylabel(ylabel)
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, name), dpi=100)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description='Обучение многослойного персептрона')
    ap.add_argument('--train', default=os.path.join('data', 'train.csv'))
    ap.add_argument('--valid', default=os.path.join('data', 'valid.csv'))
    ap.add_argument('--scaler', default=os.path.join('data', 'scaler.pkl'))
    ap.add_argument('--layer', type=int, nargs='+', default=[16, 16], help='размеры скрытых слоёв')
    ap.add_argument('--activation', default='relu', choices=ACTIVATIONS)
    ap.add_argument('--initializer', default='heUniform', choices=['heUniform', 'heNormal', 'xavier'])
    ap.add_argument('--loss', default='binaryCrossentropy', choices=['binaryCrossentropy'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=0.0001)
    ap.add_argument('--momentum', type=float, default=0.0)
    ap.add_argument('--patience', type=int, default=0, help='early stopping (0 = выкл.)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', default=os.path.join('models', 'model.pkl'))
    ap.add_argument('--plots', default=os.path.join('results', 'plots'))
    ap.add_argument('--log', default=os.path.join('results', 'train_log.txt'))
    r = ap.parse_args()

    os.makedirs(os.path.dirname(r.log) or '.', exist_ok=True)
    logfile = open(r.log, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        logfile.write(msg + '\n')

    sc = load_scaler(r.scaler)
    Xt, yt = load_csv(r.train)
    Xv, yv = load_csv(r.valid)
    Xt, Xv = standardize(Xt, sc['mean'], sc['std']), standardize(Xv, sc['mean'], sc['std'])
    log(f'x_train shape : {Xt.shape}\nx_valid shape : {Xv.shape}')

    net = MLP([Xt.shape[1], *r.layer, 2], r.activation, r.initializer, r.seed)
    hist = fit(net, Xt, yt, Xv, yv, r.epochs, r.learning_rate, r.batch_size, r.momentum, r.patience, r.seed, log)
    net.save(r.model, sc['mean'], sc['std'])
    log(f'saved to {r.model}')
    plot_history(hist, r.plots)
    log(f'saved to {r.plots}')
    logfile.close()


if __name__ == '__main__':
    main()