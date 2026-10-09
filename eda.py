import numpy as np
from split import load_csv

base = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dim']
names = [f'{p}_{b}' for p in ('mean', 'se', 'worst') for b in base]

X, y = load_csv('data.csv')

print('Размер данных:', X.shape)
print('Доброкачественных (B):', (y == 0).sum(), '| Злокачественных (M):', (y == 1).sum())
print('Пропусков:', int(np.isnan(X).sum()))

print('\nСреднее по классам:')
print(f"{'признак':25s} {'B':>10s} {'M':>10s}")
for name in ['mean_radius', 'mean_area', 'mean_concave_points', 'worst_perimeter']:
    i = names.index(name)
    print(f'{name:25s} {X[y == 0, i].mean():10.3f} {X[y == 1, i].mean():10.3f}')

corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(30)])
order = np.argsort(corr)[::-1]
print('\nТоп-5 признаков по корреляции с диагнозом:')
for i in order[:5]:
    print(f'  {names[i]:25s} {corr[i]:.3f}')

C = np.corrcoef(X.T)
print(f'\nПар признаков с |корреляцией| > 0.9: {(np.abs(np.triu(C, 1)) > 0.9).sum()}')