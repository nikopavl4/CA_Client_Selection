import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'federated.py')

print("Starting Running CA Selection Experiments")

simple_selectors = ['random', 'cellular']
seeds = [97547, 74184, 21094, 96107, 58890, 45103, 43181, 14704, 67601, 26513, 64972, 56930, 35609, 48968, 49530, 55602, 34754, 97245, 13277, 13575]
datasets = ['MNIST','CIFAR10']

for dataset in datasets:
    print(f'{dataset} Dataset')
    for selector in simple_selectors:
            for seed in seeds:
                #print('--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}')
                sub.call(["python" , filename, '--dataset', f'{dataset}', '--seed', f'{seed}', '--selector', f'{selector}'])