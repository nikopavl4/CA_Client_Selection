import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'federated.py')

print("Starting Running CA Selection Experiments")

simple_selectors = ['random', 'cellular']
seeds = [0, 12345, 123, 2023, 4041, 97547, 74184, 21094, 96107, 58890]
datasets = ['MNIST','CIFAR10']

for dataset in datasets:
    print(f'{dataset} Dataset')
    for selector in simple_selectors:
            for seed in seeds:
                #print('--dataset', f'{dataset}', '--model', f'{model}', '--seed', f'{seed}', '--featurizer', f'{feat}')
                sub.call(["python" , filename, '--dataset', f'{dataset}', '--seed', f'{seed}', '--selector', f'{selector}'])