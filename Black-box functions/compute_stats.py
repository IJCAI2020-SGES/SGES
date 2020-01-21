import numpy as np
import pandas as pd
import collections
import os

Result = collections.namedtuple('Result', 'name progress')


def load_results(root_dir, verbose=True):
    all_results = []
    for func_name in os.listdir(root_dir):
        if func_name.startswith('.'):
            continue
        for dirname in os.listdir(os.path.join(root_dir, func_name)):
            if dirname.endswith('.csv'):
                name = '%s-%s' % (func_name, dirname)
                progress = pd.read_csv(os.path.join(root_dir, func_name, dirname))
                result = Result(name=name, progress=progress)
                all_results.append(result)
                print('load %s ' % name)
    print('load %d results' % len(all_results))
    return all_results


def main(root_dir):
    all_results = load_results(root_dir)

    def split_fn(r):
        name = r.name
        splits = name.split('-')
        return splits[0]

    def group_fn(r):
        name = r.name
        splits = name.split('-')
        alg_name = splits[1]
        if alg_name == 'SGES':
            return 'SGES'
        elif alg_name == 'CMA':
            return 'CMA-ES'
        elif alg_name == 'GES':
            return 'Guided ES'
        elif alg_name == 'ES':
            return 'Vanilla ES'
        elif alg_name == 'ASEBO':
            return 'ASEBO'
        else:
            raise ValueError('%s not supported' % alg_name)

    dict_result = dict()
    for result in all_results:
        env_name = split_fn(result)
        if env_name not in dict_result:
            dict_result[env_name] = dict()
        alg_name = group_fn(result)
        if alg_name not in dict_result[env_name]:
            dict_result[env_name][alg_name] = [[], []]
        dict_result[env_name][alg_name][0].append(list(result.progress['ys'])[-1])
        dict_result[env_name][alg_name][1].append(np.sum(result.progress['ts']))

    for env_name in dict_result:
        print('**********%s************' % env_name)
        for alg_name in dict_result[env_name]:
            print('%s: mean:%.4f std:%.4f time:%.4f' % (
                alg_name,
                np.mean(dict_result[env_name][alg_name][0]),
                np.std(dict_result[env_name][alg_name][0]),
                np.mean(dict_result[env_name][alg_name][1])
            ))
    

if __name__ == '__main__':
    main(root_dir='logs')