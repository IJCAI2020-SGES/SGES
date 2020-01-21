# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np
import os
import csv


def output_data(filename, data):
    file = open(filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(file, delimiter='\n', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(data)
    file.close()


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64), np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def get_output_folder(parent_dir, env_name, seed):
    os.makedirs(parent_dir, exist_ok=True)
    parent_dir = os.path.join(parent_dir, env_name, 'seed' + str(seed) + '/')
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir
