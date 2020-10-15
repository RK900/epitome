import tensorflow as tf
import multiprocessing
import epitome.generators as generators
import epitome.functions as functions
from epitome.constants import *
from epitome.models import VLP

import numpy as np
import os

import itertools

def gen():
  for i in itertools.count(10):
    yield (i, [1] * i)


def func2(num, all_data, all_data_regions, model, idx, joined):
    peaks_i = np.zeros((len(all_data_regions)))
    peaks_i[idx] = accessilibility_peak_matrix[num, joined['idx']]
    load_bs = generators.load_data(all_data,
                 model.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                 model.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                 model.matrix,
                 model.assaymap,
                 model.cellmap,
                 radii = model.radii,
                 mode = Dataset.RUNTIME,
                 similarity_matrix = peaks_i,
                 similarity_assays = model.similarity_assays,
                 indices = idx)
    print('done loading')
    input_shapes, output_shape, v = generators.generator_to_tf_dataset(load_bs, model.batch_size, 1, model.prefetch_size)
    print('done generating')
    dataset = tf.data.Dataset.range(4)
    dataset = tf.data.Dataset.from_generator(
     gen,
     (tf.int64, tf.int64),
     (tf.TensorShape([]), tf.TensorShape([None])))
    # dataset = dataset.batch(model.batch_size)
    # dataset = dataset.shuffle(model.shuffle_size)
    # dataset = dataset.repeat()
    # dataset = dataset.prefetch(model.prefetch_size)
    print('before loop')
    for i in v.take(2):
        print('in loop')
    return 2 * num

def func1(num):
    regions_peak_file = os.getcwd() + '/data/test_regions.bed'
    model = VLP(['CEBPB'], test_celltypes=['K562'])

    regions_bed = functions.bed2Pyranges(regions_peak_file)
    all_data_regions = functions.bed2Pyranges(model.regionsFile)
    joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df
    idx = joined['idx_alldata']
    all_data = functions.concatenate_all_data(model.data, model.regionsFile)

    arg = num
    answer = func2(arg, all_data, all_data_regions, model, idx, joined)

    return answer

if __name__ == '__main__':
    accessilibility_peak_matrix = np.random.rand(10, 10)

    p = multiprocessing.Pool()
    print('before call')
    result = p.map(func1, range(accessilibility_peak_matrix.shape[0]))
    print(result)

