import tensorflow as tf
import multiprocessing
import epitome.generators as generators
import epitome.functions as functions
from epitome.constants import *
from epitome.models import VLP

import numpy as np
import os

import itertools

from ray import serve
import ray
# serve.init()

def gen():
    for i in itertools.count(10):
        yield (i, [1] * i)

class A:
    def __init__(self):
        model = VLP(['CEBPB'], test_celltypes=['K562'])
        self.model = model

    
    def func2(self, num, all_data, all_data_regions, model, idx, joined, accessilibility_peak_matrix):
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

    @serve.accept_batch
    def __call__(self, *, input_data=None):
        num = input_data
        accessilibility_peak_matrix =  np.random.rand(4, 10)
        regions_peak_file = os.getcwd() + '/data/test_regions.bed'
        model = VLP(['CEBPB'], test_celltypes=['K562'])

        regions_bed = functions.bed2Pyranges(regions_peak_file)
        all_data_regions = functions.bed2Pyranges(model.regionsFile)
        joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df
        idx = joined['idx_alldata']
        all_data = functions.concatenate_all_data(model.data, model.regionsFile)

        arg = num
        answer = self.func2(arg, all_data, all_data_regions, model, idx, joined, accessilibility_peak_matrix)

        return answer
    
    def score_matrix(self):
        # eeeeeeeeeeeee
        accessilibility_peak_matrix = np.random.rand(4, 10)
        processes = []
        for i in range(accessilibility_peak_matrix.shape[0]): # accessibility matrix shape[0]
            p = multiprocessing.Process(target=A.func1, args=(self, i, ))
            p.start()
        # print('before call')
        # result = p.map(A.func1, range(accessilibility_peak_matrix.shape[0]))
        for p in processes:
            p.join()
    
    def meme(self, *args, something_else=None):
        print(something_else)
        return args


if __name__ == '__main__':
    a = A()
    asd = [('ye', 1), ('helo', 2), ('ja', 3)]
    results = [a.meme(i) for i in asd]

    # accessilibility_peak_matrix = np.random.rand(4, 10)
    # serve.create_backend("tf", A,
    #     # configure resources
    #     ray_actor_options={"num_cpus": 2},
    #     # configure replicas
    #     config={
    #         "num_replicas": 2, 
    #         "max_batch_size": 24,
    #         "batch_wait_timeout": 0.1
    #     }
    # )
    # serve.create_endpoint("tf", backend="tf")
    # handle = serve.get_handle("tf")
    # args = [[1, accessilibility_peak_matrix], 
    #         [2, accessilibility_peak_matrix], 
    #         [3, accessilibility_peak_matrix]
    #     ]
    # args = [1, 
    #         2, 
    #         3
    #     ]
    # futures = [handle.remote(input_data=i) for i in args]
    # result = ray.get(futures)



