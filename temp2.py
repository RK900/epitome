import ray
from ray import serve
import tensorflow as tf

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

class A:
    def __init__(self):
        model = VLP(['CEBPB'], test_celltypes=['K562'])
        self.model = model
        self.accessilibility_peak_matrix = np.random.rand(4, 10)
    
    def func2(self, num, all_data, all_data_regions, idx, joined):
        peaks_i = np.zeros((len(all_data_regions)))
        peaks_i[idx] = self.accessilibility_peak_matrix[0, joined['idx']]
        load_bs = generators.load_data(all_data,
                    self.model.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                    self.model.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                    self.model.matrix,
                    self.model.assaymap,
                    self.model.cellmap,
                    radii = self.model.radii,
                    mode = Dataset.RUNTIME,
                    similarity_matrix = peaks_i,
                    similarity_assays = self.model.similarity_assays,
                    indices = idx)
        print('done loading')
        input_shapes, output_shape, v = generators.generator_to_tf_dataset(load_bs, self.model.batch_size, 1, self.model.prefetch_size)
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
        for f in v.take(2):
            print('in loop')
        return 2 * num

    @serve.accept_batch
    def __call__(self, requests):
        for req in requests:
            regions_peak_file = os.getcwd() + '/data/test_regions.bed'
            regions_bed = functions.bed2Pyranges(regions_peak_file)
            all_data_regions = functions.bed2Pyranges(self.model.regionsFile)
            joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df
            idx = joined['idx_alldata']
            all_data = functions.concatenate_all_data(self.model.data, self.model.regionsFile)
            print(req.data) # test if method is entered
            arg = 42
            answer = self.func2(arg, all_data, all_data_regions, idx, joined)
            return [42]

            
        
        # do stuff, serve model

if __name__ == '__main__':
    client = serve.start()
    client.create_backend("tf", A,
        # configure resources
        ray_actor_options={"num_cpus": 2},
        # configure replicas
        config={
            "num_replicas": 2, 
            "max_batch_size": 24,
            "batch_wait_timeout": 0.1
        }
    )
    client.create_endpoint("tf", backend="tf")
    handle = client.get_handle("tf")

    args = [1,2]

    # futures = [handle.remote(i) for i in args]
    futures = []
    for i in args:
        j = i + 1
        futures.append(handle.remote(j))
    
    result = ray.get(futures)
    print(result)