import ray
from ray import serve
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import epitome.generators as generators
import epitome.functions as functions
from epitome.constants import *
from epitome.models import VLP
from epitome import *

import os

import itertools
import psutil
import gc

ray.init(num_cpus=24)

def auto_garbage_collect(pct=50.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

def gen():
    for i in itertools.count(10):
        yield (i, [1] * i)

@ray.remote
class A(VLP):
    def __init__(self, accessilibility_peak_matrix, regions_peak_file):
        VLP.__init__(self, assays=['CEBPB', "JUN", 'TCF7', 'CEBPZ'], test_celltypes=['K562'])
        # self.model = model
        self.accessilibility_peak_matrix = accessilibility_peak_matrix
        self.regions_peak_file = regions_peak_file
        self.all_data = functions.concatenate_all_data(self.data, self.regionsFile)

    
    def func2(self, data, matrix, indices, samples = 50):
        # peaks_i = np.zeros((len(all_data_regions)))
        # peaks_i[idx] = self.accessilibility_peak_matrix[0, joined['idx']]
        load_bs = generators.load_data(data,
                    self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                    self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                    self.matrix,
                    self.assaymap,
                    self.cellmap,
                    radii = self.radii,
                    mode = Dataset.RUNTIME,
                    similarity_matrix = matrix,
                    similarity_assays = self.similarity_assays,
                    indices = indices)
        num_samples = len(indices)
        print('done loading')
        input_shapes, output_shape, v = generators.generator_to_tf_dataset(load_bs, self.batch_size, 1, self.prefetch_size)
        print('done generating')
        dataset = tf.data.Dataset.range(4)
        dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.int64, tf.int64),
        (tf.TensorShape([]), tf.TensorShape([None])))
        print('before loop')
        for f in v.take(2):
            print('in loop')
        return 2

    # @serve.accept_batch
    def eval_vector(self, req):
        data = self.all_data
        matrix = req[0]
        indices = req[1]
        input_shapes, output_shape, ds = generators.generator_to_tf_dataset(generators.load_data(data,
                self.test_celltypes,   # used for labels. Should be all for train/eval and subset for test
                self.eval_cell_types,   # used for rotating features. Should be all - test for train/eval
                self.matrix,
                self.assaymap,
                self.cellmap,
                radii = self.radii,
                mode = Dataset.RUNTIME,
                similarity_matrix = matrix,
                similarity_assays = self.similarity_assays,
                indices = indices), self.batch_size, 1, self.prefetch_size)

        num_samples = len(indices)
        auto_garbage_collect()
        results = self.run_predictions(num_samples, ds, calculate_metrics = False)
        auto_garbage_collect()

        return [results['preds_mean']]
        # return [69]

    # @serve.accept_batch
    def old_call(self, requests):
        all_data_regions = functions.bed2Pyranges(self.regionsFile)
        for req in requests:
            regions_peak_file = req.data[1]
            regions_bed = functions.bed2Pyranges(regions_peak_file)
            
            joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df
            idx = joined['idx_alldata']
            all_data = functions.concatenate_all_data(self.data, self.regionsFile)
            # print(req.data[1]) # test if method is entered
            arg = 42
            answer = self.func2(arg, all_data, all_data_regions, idx, joined)
            return [42]

            
        
        # do stuff, serve model
    
    def score_matrix(self, regions_indices = None, all_data = None):

        if all_data is None:
            all_data = functions.concatenate_all_data(self.data, self.regionsFile)

        regions_bed = functions.bed2Pyranges(self.regions_peak_file)
        all_data_regions = functions.bed2Pyranges(self.regionsFile)

        joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df

        # select regions with data to score
        if regions_indices is not None:
            joined = joined[joined['idx'].isin(regions_indices)]
            joined = joined.reset_index()
        
        idx = joined['idx_alldata']

        # args = [(1, self.regions_peak_file), (2, self.regions_peak_file)]

        # # futures = [handle.remote(i) for i in args]
        futures = []
        results = []
        for sample_i in tqdm(range(self.accessilibility_peak_matrix.shape[0])):
            peaks_i = np.zeros((len(all_data_regions)))
            peaks_i[idx] = self.accessilibility_peak_matrix[sample_i, joined['idx']]
            value = handle.remote((peaks_i, idx))
            futures += [value]
            results.append(ray.get(futures))
            auto_garbage_collect()
        
        # results = ray.get(futures)
        # return result
        print(results)
        print(results.shape)
        tmp = np.stack(results)

        # get the index break for each region_bed region
        reduce_indices = joined.drop_duplicates('idx',keep='first').index.values

        # get the number of times there was a scored region for each region_bed region
        # used to calculate reduced means
        indices_counts = joined['idx'].value_counts(sort=False).values[:,None]

        # reduce means on middle axis
        final = np.add.reduceat(tmp, reduce_indices, axis = 1)/indices_counts

        # TODO 9/10/2020: code is currently scoring missing values and setting to nan.
        # You could be more efficient and remove these so you are not taking
        # the time to score garbage data.

        # fill missing indices with nans
        missing_indices = joined[joined['idx_alldata']==-1]['idx'].values
        final[:,missing_indices, :] = np.NAN

        return final

    
    def test(self):
        print(self.eval_vector)
        print(self.regionsFile)

if __name__ == '__main__':
    apm = np.random.rand(21, 100_000)
    rpf = os.getcwd() + '/data/test_regions.bed'
    a = A.remote(accessilibility_peak_matrix=apm, regions_peak_file=rpf)
    metadata_class = VLP( assays=['CEBPB', "JUN", 'TCF7', 'CEBPZ'])
    regionsFile = metadata_class.regionsFile
    all_data = functions.concatenate_all_data(metadata_class.data, metadata_class.regionsFile)
    # a.test.remote()
    # t = ray.get(a.predict_step.remote((1, [1, 2, 3])))

    regions_bed = functions.bed2Pyranges(rpf)
    all_data_regions = functions.bed2Pyranges(regionsFile)

    joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df

    # select regions with data to score
    # if regions_indices is not None:
    #     joined = joined[joined['idx'].isin(regions_indices)]
    #     joined = joined.reset_index()
    
    idx = joined['idx_alldata']

    # args = [(1, self.regions_peak_file), (2, self.regions_peak_file)]

    # # futures = [handle.remote(i) for i in args]
    futures = []
    results = []
    num_classes = 1
    classes = [A.remote(accessilibility_peak_matrix=apm, regions_peak_file=rpf) for i in range(num_classes)]

    for sample_i in tqdm(range(apm.shape[0])):
        peaks_i = np.zeros((len(all_data_regions)))
        peaks_i[idx] = apm[sample_i, joined['idx']]
        value = classes[sample_i % num_classes].eval_vector.remote((peaks_i, idx))
        futures += [value]
    
    results = ray.get(futures)
    # return result
    # print(results)
    # print(results[0].shape)
    results = np.array(results)
    print(results.shape)
    results = results[:, 0, :, :]
    tmp = np.stack(results)

    # get the index break for each region_bed region
    reduce_indices = joined.drop_duplicates('idx',keep='first').index.values

    # get the number of times there was a scored region for each region_bed region
    # used to calculate reduced means
    indices_counts = joined['idx'].value_counts(sort=False).values[:,None]

    # reduce means on middle axis
    final = np.add.reduceat(tmp, reduce_indices, axis = 1)/indices_counts

    # TODO 9/10/2020: code is currently scoring missing values and setting to nan.
    # You could be more efficient and remove these so you are not taking
    # the time to score garbage data.

    # fill missing indices with nans
    missing_indices = joined[joined['idx_alldata']==-1]['idx'].values
    final[:,missing_indices, :] = np.NAN

    print(final)
    print(final.shape)
    print('DONE')