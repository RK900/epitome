import ray
from ray import serve
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from epitome import *
import epitome.functions as functions
from epitome.constants import *
import epitome.generators as generators
from epitome.models import VLP

import os

class ScoreMatrix_Runner(VLP):
    def __init__(self):
        VLP.__init__(self, assays=['CEBPB'], test_celltypes=['K562'])

    @serve.accept_batch
    def __call__(self, requests):
        for req in requests:
            data = req.data[0]
            matrix = req.data[1]
            indices = req.data[2]
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

            results = self.run_predictions(num_samples, ds, calculate_metrics = False)

            return [results['preds_mean']]
    
    def score_matrix(self, accessilibility_peak_matrix, regions_peak_file, regions_indices = None, all_data = None):
        client = serve.start()
        client.create_backend("tf", ScoreMatrix_Runner,
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

        if all_data is None:
            all_data = functions.concatenate_all_data(self.data, self.regionsFile)

        regions_bed = functions.bed2Pyranges(regions_peak_file)
        all_data_regions = functions.bed2Pyranges(self.regionsFile)

        joined = regions_bed.join(all_data_regions, how='left',suffix='_alldata').df

        # select regions with data to score
        if regions_indices is not None:
            joined = joined[joined['idx'].isin(regions_indices)]
            joined = joined.reset_index()
        
        idx = joined['idx_alldata']

        futures = []
        for sample_i in tqdm(range(accessilibility_peak_matrix.shape[0])):
            peaks_i = np.zeros((len(all_data_regions)))
            peaks_i[idx] = accessilibility_peak_matrix[sample_i, joined['idx']]
            futures.append(handle.remote((all_data, peaks_i, idx)))
        
        results = ray.get(futures)
        # return result
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


if __name__ == '__main__':
    apm = np.random.rand(4, 10)
    rpf = os.getcwd() + '/runners/test_regions.bed'
    a = ScoreMatrix_Runner()
    par_results = a.score_matrix(accessilibility_peak_matrix=apm, regions_peak_file=rpf)
    print(par_results)

    serial = VLP(assays=['CEBPB'], test_celltypes=['K562'])
    ser_results = serial.score_matrix(apm, rpf)
    print(ser_results)
    sr2 = serial.score_matrix(apm, rpf)
    print(par_results == ser_results)
    print(ser_results == sr2)
