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

class A:
    def __init__(self):
        model = VLP(['CEBPB'], test_celltypes=['K562'])
        self.model = model
        self.accessilibility_peak_matrix = np.random.rand(4, 10)

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

    futures = [handle.remote(i) for i in args]
    result = ray.get(futures)
    print(result)