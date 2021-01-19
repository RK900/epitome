import ray
from ray import serve
import tensorflow as tf
import requests


class A:
    def __init__(self):
        self.model = tf.constant(1.0) # dummy example
        self.num = 1

    @serve.accept_batch
    def __call__(self, request):
        # for req in requests:
        #     return [req.data] # test if method is entered
        self.num += 1
        return {"count": self.num}
        
        # do stuff, serve model

if __name__ == '__main__':
    client = serve.start()
    client.create_backend("tf", A,
        # configure resources
        # ray_actor_options={"num_cpus": 2},
        # configure replicas
        # config={
        #     "num_replicas": 2, 
        #     "max_batch_size": 24,
        #     "batch_wait_timeout": 0.1
        # }
    )
    client.create_endpoint("my_endpoint", backend="tf", route="/counter")
    handle = client.get_handle("my_endpoint")

    args = [i for i in range(10)]

    futures = [handle.remote(i) for i in args]
    print(len(futures))
    result = ray.get([futures])
