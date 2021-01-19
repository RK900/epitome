import ray
from ray import serve
import requests

ray.init(num_cpus=4)
client = serve.start(detached=True)


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, request):
        self.count += 1
        return request.data[0]


# Form a backend from our class and connect it to an endpoint.
client.create_backend("my_backend", Counter)
client.create_endpoint("my_endpoint", backend="my_backend", route="/counter")

# Query our endpoint in two different ways: from HTTP and from Python.
results = ray.get([client.get_handle("my_endpoint").remote((12,)) for i in range(10)])
print(results)
print("done")
# > {"count": 2}
