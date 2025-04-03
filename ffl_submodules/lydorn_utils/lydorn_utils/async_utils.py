import multiprocessing
import json
import time


def async_func_wrapper(async_func, out_queue):
    while True:
        if not out_queue.empty():
            data = out_queue.get()
            if data is not None:
                async_func(data)
            else:
                break


class AsyncMultiprocess(object):
    def __init__(self, async_func, num_workers=1):
        """

        :param async_func: Takes a queue as input. Listens to the queue and perform operations
        on queue elements as long as elements are not None. Stops at the first None element encountered.
        :param num_workers:
        """
        assert 0 < num_workers, "num_workers should be at least 1."
        self.num_workers = num_workers
        self.queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
        self.subprocesses = [
            multiprocessing.Process(target=async_func_wrapper, args=(async_func, self.queues[i])) for i in range(self.num_workers)
        ]

        self.current_process = 0

    def start(self):
        for subprocess in self.subprocesses:
            subprocess.start()

    def add_work(self, data):
        # TODO: add work to the least busy process (shortest queue)
        self.queues[self.current_process].put(data)
        self.current_process = (self.current_process + 1) % self.num_workers

    def join(self):
        for subprocess, queue in zip(self.subprocesses, self.queues):
            queue.put(None)
            subprocess.join()


class Async(object):
    def __init__(self, async_func):
        """

        :param async_func: Takes a queue as input. Listens to the queue and perform operations
        on queue elements as long as elements are not None. Stops at the first None element encountered.
        """
        self.queue = multiprocessing.Queue()
        self.subprocess = multiprocessing.Process(target=async_func_wrapper, args=(async_func, self.queue))

    def start(self):
        self.subprocess.start()

    def add_work(self, data):
        self.queue.put(data)

    def join(self):
        self.queue.put(None)
        self.subprocess.join()


def main():
    def process(data):
        print("--- process() ---")
        data["out_numbers"] = [number * number for number in data["in_numbers"]]
        time.sleep(0.5)
        return data

    def save(data):
        print("--- save() ---")
        time.sleep(1)
        print("Finished saving")

    num_workers = 8
    data_list = [
        {
            "filepath": "out/data.{}.json".format(i),
            "in_numbers": list(range(1000))

        } for i in range(5)
    ]

    # AsyncMultiprocess
    print("AsyncMultiprocess")
    saver_async_multiprocess = AsyncMultiprocess(save, num_workers)
    saver_async_multiprocess.start()

    t0 = time.time()
    for data in data_list:
        data = process(data)
        saver_async_multiprocess.add_work(data)
    saver_async_multiprocess.join()
    print("Done in {}s".format(time.time() - t0))

    print("Finished all!")


if __name__ == "__main__":
    main()
