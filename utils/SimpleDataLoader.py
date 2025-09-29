import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def seed_worker(worker_id: int, base_seed: int = 42):
    """
    设置每个 worker 的随机种子，确保多线程可复现
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=None, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn is not None else lambda x: x
        self.seed = seed

    def __iter__(self):
        # 固定主线程的随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # 单线程模式
        if self.num_workers <= 0:
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch = [self.dataset[i] for i in batch_indices]
                yield self.collate_fn(batch)
        else:
            # 多线程模式
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for start in range(0, len(indices), self.batch_size):
                    batch_indices = indices[start:start + self.batch_size]

                    # 每个任务传入独立 worker_id
                    futures = []
                    for wid, idx in enumerate(batch_indices):
                        def fetch(i=idx, worker_id=wid):
                            seed_worker(worker_id=worker_id, base_seed=self.seed)
                            return self.dataset[i]

                        futures.append(executor.submit(fetch))

                    batch = [f.result() for f in futures]
                    yield self.collate_fn(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
