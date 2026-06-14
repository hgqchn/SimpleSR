import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simplesr.data import build_dataset, build_dataloader


if __name__ == '__main__':

    opt1 = {
        "name": "test_DOTA",
        "path": "simplesr.data.dataset.PairedImageDataset",
        "kwargs": {
            "dataroot_gt": r"D:\codes\SimpleSR\dataset\DOTA_crop_dataset\dota_samples\train\images",
            "dataroot_lq": r'D:\codes\SimpleSR\dataset\DOTA_crop_dataset\dota_samples\train_lrx4\images',
            "phase": "train",
            "scale": 4,
            "io_backend_opt": {
                "type": 'disk',
            },
        }
    }

    dataset = build_dataset(opt1)

    dataloader_opt = {
        "phase": "train",
        "batch_size_per_gpu": 32,
        "num_worker_per_gpu": 8,
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_mode": None,
    }

    dataloader = build_dataloader(
        dataset=dataset,
        dataset_opt=dataloader_opt,
        num_gpu=1,
        dist=False,
        sampler=None,
    )

    print(f"dataset length: {len(dataset)}")
    print(f"dataloader length: {len(dataloader)}")
    print(f"batch_size: {dataloader.batch_size}")
    print(f"num_workers: {dataloader.num_workers}")

    dataloader_iter = iter(dataloader)
    num_test_batches = 10
    total_time = 0.0
    batch = None
    start_time = time.perf_counter()
    batch = next(dataloader_iter)
    cost_time = time.perf_counter() - start_time
    print(f"{cost_time:.4f} s")
    start_time = time.perf_counter()
    batch = next(dataloader_iter)
    cost_time = time.perf_counter() - start_time
    print(f"{cost_time:.4f} s")

    for i in range(num_test_batches):
        start_time = time.perf_counter()
        batch = next(dataloader_iter)
        cost_time = time.perf_counter() - start_time
        total_time += cost_time
        print(f"batch {i + 1}: {cost_time:.4f} s")

    print(f"avg batch time: {total_time / num_test_batches:.4f} s")

    print("batch keys:", batch.keys())
    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        print(f"{key}: type={type(value)}, shape={shape}")
