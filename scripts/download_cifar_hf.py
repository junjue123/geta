#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 hf-mirror.com 下载 CIFAR-10 parquet 并转成 torchvision 兼容格式"""
import os, sys, pickle, time
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "datasets/cifar10"
BATCHES_DIR = os.path.join(OUT_DIR, "cifar-10-batches-py")

LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def download_parquets():
    print("从 hf-mirror.com 下载 cifar10 parquet ...", flush=True)
    t0 = time.time()
    train_path = hf_hub_download(
        repo_id="uoft-cs/cifar10", repo_type="dataset",
        filename="plain_text/train-00000-of-00001.parquet",
        local_dir=os.path.join(OUT_DIR, "_parquet"))
    print(f"  train: {train_path} ({os.path.getsize(train_path)/1e6:.1f}MB)", flush=True)
    test_path = hf_hub_download(
        repo_id="uoft-cs/cifar10", repo_type="dataset",
        filename="plain_text/test-00000-of-00001.parquet",
        local_dir=os.path.join(OUT_DIR, "_parquet"))
    print(f"  test:  {test_path} ({os.path.getsize(test_path)/1e6:.1f}MB)", flush=True)
    print(f"  下载耗时 {time.time()-t0:.1f}s", flush=True)
    return train_path, test_path


def parquet_to_batches(train_path, test_path):
    import pandas as pd
    os.makedirs(BATCHES_DIR, exist_ok=True)

    print("读取 parquet 并转换 ...", flush=True)
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    print(f"  train rows: {len(df_train)}, cols: {list(df_train.columns)}", flush=True)
    print(f"  test rows:  {len(df_test)}", flush=True)

    # img 列: dict {'bytes': ..., 'path': None} 或 PIL Image
    def img_to_array(img_val):
        if isinstance(img_val, dict) and 'bytes' in img_val:
            from io import BytesIO
            from PIL import Image
            im = Image.open(BytesIO(img_val['bytes']))
            return np.array(im)  # (32,32,3) uint8
        elif isinstance(img_val, (bytes, bytearray)):
            from io import BytesIO
            from PIL import Image
            im = Image.open(BytesIO(img_val))
            return np.array(im)
        else:
            # 可能是 dict with path
            raise ValueError(f"未知 img 格式: {type(img_val)}")

    # 训练集分 5 个 batch (每个 10000)
    n_train = len(df_train)
    per_batch = n_train // 5
    for i in range(5):
        start = i * per_batch
        end = (i + 1) * per_batch if i < 4 else n_train
        rows = df_train.iloc[start:end]
        data = np.zeros((len(rows), 3 * 32 * 32), dtype=np.uint8)
        labels = []
        for j, (_, row) in enumerate(rows.iterrows()):
            arr = img_to_array(row['img'])  # (32,32,3) HWC
            # CIFAR 原始格式: (3072,) RGB flattened, 通道优先 (R,G,B 各 1024)
            arr_flat = arr.transpose(2, 0, 1).reshape(-1)  # (3*32*32,) CHW
            data[j] = arr_flat
            labels.append(int(row['label']))
        batch = {'data': data, 'labels': labels}
        path = os.path.join(BATCHES_DIR, f"data_batch_{i+1}")
        with open(path, 'wb') as f:
            pickle.dump(batch, f)
        print(f"  写入 {path}: {len(rows)} 样本", flush=True)

    # 测试集 1 个 batch
    data = np.zeros((len(df_test), 3 * 32 * 32), dtype=np.uint8)
    labels = []
    for j, (_, row) in enumerate(df_test.iterrows()):
        arr = img_to_array(row['img'])
        data[j] = arr.transpose(2, 0, 1).reshape(-1)
        labels.append(int(row['label']))
    with open(os.path.join(BATCHES_DIR, "test_batch"), 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"  写入 test_batch: {len(df_test)} 样本", flush=True)

    # meta
    meta = {'num_cases_per_batch': 10000, 'label_names': LABEL_NAMES,
            'num_vis': 3 * 32 * 32}
    with open(os.path.join(BATCHES_DIR, "batches.meta"), 'wb') as f:
        pickle.dump(meta, f)
    print(f"  写入 batches.meta", flush=True)


def main():
    train_path, test_path = download_parquets()
    parquet_to_batches(train_path, test_path)
    # 校验: 用 torchvision 加载
    import torchvision
    import torchvision.transforms as T
    ds = torchvision.datasets.CIFAR10(root=OUT_DIR, train=True, download=False,
                                      transform=T.ToTensor())
    print(f"\n[校验] torchvision 加载成功: {len(ds)} 训练样本", flush=True)
    img, lbl = ds[0]
    print(f"  样本形状: {img.shape}, 标签: {lbl} ({LABEL_NAMES[lbl]})", flush=True)
    ds_test = torchvision.datasets.CIFAR10(root=OUT_DIR, train=False, download=False,
                                           transform=T.ToTensor())
    print(f"  测试样本: {len(ds_test)}", flush=True)
    print("\n[完成] CIFAR-10 数据已就绪 (torchvision 兼容格式)", flush=True)


if __name__ == "__main__":
    main()
