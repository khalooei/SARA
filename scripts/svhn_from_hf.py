"""Build the SVHN tensor cache from the authors' copy on the Hugging Face hub (ufldl-stanford/svhn).

Usage: python scripts/svhn_from_hf.py PARQUET_DIR DATA_DIR
Writes DATA_DIR/svhn_{train,test}.pt in the format read by sara.data._tensor_cache."""
import io
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
EXPECTED = {"train": 73257, "test": 26032}
for split in ["train", "test"]:
    t = pq.read_table(f"{src}/{split}-00000-of-00001.parquet").to_pydict()
    imgs = [np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["image"]]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(t["label"], dtype=torch.long)
    assert x.shape == (EXPECTED[split], 3, 32, 32), x.shape
    assert x.dtype == torch.uint8 and int(y.min()) == 0 and int(y.max()) == 9
    torch.save({"x": x, "y": y}, f"{dst}/svhn_{split}.pt")
    print(split, tuple(x.shape), "label counts", torch.bincount(y).tolist())
