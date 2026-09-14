from glob import glob
import re
import random
import numpy 
import torch


def set_seed(seed, cudnn_deterministic=True):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def get_version_num(filename, extension="feather"):
    if extension[0] != ".":
        extension = "." + extension

    pattern = f"{filename}_v*{extension}"
    existing_files = glob(pattern)

    version_regex = re.compile(r"_v(\d+)" + re.escape(extension) + r"$")

    max_version = -1

    for f in existing_files:
        match = version_regex.search(f)
        if match:
            v = int(match.group(1))
            max_version = max(max_version, v)

    new_version = max_version + 1
    return new_version
