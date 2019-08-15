import gzip
import base64
import os
from pathlib import Path
from typing import Dict

# for kaggle team
# this is base64-encoded source code.
# code will be decoded and install at the beginning of kernel run.
# you can see all code at https://github.com/katotetsuro/chainer-stylegan/tree/ac


# this is base64 encoded source code
file_data: Dict = {file_data}

# decode to local storage
for path, encoded in file_data.items():
    print(path)
    path = Path('/home').joinpath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/home && ' + command)


run('cd /home && python setup.py develop --install-dir /home')
print('cropping images start.')
run('python -m src.dataset.trim_images /kaggle/input/all-dogs/all-dogs /kaggle/input/annotation/Annotation /home/images')
print('cropping images finished.')
run('python -m src.stylegan.train --gpu 0 --image_dir /home/images/data.npy --out result --ch 256 --mapping-ch 320 --stage_interval 500000 --keep_smoothed_gen --ac 0.05')