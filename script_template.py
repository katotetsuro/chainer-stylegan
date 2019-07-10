import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path('/home').joinpath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/home && ' + command)


run('cd /home && python setup.py develop --install-dir /home')
run('python -m src.stylegan.train --gpu 0 --image_dir /kaggle/input/all-dogs/all-dogs --out result --ch 256 --stage_interval 100000 --keep_smoothed_gen')