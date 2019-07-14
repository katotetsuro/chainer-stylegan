import zipfile
import numpy as np
import chainer
from PIL import Image
import os

def create_submit_data(mapping, gen, smooth):
    z = zipfile.PyZipFile('images_ns.zip' if not smooth else 'images.zip', mode='w')
    xp = gen.xp
    batchsize = 100
    num_output = 10000
    assert num_output % batchsize == 0

    for i in range(num_output // batchsize):
        h = chainer.Variable(xp.asarray(mapping.make_hidden(batchsize)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(h), stage=8)
            x = (x + 1) * 127.5
        x = chainer.cuda.to_cpu(x.array)
        x = xp.clip(x, 0.0, 255.0)
        x = x.transpose(0, 2, 3, 1).astype(np.uint8)
        for j in range(batchsize):
            k = i * batchsize + j
            f = str(k)+'.png'
            Image.fromarray(x[j]).save(f,'PNG')
            z.write(f)
            os.remove(f)
    z.close()
    print('output to zipfile completed')