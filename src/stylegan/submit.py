import zipfile
import numpy as np
import chainer
from PIL import Image
import os

def create_submit_data(mapping, gen, smooth):
    xp = gen.xp
    batchsize = 100
    num_output = 10000
    assert num_output % batchsize == 0

    # w_batch_size = 100
    # n_avg_w = 20000

    # n_batches = n_avg_w // w_batch_size
    # w_avg = xp.zeros(256).astype('f')
    # with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    #     for i in range(n_batches):
    #         z = chainer.Variable(xp.asarray(mapping.make_hidden(w_batch_size)))
    #         w_cur = mapping(z)
    #         w_avg = w_avg + xp.average(w_cur.data, axis=0)
    # w_avg = w_avg / n_batches

    # trc_psi = 0.0
    z = zipfile.PyZipFile('images_ns.zip' if not smooth else 'images.zip', mode='w')
    for i in range(num_output // batchsize):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            w = chainer.Variable(xp.asarray(mapping.make_hidden(batchsize)))
            w = mapping(w).array
            #delta = w - w_avg
            #w = delta * trc_psi + w_avg
            x = gen(mapping(w), stage=8)
            x = (x + 1) * 127.5
        x = chainer.cuda.to_cpu(x.array)
        x = xp.clip(x, 0.0, 255.0)
        x = x.transpose(0, 2, 3, 1).astype(np.uint8)
        for j in range(batchsize):
            k = i * batchsize + j
            f = str(k)+'.png'
            Image.fromarray(x[j]).resize((64, 64)).save(f,'PNG')
            z.write(f)
            os.remove(f)
    z.close()
    print('output to zipfile completed')