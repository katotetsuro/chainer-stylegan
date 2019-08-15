#!/usr/bin/env python3

import os
import sys
import re
import json
from pathlib import Path

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
from chainer import training
from chainer.training import extension
from chainer.training import extensions
import chainercv

try:
    import chainermn
    from mpi4py import MPI
    mpi_is_master = False
    mpi_available = True
except:  #pylint:disable=bare-except
    mpi_is_master = True 
    mpi_available = False

# from dataset import HDF5Dataset
from .updater import StageManager, Updater
from .net import Discriminator, StyleGenerator, MappingNetwork 
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

import common
from common.utils.record import record_setting
from common.datasets.base.base_dataset import BaseDataset
from .config import FLAGS
from common.utils.save_images import convert_batch_images
from common.evaluation.fid import API as FIDAPI, fid_extension
from .submit import create_submit_data
from dataset.trim_images import load_dataset

class Transform():
    def __init__(self, size):
        self.size = size

    def __call__(self, in_data):
        x, t = in_data
        t = t.astype(np.int32)
        x = chainercv.transforms.resize(x, (self.size, self.size))
        x = chainercv.transforms.random_flip(x, False, True)
        x = x / 127.5 - 1
        return x, t

def sample_generate_light(gen, mapping, dst, rows=8, cols=8, z=None, seed=0, subdir='preview'):
    @chainer.training.make_extension()
    def make_image(trainer):
        nonlocal rows, cols, z
        # if trainer.updater.stage < 8:
        #     print('skip visualization')
        #     return
        # else:
        #     print('visualize.')
        if trainer.updater.stage > 15:
            rows = min(rows, 2)
            cols = min(cols, 2)
        elif trainer.updater.stage > 13:
            rows = min(rows, 3)
            cols = min(cols, 3)
        elif trainer.updater.stage > 11:
            rows = min(rows, 4)
            cols = min(cols, 4)

        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        if z is None:
            z = Variable(xp.asarray(mapping.make_hidden(n_images)[0]))
        else:
            z = z[:n_images]
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = convert_batch_images(x, rows, cols)

        preview_dir = '{}/{}'.format(dst, subdir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = preview_dir + '/image_latest.png'
        Image.fromarray(x).save(preview_path)
        preview_path = preview_dir + '/image{:0>8}.png'.format(trainer.updater.iteration)
        Image.fromarray(x).save(preview_path)

    return make_image

def make_iterator_func(dataset, batch_size):
    return chainer.iterators.MultithreadIterator(dataset, batch_size=batch_size,  repeat=True, shuffle=True, n_threads=FLAGS.dataset_worker_num)
    #return chainer.iterators.SerialIterator(dataset, batch_size=batch_size,  repeat=True, shuffle=True)


def batch_generate_func(gen, mapping, trainer):
    def generate(n_images):
        xp = gen.xp
        z = Variable(xp.asarray(mapping.make_hidden(n_images)[0]))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(mapping(z), stage=trainer.updater.stage)
        x = chainer.cuda.to_cpu(x.data)
        return x
    return generate


class RunningHelper(object):

    def __init__(self, use_mpi):
        self.use_mpi = use_mpi

        # Setup
        if self.use_mpi:
            if not mpi_available:
                raise RuntimeError('ChainerMN required for MPI but cannot be imported. Abort.')
            comm = chainermn.create_communicator(FLAGS.comm_name)
            if comm.mpi_comm.rank == 0:
                print('==========================================')
                print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
                print('Communcator name: {}'.format(FLAGS.comm_name))
                print('==========================================')
            fleet_size = MPI.COMM_WORLD.Get_size()
            device = comm.intra_rank
        else:
            fleet_size = 1
            comm = None
            device = FLAGS.gpu

        self.fleet_size, self.comm, self.device = fleet_size, comm, device

        self.is_master = is_master = not self.use_mpi or (self.use_mpi and comm.rank == 0)

        # Show effective hps
        effective_hps = {
            'is_master': self.is_master,
            'stage_interval': self.stage_interval,
            'dynamic_batch_size': self.dynamic_batch_size
        }
        self.print_log('Effective hps: {}'.format(effective_hps))

    @property
    def keep_smoothed_gen(self):
        return FLAGS.keep_smoothed_gen and self.is_master

    @property
    def use_cleargrads(self):
        # 1. Chainer 2/3 does not support clear_grads when running with MPI, so use zero_grads instead
        # 2. zero_grads on chainer >= 5.0.0 has a critical bug when running with MPI 
        return True

    @property
    def stage_interval(self):
        return FLAGS.stage_interval // self.fleet_size

    @property
    def dynamic_batch_size(self):
        fleet_size = self.fleet_size
        return [int(_) for _ in FLAGS.dynamic_batch_size.split(',')]

    def print_log(self, msg):
        print('[Device {}] {}'.format(self.device, msg))

    def check_hps_consistency(self):
        assert FLAGS.max_stage % 2 == 1
        # assert 4 * (2 ** ((FLAGS.max_stage - 1) // 2)) == FLAGS.image_size
        # assert 2 ** int(np.floor(np.log2(FLAGS.image_size))) == FLAGS.image_size
        assert len(self.dynamic_batch_size) >= FLAGS.max_stage

    def make_optimizer(self, model, alpha, beta1, beta2):
        self.print_log('Use Adam Optimizer with alpah = {}, beta1 = {}, beta2 = {}'.format(alpha, beta1, beta2))
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        if self.use_mpi:
            self.print_log('Use Optimizer with MPI')
            optimizer = chainermn.create_multi_node_optimizer(optimizer, self.comm)
        optimizer.setup(model)
        return optimizer

    def make_dataset(self, stage_int):
        if self.is_master:
            size = 4 * (2 ** ((stage_int + 1) // 2))
            _dataset = BaseDataset(
                json.load(open(FLAGS.dataset_config, 'r')),
                '%dx%d' % (size, size),
                [["resize", {"probability": 1, "width": size, "height": size, "resample_filter": "ANTIALIAS"}]]
            )
            self.print_log('Add (master) dataset for size {}'.format(size))
        else:
            _dataset = None
            self.print_log('Add (slave) dataset')

        if self.use_mpi:
            _dataset = chainermn.scatter_dataset(_dataset, self.comm)

        return _dataset
    
    def make_image_dataset(self, stage_int, image_dir):
        class Transform():
            def __init__(self, size):
                self.size = size

            def __call__(self, in_data):
                x, t = in_data
                t = t.astype(np.int32)
                x = chainercv.transforms.resize(x, (self.size, self.size))
                x = chainercv.transforms.random_flip(x, False, True)
                x = x / 127.5 - 1
                return x, t

        if self.is_master:
            size = 4 * (2 ** ((stage_int + 1) // 2))
            #_dataset = chainer.datasets.ImageDataset(images, dtype=np.float32)
            _dataset = np.load(image_dir, allow_pickle=True)
            _dataset = chainer.datasets.TransformDataset(_dataset, Transform(size))
            self.print_log('Add (master) dataset for size {}'.format(size))
        else:
            _dataset = None
            self.print_log('Add (slave) dataset')

        if self.use_mpi:
            _dataset = chainermn.scatter_dataset(_dataset, self.comm)

        return _dataset


def main():
    # chainer.global_config.type_check = False
    chainer.global_config.autotune = True
    chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)

    print(FLAGS)

    # Setup Models
    discriminator = Discriminator(ch=FLAGS.ch, enable_blur=FLAGS.enable_blur)
    if FLAGS.gpu > -1:
        chainer.cuda.get_device_from_id(FLAG.gpu).use()
        discriminator.to_gpu()

    _dataset = np.load(FLAGS.image_dir, allow_pickle=True)
    _dataset = chainer.datasets.TransformDataset(_dataset, Transform(64))
    optimizer = chainer.optimizers.Adam(alpha=args.adam_alpha_d)
    chainer.training.updater.StandardUpdater(tain_iter, optimizer)
    class TimeupTrigger():
        def __call__(self, _trainer):
            time = _trainer.elapsed_time
            if time > 8.75 * 60 * 60:
                print('facing time-limit. elapsed time=:{}'.format(time))
                return True
            return False
    trainer = training.Trainer(
        updater, TimeupTrigger(), out=FLAGS.out)

    classifier = chainer.links.Classifier(discriminator)

    trainer.extend(
        extensions.snapshot_object(discriminator, 'discriminator' + '_{.updater.iteration}.npz'),
        trigger=(FLAGS.snapshot_interval, 'iteration'))

    trainer.extend(
        extensions.ProgressBar(training_length=(updater.total_iteration, 'iteration'), update_interval=FLAGS.display_interval))

        
    report_keys = ['epoch', 'main/loss', 'validation/main/loss',
    'main/accuracy', 'validation/main/accuracy']
    trainer.extend(extensions.LogReport(keys=report_keys, trigger=(FLAGS.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(FLAGS.display_interval, 'iteration'))
    trainer.run()

import pdb, traceback, sys, code 

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except: 
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
