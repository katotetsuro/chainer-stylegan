from argparse import ArgumentParser
import math


parser = ArgumentParser()

# hps (training dynamics)
parser.add_argument('--seed', type=int, default=19260817)
parser.add_argument('--adam_alpha_g', type=float, default=0.001, help='alpha in Adam optimizer')
parser.add_argument('--adam_alpha_d', type=float, default=0.001, help='alpha in Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.0, help='beta1 in Adam optimizer')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 in Adam optimizer')
parser.add_argument('--lambda_gp', type=float, default=5.0, help='Lambda GP')
parser.add_argument('--smoothing', default=0.999)
parser.add_argument('--keep_smoothed_gen', action='store_true', help='Whether to keep a smoothed version of generator.')
parser.add_argument('--dynamic_batch_size', type=str, default='256,256,256,128,128,64,64,32,32',
                     help='comma-split list of dynamic batch size w.p.t. stage')
parser.add_argument('--stage_interval', type=int, default=40000)
parser.add_argument('--max_stage', default=9, help='Size of image.')


parser.add_argument('--auto_resume', action='store_true', help='Whether to automatically resume')

# algorithm & architecture
parser.add_argument('--ch', type=int, default=512, help='#Channels')
parser.add_argument('--debug_start_instance', default=0, help='Change starting iteration for debugging.')


# hps (device)
parser.add_argument('--gpu', default=0, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--use_mpi', action='store_true', help='Whether to use MPI for multi-GPU training.')
parser.add_argument('--comm_name', default='pure_nccl', help='ChainerMN communicator name')
parser.add_argument('--enable_cuda_profiling', action='store_true', help='Whether to enable CUDA profiling.')


# hps (I/O)
parser.add_argument('--out', default='result', help='Directory to output the result')
parser.add_argument('--auto_resume_dir', default='', help='Directory for loading the saved models')
parser.add_argument('--dataset_config', default='', help='Dataset config json')
parser.add_argument('--dataset_worker_num', default=4, help='Number of threads in dataset loader')
parser.add_argument('--image_dir', default='', help='Image directory containing training data')

parser.add_argument('--snapshot_interval', type=int, default=30000, help='Interval of snapshot')
parser.add_argument('--evaluation_sample_interval', type=int, default=2000, help='Interval of evaluation sampling')
parser.add_argument('--display_interval', default=100, help='Interval of displaying log to console')
parser.add_argument('--get_model_from_interation', default='', help='Load this iteration (it is a string)')

# hps FID
parser.add_argument('--fid_interval', default=0, help='Enable FID when > 0')
parser.add_argument('--fid_real_stat', default='', help='Save NPZ of real images')
parser.add_argument('--fid_clfs_type', default='', help='i2v_v5/inception')
parser.add_argument('--fid_clfs_path', default='', help='classifier path')
parser.add_argument('--fid_skip_first', action='store_true', help='Whether to skip FID calculation when iter = 0')

# Style GAN
parser.add_argument('--style_mixing_rate', type=float, default=0.9, help=' Style Mixing Prob')
parser.add_argument('--enable_blur', action='store_true', help='Enable blur function after upscaling/downscaling')


FLAGS = parser.parse_args()

stage2reso = {
    0: 4,
    1: 8,
    2: 8,
    3: 16,
    4: 16,
    5: 32,
    6: 32,
    7: 64,
    8: 64,
    9: 128,
    10: 128,
    11: 256,
    12: 256,
    13: 512,
    14: 512,
    15: 1024,
    16: 1024,
    17: 1024
}

gpu_lr = {
    1: {6: 1.5, 7: 1.5, 8: 1.5},
    2: {13: 1.5, 14: 1.5, 15: 2, 16: 2, 17: 2},
    3: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 2.5, 16: 2.5, 17: 2.5},
    4: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 3, 16: 3, 17: 3},
    8: {9: 1.5, 10: 1.5, 11: 2, 12: 2, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3},
}

def get_lr_scale_factor(total_gpu, stage):
    gpu_lr_d = gpu_lr.get(total_gpu, gpu_lr[1])
    stage = math.floor(stage)
    if stage >= 18:
        return gpu_lr_d[17]
    return gpu_lr_d.get(stage, 1)
