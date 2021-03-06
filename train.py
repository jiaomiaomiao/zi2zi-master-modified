# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse
import numpy as np

# from model.unet import UNet
#from model.unet import UNet
from model.unet import UNet


input_args = ['--training_mode','0',
              '--base_trained_model_dir', './',

              '--experiment_id','20170901',

              '--train_name','/dataA/Harric/Chinese_Character_Generation/Font_Binary_Data/Font_Obj_80_PF/train.obj',
              '--val_name','/dataA/Harric/Chinese_Character_Generation/Font_Binary_Data/Font_Obj_80_PF/val.obj',

#'--train_name','/dataA/Harric/Chinese_Character_Generation/Font_Binary_Data/Font_Obj_80_PF/essay_simplified.obj',
#              '--val_name','/dataA/Harric/Chinese_Character_Generation/Font_Binary_Data/Font_Obj_80_PF/essay_traditional.obj',

              '--batch_size', '64',
              '--resume_training','0',

              '--samples_per_font','120000',
              '--schedule','5',
              '--optimization_method','adam',

              '--base_training_font_num','80',
              '--sub_train_set_num','-1',

              '--freeze_encoder','0',
              '--freeze_decoder','0',

	          '--ebdd_dictionary_dim','256',


              '--device_mode','3',

              '--final_learning_rate_pctg','0.2',


              ]
# device_mode=0: training only on cpu
# device_mode=1: forward & backward on multiple gpus && parameter update on cpu
# device_mode=2: forward & backward on multiple gpus && parameter update on one single gpu
# device_mode=3: forward & backward & parameter update on a single gpu


parser = argparse.ArgumentParser(description='Train')


# mode setting
# 0 --> full train
# 1`--> fine_tune_trained
# 2 --> fine_tune_untrained
parser.add_argument('--training_mode', dest='training_mode',type=int,required=True)


# directories setting

parser.add_argument('--experiment_dir', dest='experiment_dir', default='./experiment/',
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=str,
                    help='sequence id for the experiments you prepare to run',required=True)


# input data setting
# parser.add_argument('--train_name',dest='train_name',type=str,default='train.obj')
# parser.add_argument('--val_name',dest='val_name',type=str,default='val.obj')
parser.add_argument('--train_name',dest='train_name',type=str,required=True)
parser.add_argument('--val_name',dest='val_name',type=str,required=True)

# for losses setting
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--ebdd_weight_penalty', dest='ebdd_weight_penalty', type=float, default=1,
                    help='weight for ebdd weight loss')


# ebdd setting
parser.add_argument('--base_training_font_num', dest='base_training_font_num', type=int, required=True,
                    help="number for distinct base fonts for train with mode 0")
parser.add_argument('--ebdd_dictionary_dim', dest='ebdd_dictionary_dim', type=int, required=True,
                    help="dimension for ebdd dictionary")

# training param setting
parser.add_argument('--samples_per_font', dest='samples_per_font', type=int, required=True,
                    help='how many samples shall be seen for a epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, help='number of examples in batch',required=True)
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--final_learning_rate_pctg', dest='final_learning_rate_pctg', type=float, default=0.2,
                    help='final leanring rate of the initial one')

parser.add_argument('--optimization_method',type=str,required=True,help='optimization method selection')
parser.add_argument('--schedule', dest='schedule', type=int, required=True, help='number of epochs to half learning rate')
parser.add_argument('--resume_training', dest='resume_training', type=int, help='resume from previous training',required=True)
parser.add_argument('--base_trained_model_dir',dest='base_trained_model_dir',type=str,required=True,
                    help='resume data from what dir')









# specific training scheme setting
parser.add_argument('--sub_train_set_num',dest='sub_train_set_num',type=int,default=-1)


parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, required=True,
                    help="freeze encoder weights during training")
parser.add_argument('--freeze_decoder', dest='freeze_decoder', type=int, required=True,
                    help="freeze decoder weights during training")
parser.add_argument('--freeze_discriminator', dest='freeze_discriminator', type=int, default=False,
                    help="freeze discriminator weights during training")
parser.add_argument('--freeze_ebdd_weights', dest='freeze_ebdd_weights', type=int, default=-1,
                    help="freeze ebdd weights during training")


# device selection
parser.add_argument('--device_mode', dest='device_mode',type=int,required=True,
                    help='Device mode selection')







def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']
    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)



def main(_):
    avalialbe_cpu, available_gpu, available_cpu_num, available_gpu_num = get_available_gpus()
    forward_backward_device = list()
    if available_gpu_num == 0:
        print("No available GPU found!!! The calculation will be performed with CPU only.")
        args.device_mode = 0

    if args.device_mode==0:
        parameter_update_device=avalialbe_cpu[0]
        forward_backward_device.append(avalialbe_cpu[0])
    elif args.device_mode==1:
        parameter_update_device = avalialbe_cpu[0]
        forward_backward_device.extend(available_gpu)
    elif args.device_mode==2:
        parameter_update_device=available_gpu[1]
        forward_backward_device.append(available_gpu[0])
        forward_backward_device.append(available_gpu[1])
        forward_backward_device.append(available_gpu[2])
    elif args.device_mode==3:
        parameter_update_device = available_gpu[0]
        forward_backward_device.append(available_gpu[0])

    forward_backward_device_list=list()
    forward_backward_device_list.extend(forward_backward_device)
    print("Available devices for forward && backward:")
    for device in forward_backward_device_list:
        print(device)
    print("Available devices for parameter update:%s" % parameter_update_device)




    model = UNet(training_mode=args.training_mode,
                 base_trained_model_dir=args.base_trained_model_dir,
                 experiment_dir=args.experiment_dir, experiment_id=args.experiment_id,
                 train_obj_name=args.train_name, val_obj_name=args.val_name,

                 optimization_method=args.optimization_method,

                 batch_size=args.batch_size, lr=args.lr, final_learning_rate_pctg=args.final_learning_rate_pctg,
                 samples_per_font=args.samples_per_font, schedule=args.schedule,

                 ebdd_dictionary_dim=args.ebdd_dictionary_dim,

                 L1_penalty=args.L1_penalty,
                 Lconst_penalty=args.Lconst_penalty,
                 ebdd_weight_penalty=args.ebdd_weight_penalty,

                 base_training_font_num=args.base_training_font_num,

                 resume_training=args.resume_training,

                 freeze_encoder=args.freeze_encoder, freeze_decoder=args.freeze_decoder,

                 sub_train_set_num=args.sub_train_set_num,

                 parameter_update_device=parameter_update_device,
                 forward_backward_device=forward_backward_device_list,




                 )

    model.train_procedures()





#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
