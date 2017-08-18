# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.client import device_lib

import argparse
import numpy as np
import os
import shutil
from os import listdir


from model.unet import UNet
#from model.unet import UNet


input_args = ['--training_mode','0',
              '--base_trained_model_dir', './',
              '--infer_copy_num','2',

              '--inferred_result_saving_path','../test_infer/',

              '--infer_name','../Font_ttf_otf/Obj/essay_simplified.obj',





              '--font_num_for_train','20',
              '--fine_tune','1,6,10',

              '--freeze_encoder','0',
              '--freeze_decoder','0',

              ]


parser = argparse.ArgumentParser(description='Infer')

# mode setting
# 0 --> full train
# 1`--> fine_tune_trained
# 2 --> fine_tune_untrained
parser.add_argument('--training_mode', dest='training_mode',type=int,required=True)

# directories setting

parser.add_argument('--inferred_result_saving_path', dest='inferred_result_saving_path', default='./experiment/',
                    help='inferred result')
parser.add_argument('--infer_copy_num',dest='infer_copy_num',type=int,required=True)



# input data setting
# parser.add_argument('--train_name',dest='train_name',type=str,default='train.obj')
# parser.add_argument('--val_name',dest='val_name',type=str,default='val.obj')
parser.add_argument('--infer_name',dest='infer_name',type=str,required=True)


# ebdd setting
parser.add_argument('--font_num_for_train', dest='font_num_for_train', type=int, required=True,
                    help="number for distinct fonts for train")
parser.add_argument('--ebdd_dictionary_dim', dest='ebdd_dictionary_dim', type=int, default=128,
                    help="dimension for ebdd dictionary")

# training param setting
parser.add_argument('--base_trained_model_dir',dest='base_trained_model_dir',type=str,required=True,
                    help='resume data from what dir')









# specific training scheme setting
parser.add_argument('--fine_tune', dest='fine_tune', type=str, required=True,
                    help='specific labels id to be fine tuned')


parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, required=True,
                    help="freeze encoder weights during training")
parser.add_argument('--freeze_decoder', dest='freeze_decoder', type=int, required=True,
                    help="freeze decoder weights during training")
parser.add_argument('--freeze_discriminator', dest='freeze_discriminator', type=int, default=False,
                    help="freeze discriminator weights during training")
parser.add_argument('--freeze_ebdd_weights', dest='freeze_ebdd_weights', type=int, default=-1,
                    help="freeze ebdd weights during training")






def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    cpu_device=[x.name for x in local_device_protos if x.device_type == 'CPU']
    gpu_device=[x.name for x in local_device_protos if x.device_type == 'GPU']
    print("Available CPU:%s with number:%d" % (cpu_device, len(cpu_device)))
    print("Available GPU:%s with number:%d" % (gpu_device, len(gpu_device)))
    return cpu_device, gpu_device,len(cpu_device),len(gpu_device)





def main(_):



    avalialbe_cpu, available_gpu,available_cpu_num, available_gpu_num = get_available_gpus()
    forward_backward_device=list()
    if available_gpu_num==0:
        print ("No available GPU found!!! The calculation will be performed with CPU only.")
        args.device_mode=0

    parameter_update_device = avalialbe_cpu[0]
    forward_backward_device.append(avalialbe_cpu[0])
    # parameter_update_device = available_gpu[0]
    # forward_backward_device.append(available_gpu[0])

    forward_backward_device_list=list()
    forward_backward_device_list.extend(forward_backward_device)
    print("Available devices for forward && backward:")
    for device in forward_backward_device_list:
        print(device)
    print("Available devices for parameter update:%s" % parameter_update_device)


    if args.fine_tune=='All':
        args.fine_tune=''
        for ii in range(args.font_num_for_train):
            if ii==0:
                args.fine_tune=args.fine_tune+str(ii)
            else:
                args.fine_tune = args.fine_tune + ','+str(ii)

    ids = args.fine_tune.split(",")
    fine_tune_list = set([int(i) for i in ids])








    model_for_train = UNet(training_mode=args.training_mode,
                           base_trained_model_dir=args.base_trained_model_dir,
                           infer_obj_name=args.infer_name,
                           infer_copy_num=args.infer_copy_num,



                           ebdd_dictionary_dim=args.ebdd_dictionary_dim,




                           font_num_for_train=args.font_num_for_train,



                           freeze_encoder=args.freeze_encoder,freeze_decoder=args.freeze_decoder,

                           fine_tune=fine_tune_list,

                           parameter_update_device=parameter_update_device,
                           forward_backward_device=forward_backward_device_list)

    base_models = listdir(args.base_trained_model_dir)
    for traveller in base_models:
        if not traveller.find('DS') == -1:
            base_models.remove(traveller)
    base_models_with_path = list()
    for ii in range(len(base_models)):
        base_models_with_path.append(os.path.join(args.base_trained_model_dir, base_models[ii]))
        print("Found Model No:%d named %s" %(ii,base_models[ii]))


    for ii in range(len(base_models_with_path)):
        # current_inferred_result_saving_path=os.path.join(args.inferred_result_saving_path,base_models[ii])
        # not_freeze_encoder=current_inferred_result_saving_path.find('encoder_not_freeze')
        # not_freeze_decoder=current_inferred_result_saving_path.find('decoder_not_freeze')
        # if args.training_mode == 0:
        #     freeze_ebdd_weights = 1
        #     freeze_encoder = 0
        #     freeze_decoder = 0
        # else:
        #     freeze_ebdd_weights = 0
        #     if not not_freeze_encoder==-1:
        #         freeze_encoder = 1
        #     else:
        #         freeze_encoder = 0
        #     if not not_freeze_decoder==-1:
        #         freeze_decoder=1
        #     else:
        #         freeze_decoder=0



        current_inferred_result_saving_path = os.path.join(args.inferred_result_saving_path, base_models[ii])
        freeze_ebdd_weights = 1
        freeze_encoder = 0
        freeze_decoder = 0



        if os.path.exists(current_inferred_result_saving_path):
            shutil.rmtree(current_inferred_result_saving_path)
        os.makedirs(current_inferred_result_saving_path)
        print("New inferred dir created for %s." % (current_inferred_result_saving_path))


        model_for_train.infer_procedures(inferred_result_saving_path='../test_infer',
                                         base_trained_model_dir='../test_infer',
                                         freeze_ebdd_weights=freeze_ebdd_weights,
                                         freeze_encoder=freeze_encoder,
                                         freeze_decoder=freeze_decoder)





#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
