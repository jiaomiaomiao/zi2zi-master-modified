# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import argparse

from model.unet import UNet



parser = argparse.ArgumentParser(description='Train')

# mode setting
# 0 --> full train
# 1`--> fine_tune_trained
# 2 --> fine_tune_untrained
parser.add_argument('--running_mode', dest='running_mode',type=int,required=True)

# directories setting

parser.add_argument('--experiment_dir', dest='experiment_dir', default='./experiment/',
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_id', dest='experiment_id', type=int,
                    help='sequence id for the experiments you prepare to run',required=True)

# input data setting
# parser.add_argument('--train_name',dest='train_name',type=str,default='train.obj')
# parser.add_argument('--val_name',dest='val_name',type=str,default='val.obj')
parser.add_argument('--train_name',dest='train_name',type=str,required=True)
parser.add_argument('--val_name',dest='val_name',type=str,required=True)

# for losses setting
parser.add_argument('--L1_penalty', dest='L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', dest='Lconst_penalty', type=int, default=15, help='weight for const loss')
parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--ebdd_weight_penalty', dest='ebdd_weight_penalty', type=float, default=1,
                    help='weight for ebdd weight loss')


# ebdd setting
parser.add_argument('--font_num_for_train', dest='font_num_for_train', type=int, default=20,
                    help="number for distinct fonts for train")
parser.add_argument('--font_num_for_fine_tune', dest='font_num_for_fine_tune', type=int, default=1,
                    help="number for distinct fonts for fine_tune")
parser.add_argument('--ebdd_dictionary_dim', dest='ebdd_dictionary_dim', type=int, default=128,
                    help="dimension for ebdd dictionary")

# training param setting
parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='number of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, help='number of examples in batch',required=True)
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--schedule', dest='schedule', type=int, default=10, help='number of epochs to half learning rate')
parser.add_argument('--resume_training', dest='resume_training', type=int, help='resume from previous training',required=True)
parser.add_argument('--base_trained_model_dir',dest='base_trained_model_dir',type=str,required=True,
                    help='resume data from what dir')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')


# checking && backup setting
parser.add_argument('--sample_steps', dest='sample_steps', type=int, required=True,
                    help='number of batches in between two samples are drawn from validation set')
parser.add_argument('--checkpoint_steps', dest='checkpoint_steps', type=int, required=True,
                    help='number of batches in between two checkpoints')






# specific training scheme setting
parser.add_argument('--fine_tune', dest='fine_tune', type=str, required=True,
                    help='specific labels id to be fine tuned')
parser.add_argument('--sub_train_set_num',dest='sub_train_set_num',type=int,default=-1)


parser.add_argument('--freeze_encoder', dest='freeze_encoder', type=int, required=True,
                    help="freeze encoder weights during training")
parser.add_argument('--freeze_decoder', dest='freeze_decoder', type=int, required=True,
                    help="freeze decoder weights during training")
parser.add_argument('--freeze_discriminator', dest='freeze_discriminator', type=int, required=True,
                    help="freeze discriminator weights during training")
parser.add_argument('--freeze_ebdd_weights', dest='freeze_ebdd_weights', type=int, required=True,
                    help="freeze ebdd weights during training")






def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        fine_tune_list = None
        if not args.fine_tune == '-1':
            ids = args.fine_tune.split(",")
            fine_tune_list = set([int(i) for i in ids])



        model = UNet(running_mode=args.running_mode,
                     base_trained_model_dir=args.base_trained_model_dir,
                     experiment_dir=args.experiment_dir,experiment_id=args.experiment_id,
                     train_obj_name=args.train_name, val_obj_name=args.val_name,
                     sample_steps=args.sample_steps, checkpoint_steps=args.checkpoint_steps,

                     batch_size=args.batch_size,lr=args.lr,epoch=args.epoch,schedule=args.schedule,

                     ebdd_dictionary_dim=args.ebdd_dictionary_dim,

                     L1_penalty=args.L1_penalty,
                     Lconst_penalty=args.Lconst_penalty,
                     Ltv_penalty=args.Ltv_penalty,
                     ebdd_weight_penalty=args.ebdd_weight_penalty,



                     font_num_for_train=args.font_num_for_train,font_num_for_fine_tune=args.font_num_for_fine_tune,


                     resume_training=args.resume_training,


                     fine_tune=fine_tune_list,
                     sub_train_set_num=args.sub_train_set_num,

                     freeze_encoder=args.freeze_encoder,
                     freeze_decoder=args.freeze_decoder,
                     freeze_discriminator=args.freeze_discriminator,
                     freeze_ebdd_weights=args.freeze_ebdd_weights)
        model.register_session(sess)
        model.build_model()

        model.train_procedures()

input_args = ['--running_mode','0',
              '--base_trained_model_dir', './experiment/base_model_0/',
              '--experiment_id','0',

              '--train_name','train_debug.obj',
              '--val_name','train_debug.obj',

              '--batch_size', '2',

              '--resume_training','0',

              '--sample_steps','5',
              '--checkpoint_steps','5',
              '--epoch','10',

              '--fine_tune','2',
              '--sub_train_set_num','-1',

              '--freeze_encoder','0',
              '--freeze_decoder','0',
              '--freeze_discriminator','0',
              '--freeze_ebdd_weights','1'
              ]
#input_args = []
args = parser.parse_args(input_args)


if __name__ == '__main__':
    tf.app.run()
