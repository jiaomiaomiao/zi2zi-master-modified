# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import shutil
import time
from collections import namedtuple
from .ops import conv2d, deconv2d, lrelu, fc, batch_norm, init_embedding_dictionary,init_embedding_weights, conditional_instance_norm, weight_norm
from .dataset import TrainDataProvider, InjectDataProvider
from .utils import scale_back, merge, save_concat_images


import matplotlib.pyplot as plt
import matplotlib.image as img


# Auxiliary wrapper classes
# Used to save handles(important nodes in computation graph) for later evaluation
LossHandle = namedtuple("LossHandle", ["d_loss", "g_loss",
                                       "const_loss", "l1_loss", "tv_loss", "ebdd_weight_loss",
                                       "category_loss", "real_category_loss", "fake_category_loss",
                                       "cheat_loss",])
InputHandle = namedtuple("InputHandle", ["real_data", "validate_image","ebdd_weights_static","targeted_label",
                                         "weight_org_fig_placeholder","weight_net_fig_placeholder","weight_loss_fig_placeholder"])
EvalHandle = namedtuple("EvalHandle", ["encoder","generator", "target", "source", "ebdd_dictionary"])

SummaryHandle = namedtuple("SummaryHandle", ["d_merged", "g_merged","valiadte_image_merged",
                                             "weight_org_fig","weight_net_fig","weight_loss_fig"])

DebugHandle = namedtuple("DebugHandle", ["ebdd_weights_org",
                                         "ebdd_weights_net",
                                         "ebdd_weights_loss"])

eps= 1e-3


class UNet(object):
    def __init__(self,
                 running_mode=-1,
                 base_trained_model_dir='./experiment/checkpoint',

                 experiment_dir=None, experiment_id='0',
                 train_obj_name='train_debug.obj', val_obj_name='val_debug.obj',

                 sample_steps=500, checkpoint_steps=500,
                 optimization_method='adam',

                 batch_size=16,lr=0.001,itrs=2000,schedule=10,

                 input_width=256, output_width=256, input_filters=3, output_filters=3,
                 generator_dim=64, discriminator_dim=64,ebdd_dictionary_dim=128,

                 L1_penalty=100, Lconst_penalty=15, Ltv_penalty=0.0,ebdd_weight_penalty=1.0,

                 font_num_for_train=20, font_num_for_fine_tune=1,

                 resume_training=True,



                 fine_tune=None,
                 sub_train_set_num=-1,

                 freeze_encoder=False,
                 freeze_decoder=False,
                 freeze_discriminator=False,
                 freeze_ebdd_weights=True
                 ):
        self.running_mode = running_mode
        self.base_trained_model_dir = base_trained_model_dir
        self.experiment_dir = experiment_dir
        self.experiment_id = experiment_id
        self.data_dir = os.path.join(self.experiment_dir, "font_binary_data")
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoint")
        self.sample_dir = os.path.join(self.experiment_dir, "sample")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.weight_bar_dir = os.path.join(self.experiment_dir,"weight_bar")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("new checkpoint directory created")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print("new log directory created")
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
            print("new sample directory created")
        if not os.path.exists(self.weight_bar_dir):
            os.makedirs(self.weight_bar_dir)
            print("new weight bar directory created")




        self.train_obj_name=train_obj_name
        self.val_obj_name = val_obj_name
        self.sample_steps = sample_steps
        self.checkpoint_steps = checkpoint_steps
        self.optimization_method=optimization_method


        self.batch_size = batch_size
        self.lr=lr
        self.itrs=itrs
        self.schedule=schedule



        self.input_width = input_width
        self.output_width = output_width
        self.input_filters = input_filters
        self.output_filters = output_filters

        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.ebdd_dictionary_dim = ebdd_dictionary_dim


        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.ebdd_weight_penalty = ebdd_weight_penalty


        self.font_num_for_train = font_num_for_train
        self.font_num_for_fine_tune = font_num_for_fine_tune

        self.resume_training = resume_training



        self.fine_tune=fine_tune
        self.sub_train_set_num=sub_train_set_num

        self.freeze_encoder=freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.freeze_discriminator = freeze_discriminator
        self.freeze_ebdd_weights = freeze_ebdd_weights



        # init all the directories
        self.sess = None
        self.counter=0
        self.print_separater="####################################################################"

        id, self.checkpoint_dir, self.log_dir, self.sample_dir, self.weight_bar_dir = self.get_model_id_and_dir()
        if self.resume_training == 0 and os.path.exists(self.log_dir):
            shutil.rmtree(self.checkpoint_dir)
            shutil.rmtree(self.log_dir)
            shutil.rmtree(self.sample_dir)
            shutil.rmtree(self.weight_bar_dir)
            print("new model ckpt / log / sample / weight bar directories removed for %s" % id )
        if not os.path.exists(self.log_dir):
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.sample_dir)
            os.makedirs(self.weight_bar_dir)
            print("new model ckpt / log / sample / weight bar directories created for %s" % id)

    def encoder(self, images, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            encode_layers = dict()

            def encode_layer(x, output_filters, layer):
                act = lrelu(x)
                conv = conv2d(act, output_filters=output_filters, scope="gen_enc%d_conv" % layer)
                enc = batch_norm(conv, is_training, scope="gen_enc%d_bn" % layer)
                encode_layers["enc%d" % layer] = enc
                return enc

            e1 = conv2d(images, self.generator_dim, scope="gen_enc1_conv")
            encode_layers["enc1"] = e1
            e2 = encode_layer(e1, self.generator_dim * 2, 2)
            e3 = encode_layer(e2, self.generator_dim * 4, 3)
            e4 = encode_layer(e3, self.generator_dim * 8, 4)
            e5 = encode_layer(e4, self.generator_dim * 8, 5)
            e6 = encode_layer(e5, self.generator_dim * 8, 6)
            e7 = encode_layer(e6, self.generator_dim * 8, 7)
            e8 = encode_layer(e7, self.generator_dim * 8, 8)

            return e8, encode_layers

    def decoder(self, encoded, encoding_layers, ids, inst_norm, is_training, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            s = self.output_width
            s2, s4, s8, s16, s32, s64, s128 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(
                s / 64), int(s / 128)

            def decode_layer(x, output_width, output_filters, layer, enc_layer, dropout=False, do_concat=True):
                dec = deconv2d(tf.nn.relu(x), [self.batch_size, output_width,
                                               output_width, output_filters], scope="gen_dec%d_deconv" % layer)
                if layer != 8:
                    # IMPORTANT: normalization for last layer
                    # Very important, otherwise GAN is unstable
                    # Trying conditional instance normalization to
                    # overcome the fact that batch normalization offers
                    # different train/test statistics
                    if inst_norm:
                        dec = conditional_instance_norm(dec, ids, self.font_num_for_train, scope="gen_dec%d_inst_norm" % layer)
                    else:
                        dec = batch_norm(dec, is_training, scope="gen_dec%d_bn" % layer)
                if dropout:
                    dec = tf.nn.dropout(dec, 0.5)
                if do_concat:
                    dec = tf.concat([dec, enc_layer], 3)
                return dec

            d1 = decode_layer(encoded, s128, self.generator_dim * 8, layer=1, enc_layer=encoding_layers["enc7"],dropout=True)
            d2 = decode_layer(d1, s64, self.generator_dim * 8, layer=2, enc_layer=encoding_layers["enc6"], dropout=True)
            d3 = decode_layer(d2, s32, self.generator_dim * 8, layer=3, enc_layer=encoding_layers["enc5"], dropout=True)
            d4 = decode_layer(d3, s16, self.generator_dim * 8, layer=4, enc_layer=encoding_layers["enc4"])
            d5 = decode_layer(d4, s8, self.generator_dim * 4, layer=5, enc_layer=encoding_layers["enc3"])
            d6 = decode_layer(d5, s4, self.generator_dim * 2, layer=6, enc_layer=encoding_layers["enc2"])
            d7 = decode_layer(d6, s2, self.generator_dim, layer=7, enc_layer=encoding_layers["enc1"])
            d8 = decode_layer(d7, s, self.output_filters, layer=8, enc_layer=None, do_concat=False)

            output = tf.nn.tanh(d8)  # scale to (-1, 1)
            return output

    def generator(self, images, ebdd_vector, ebdd_weights, inst_norm, is_training, reuse=False):
        e8, enc_layers = self.encoder(images, is_training=is_training, reuse=reuse)


        # return ebdds[ebdd_weights], ebdd_weights is imported labels

        embedded = tf.concat([e8, ebdd_vector], 3)
        output = self.decoder(embedded, enc_layers, ebdd_weights, inst_norm, is_training=is_training, reuse=reuse)
        return output, e8

    def discriminator(self, image, is_training, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = lrelu(conv2d(image, self.discriminator_dim, scope="dis_h0_conv"))
            h1 = lrelu(batch_norm(conv2d(h0, self.discriminator_dim * 2, scope="dis_h1_conv"),
                                  is_training, scope="dis_bn_1"))
            h2 = lrelu(batch_norm(conv2d(h1, self.discriminator_dim * 4, scope="dis_h2_conv"),
                                  is_training, scope="dis_bn_2"))
            h3 = lrelu(batch_norm(conv2d(h2, self.discriminator_dim * 8, sh=1, sw=1, scope="dis_h3_conv"),
                                  is_training, scope="dis_bn_3"))
            # real or fake binary loss
            fc1 = fc(tf.reshape(h3, [self.batch_size, -1]), 1, scope="dis_fc1")
            # category loss
            fc2 = fc(tf.reshape(h3, [self.batch_size, -1]), self.font_num_for_train, scope="dis_fc2")

            return tf.nn.sigmoid(fc1), fc1, fc2

    def multi_embedding_weights_init(self):
        ebdd_weights_static = tf.placeholder(tf.float32, shape=(self.batch_size, self.font_num_for_train),
                                             name="gen_ebdd_weights_static")
        ebdd_weights_dynamic = init_embedding_weights(size=[1, self.font_num_for_train],
                                                      name="gen_ebdd_weights_dynamic")
        if self.freeze_ebdd_weights == True:
            ebdd_weights_org = ebdd_weights_static
            ebdd_weights_batch_normed = weight_norm(ebdd_weights_org)
            ebdd_weights_for_net = ebdd_weights_batch_normed
            ebdd_weights_for_loss = ebdd_weights_batch_normed
        else:
            ebdd_weights_org = tf.matmul(tf.ones([self.batch_size, 1], dtype=tf.float32), ebdd_weights_dynamic)
            ebdd_weights_for_loss = tf.nn.softmax(ebdd_weights_org)
            ebdd_weights_for_net = weight_norm(ebdd_weights_org)

        return ebdd_weights_static,ebdd_weights_dynamic,ebdd_weights_org,ebdd_weights_for_net,ebdd_weights_for_loss





    def build_model(self, is_training=True, inst_norm=False):



        real_data = tf.placeholder(tf.float32,
                                   [self.batch_size, self.input_width, self.input_width,
                                    self.input_filters + self.output_filters],
                                   name='real_A_and_B_images')





        ebdd_weights_static, \
        ebdd_weights_dynamic, \
        ebdd_weights_org, \
        ebdd_weights_for_net, \
        ebdd_weights_for_loss = self.multi_embedding_weights_init()
        ebdd_weights_hist_org_summary = tf.summary.histogram("ebdd_weight_org_hist", ebdd_weights_org)
        ebdd_weights_hist_net_summary = tf.summary.histogram("ebdd_weight_net_hist", ebdd_weights_for_net)
        ebdd_weights_hist_loss_summary = tf.summary.histogram("ebdd_weight_loss_hist", ebdd_weights_for_loss)

        weight_org_fig_placeholder = tf.placeholder(tf.float32, [1, 900, 1200, 4])
        weight_net_fig_placeholder = tf.placeholder(tf.float32, [1, 900, 1200, 4])
        weight_loss_fig_placeholder = tf.placeholder(tf.float32, [1, 900, 1200, 4])
        weight_org_fig_summary = tf.summary.image('weight_org_fig', weight_org_fig_placeholder)
        weight_net_fig_summary = tf.summary.image('weight_net_fig', weight_net_fig_placeholder)
        weight_loss_fig_summary = tf.summary.image('weight_loss_fig', weight_loss_fig_placeholder)


        ebdd_dictionary = init_embedding_dictionary(self.font_num_for_train, self.ebdd_dictionary_dim)
        ebdd_vector = tf.matmul(ebdd_weights_for_net, ebdd_dictionary)
        ebdd_vector = tf.reshape(ebdd_vector, [self.batch_size, 1, 1, self.ebdd_dictionary_dim])



        # target images
        real_B = real_data[:, :, :, :self.input_filters]
        # source images
        real_A = real_data[:, :, :, self.input_filters:self.input_filters + self.output_filters]


        # ebdd processing

        #ebdd_vector = tf.nn.ebdd_lookup(ebdd_dictionary, ids=ebdd_weights)


        fake_B, encoded_real_B = self.generator(images=real_A, ebdd_vector=ebdd_vector, ebdd_weights=ebdd_weights_for_net, is_training=is_training,
                                                inst_norm=inst_norm)
        real_AB = tf.concat([real_A, real_B], 3)
        fake_AB = tf.concat([real_A, fake_B], 3)

        # Note it is not possible to set reuse flag back to False
        # initialize all variables before setting reuse to True
        real_D, real_D_logits, real_category_logits = self.discriminator(real_AB, is_training=is_training, reuse=False)
        fake_D, fake_D_logits, fake_category_logits = self.discriminator(fake_AB, is_training=is_training, reuse=True)

        # encoding constant loss
        # this loss assume that generated imaged and real image
        # should reside in the same space and close to each other
        encoded_fake_B = self.encoder(fake_B, is_training, reuse=True)[0]
        const_loss = tf.reduce_mean(tf.square(encoded_real_B - encoded_fake_B)) * self.Lconst_penalty
        const_loss_summary = tf.summary.scalar("const_loss", const_loss)

        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * tf.reduce_mean(tf.abs(fake_B - real_B))
        l1_loss_summary = tf.summary.scalar("l1_loss", l1_loss)
        # total variation loss
        width = self.output_width
        tv_loss = (tf.nn.l2_loss(fake_B[:, 1:, :, :] - fake_B[:, :width - 1, :, :]) / width
                   + tf.nn.l2_loss(fake_B[:, :, 1:, :] - fake_B[:, :, :width - 1, :]) / width) * self.Ltv_penalty
        tv_loss_summary = tf.summary.scalar("tv_loss", tv_loss)


        # category loss
        true_labels = tf.reshape(ebdd_weights_for_loss,
                                 shape=[self.batch_size, self.font_num_for_train])
        real_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_logits,
                                                                                    labels=true_labels))
        fake_category_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_logits,
                                                                                    labels=true_labels))
        category_loss = (real_category_loss + fake_category_loss) / 2.0
        fake_category_loss_summary = tf.summary.scalar("category_fake_loss", fake_category_loss)
        real_category_loss_summary = tf.summary.scalar("category_real_loss", real_category_loss)
        category_loss_summary = tf.summary.scalar("category_loss", category_loss)



        # binary real/fake loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits,
                                                                             labels=tf.ones_like(real_D)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                             labels=tf.zeros_like(fake_D)))
        d_loss_real_summary = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_summary = tf.summary.scalar("d_loss_fake", d_loss_fake)


        # maximize the chance generator fool the discriminator (for the generator)
        cheat_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits,
                                                                            labels=tf.ones_like(fake_D)))
        cheat_loss_summary = tf.summary.scalar("cheat_loss", cheat_loss)





        # embedding weight loss && difference checker
        ebdd_weight_loss = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(ebdd_weights_org,axis=1),tf.ones([self.batch_size],dtype=tf.float32)))) * self.ebdd_weight_penalty
        ebdd_weight_loss_summary = tf.summary.scalar("ebdd_weight_loss", ebdd_weight_loss)
        targeted_label = tf.placeholder(tf.float32, shape=(self.batch_size, self.font_num_for_train),name="target_label")
        if self.running_mode==1:
            label_difference_org = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_org)))
            label_difference_net = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_for_net)))
            label_difference_loss = tf.reduce_mean(tf.abs(tf.subtract(targeted_label, ebdd_weights_for_loss)))
            ebdd_label_diff_org_summary = tf.summary.scalar("ebdd_label_diff_org", label_difference_org)
            ebdd_label_diff_net_summary = tf.summary.scalar("ebdd_label_diff_net", label_difference_net)
            ebdd_label_diff_loss_summary = tf.summary.scalar("ebdd_label_diff_loss", label_difference_loss)

            fine_tune_list = list()
            for ii in self.fine_tune:
                fine_tune_list.append(ii)

            weight_checker_for_org = ebdd_weights_org[0, fine_tune_list[0]]
            weight_checker_for_net = ebdd_weights_for_net[0, fine_tune_list[0]]
            weight_checker_for_loss = ebdd_weights_for_loss[0, fine_tune_list[0]]
            ebdd_weight_checker_for_org_summary = tf.summary.scalar("ebdd_weight_checker_for_org",
                                                                    weight_checker_for_org)
            ebdd_weight_checker_for_net_summary = tf.summary.scalar("ebdd_weight_checker_for_net",
                                                                    weight_checker_for_net)
            ebdd_weight_checker_for_loss_summary = tf.summary.scalar("ebdd_weight_checker_for_loss",
                                                                     weight_checker_for_loss)









        d_loss = d_loss_real + d_loss_fake + category_loss
        d_loss_summary = tf.summary.scalar("d_loss", d_loss)

        g_loss = l1_loss + const_loss + ebdd_weight_loss + cheat_loss + fake_category_loss
        g_loss_summary = tf.summary.scalar("g_loss", g_loss)



        validate_image = tf.placeholder(tf.float32,[1,self.batch_size*self.input_width,self.input_width+self.output_width,3])
        validate_image_summary = tf.summary.image('Validate_Image',validate_image)





        d_merged_summary = tf.summary.merge([d_loss_real_summary, d_loss_fake_summary,
                                             category_loss_summary, real_category_loss_summary, fake_category_loss_summary,
                                             d_loss_summary])

        g_merged_summary = tf.summary.merge([l1_loss_summary, const_loss_summary,
                                             ebdd_weight_loss_summary,
                                             ebdd_weights_hist_org_summary, ebdd_weights_hist_net_summary,ebdd_weights_hist_loss_summary,
                                             tv_loss_summary,
                                             cheat_loss_summary,
                                             fake_category_loss_summary,
                                             g_loss_summary])

        if self.running_mode==1:
            g_merged_summary = tf.summary.merge([g_merged_summary,
                                                 ebdd_label_diff_org_summary,ebdd_label_diff_net_summary,ebdd_label_diff_loss_summary,
                                                 ebdd_weight_checker_for_org_summary,
                                                 ebdd_weight_checker_for_net_summary,
                                                 ebdd_weight_checker_for_loss_summary])



        # expose useful nodes in the graph as handles globally
        input_handle = InputHandle(real_data=real_data,
                                   validate_image=validate_image,
                                   ebdd_weights_static=ebdd_weights_static,
                                   targeted_label=targeted_label,
                                   weight_org_fig_placeholder=weight_org_fig_placeholder,
                                   weight_net_fig_placeholder=weight_net_fig_placeholder,
                                   weight_loss_fig_placeholder=weight_loss_fig_placeholder
                                   )


        loss_handle = LossHandle(d_loss=d_loss,
                                 g_loss=g_loss,
                                 const_loss=const_loss,
                                 l1_loss=l1_loss,
                                 tv_loss=tv_loss,
                                 ebdd_weight_loss=ebdd_weight_loss,
                                 category_loss=category_loss,
                                 real_category_loss=real_category_loss,
                                 fake_category_loss=fake_category_loss,
                                 cheat_loss=cheat_loss)

        eval_handle = EvalHandle(encoder=encoded_real_B,
                                 generator=fake_B,
                                 target=real_B,
                                 source=real_A,
                                 ebdd_dictionary=ebdd_dictionary)

        summary_handle = SummaryHandle(d_merged=d_merged_summary,
                                       g_merged=g_merged_summary,
                                       valiadte_image_merged=validate_image_summary,
                                       weight_org_fig=weight_org_fig_summary,
                                       weight_net_fig=weight_net_fig_summary,
                                       weight_loss_fig=weight_loss_fig_summary)

        debug_handle = DebugHandle(ebdd_weights_org=ebdd_weights_org,
                                   ebdd_weights_net=ebdd_weights_for_net,
                                   ebdd_weights_loss=ebdd_weights_for_loss,
                                   )

        # those operations will be shared, so we need
        # to make them visible globally
        setattr(self, "input_handle", input_handle)
        setattr(self, "loss_handle", loss_handle)
        setattr(self, "eval_handle", eval_handle)
        setattr(self, "summary_handle", summary_handle)
        setattr(self, "debug_handle", debug_handle)



    def emdd_difference_checker(self):
        return 1

    def register_session(self, sess):
        self.sess = sess

    def retrieve_trainable_vars(self, freeze_encoder=False, freeze_decoder=False,freeze_discriminator=False,freeze_ebdd_weights=False):
        t_vars = tf.trainable_variables()

        dis_vars = [var for var in t_vars if 'dis_' in var.name]
        # gen_vars = [var for var in t_vars if 'gen_' in var.name]
        gen_enc_vals =  [var for var in t_vars if 'gen_enc' in var.name]
        gen_dec_vals =  [var for var in t_vars if 'gen_dec' in var.name]
        gen_ebdd_weights_vals = [var for var in t_vars if 'gen_ebdd_weights' in var.name]



        gen_vars_1 = list()
        dis_vars_1 = list()

        if freeze_encoder==0:
            print("Encoder Not Frozen")
            gen_vars_1.extend(gen_enc_vals)
        else:
            print("Encoder IS Frozen")

        if freeze_decoder==0:
            print("Decoder Not Frozen")
            gen_vars_1.extend(gen_dec_vals)
        else:
            print("Decoder IS Frozen")


        if freeze_discriminator==0:
            print("Discriminator Not Frozen")
            dis_vars_1.extend(dis_vars)
        else:
            print("Discriminator IS Frozen")

        if freeze_ebdd_weights==0:
            print("ebdd Weight Not Frozen")
            gen_vars_1.extend(gen_ebdd_weights_vals)
        else:
            print("Embedding weights IS Frozen")


        return gen_vars_1, dis_vars_1, t_vars

    def retrieve_generator_vars(self):
        all_vars = tf.global_variables()
        generate_vars = [var for var in all_vars if 'ebdd' in var.name or "g_" in var.name]
        return generate_vars

    def retrieve_handles(self):
        input_handle = getattr(self, "input_handle")
        loss_handle = getattr(self, "loss_handle")
        eval_handle = getattr(self, "eval_handle")
        summary_handle = getattr(self, "summary_handle")
        debug_handle = getattr(self,"debug_handle")

        return input_handle, loss_handle, eval_handle, summary_handle,debug_handle

    def get_model_id_and_dir(self):
        model_id = "experiment_%s_batch_%d_mode_%d" % (self.experiment_id, self.batch_size,self.running_mode)
        model_ckpt_dir = os.path.join(self.checkpoint_dir, model_id)
        model_log_dir = os.path.join(self.log_dir, model_id)
        model_sample_dir = os.path.join(self.sample_dir, model_id)
        model_weight_bar_dir = os.path.join(self.weight_bar_dir,model_id)
        return model_id,model_ckpt_dir,model_log_dir,model_sample_dir,model_weight_bar_dir

    def checkpoint(self, saver):
        model_name = "unet.model"
        saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=self.counter)

    def restore_model(self, saver, model_dir):

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("restored model %s" % model_dir)
        else:
            print("fail to restore model %s" % model_dir)

    def generate_fake_samples(self,input_images, ebdd_weights):
        input_handle, loss_handle, eval_handle, summary_handle,_ = self.retrieve_handles()
        fake_images, real_images, \
        d_loss, g_loss, l1_loss = self.sess.run([eval_handle.generator,
                                                 eval_handle.target,
                                                 loss_handle.d_loss,
                                                 loss_handle.g_loss,
                                                 loss_handle.l1_loss],
                                                feed_dict={
                                                    input_handle.real_data: input_images,
                                                    input_handle.ebdd_weights_static: ebdd_weights,
                                                })


        return fake_images, real_images, d_loss, g_loss, l1_loss

    def validate_model(self,val_iter, epoch):
        #input_handle, _, _, summary_handle,_ = self.retrieve_handles()
        labels, images = next(val_iter)
        labels = self.dense_to_one_hot(labels, self.font_num_for_train)
        fake_imgs, real_imgs, d_loss, g_loss, l1_loss = self.generate_fake_samples(images, labels)

        current_time=time.strftime('%Y-%m-%d @ %H:%M:%S',time.localtime())
        print("Time:%s,Sample:%.5f,g:%.5f,l1:%.5f" % (current_time,d_loss, g_loss, l1_loss))
        print(self.print_separater)

        merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
        merged_real_images = merge(scale_back(real_imgs), [self.batch_size, 1])
        merged_pair = np.concatenate([merged_real_images, merged_fake_images], axis=1)



        sample_img_path = os.path.join(self.sample_dir, "sample_%02d_%04d.png" % (epoch, self.counter))
        misc.imsave(sample_img_path, merged_pair)



        return merged_pair





    def export_generator(self, save_dir, model_dir, model_name="gen_model"):
        saver = tf.train.Saver()
        self.restore_model(saver, model_dir)

        gen_saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        gen_saver.save(self.sess, os.path.join(save_dir, model_name), global_step=0)

    def infer(self, source_obj, ebdd_weights, model_dir, save_dir):
        source_provider = InjectDataProvider(source_obj)

        if isinstance(ebdd_weights, int) or len(ebdd_weights) == 1:
            ebdd_id = ebdd_weights if isinstance(ebdd_weights, int) else ebdd_weights[0]
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, ebdd_id)
        else:
            source_iter = source_provider.get_random_embedding_iter(self.batch_size, ebdd_weights)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)

        count = 0
        batch_buffer = list()
        for labels, source_imgs in source_iter:
            fake_imgs = self.generate_fake_samples(source_imgs, labels)[0]
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)

    def interpolate(self, source_obj, between, model_dir, save_dir, steps):
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)
        # new interpolated dimension
        new_x_dim = steps + 1
        alphas = np.linspace(0.0, 1.0, new_x_dim)

        def _interpolate_tensor(_tensor):
            """
            Compute the interpolated tensor here
            """

            x = _tensor[between[0]]
            y = _tensor[between[1]]

            interpolated = list()
            for alpha in alphas:
                interpolated.append(x * (1. - alpha) + alpha * y)

            interpolated = np.asarray(interpolated, dtype=np.float32)
            return interpolated

        def filter_ebdd_vars(var):
            var_name = var.name
            if var_name.find("ebdd") != -1:
                return True
            if var_name.find("inst_norm/shift") != -1 or var_name.find("inst_norm/scale") != -1:
                return True
            return False

        ebdd_vars = filter(filter_ebdd_vars, tf.trainable_variables())
        # here comes the hack, we overwrite the original tensor
        # with interpolated ones. Note, the shape might differ

        # this is to restore the ebdd at the end
        ebdd_snapshot = list()
        for e_var in ebdd_vars:
            val = e_var.eval(session=self.sess)
            ebdd_snapshot.append((e_var, val))
            t = _interpolate_tensor(val)
            op = tf.assign(e_var, t, validate_shape=False)
            print("overwrite %s tensor" % e_var.name, "old_shape ->", e_var.get_shape(), "new shape ->", t.shape)
            self.sess.run(op)

        source_provider = InjectDataProvider(source_obj)
        input_handle, _, eval_handle, _,_ = self.retrieve_handles()
        for step_idx in range(len(alphas)):
            alpha = alphas[step_idx]
            print("interpolate %d -> %.4f + %d -> %.4f" % (between[0], 1. - alpha, between[1], alpha))
            source_iter = source_provider.get_single_embedding_iter(self.batch_size, 0)
            batch_buffer = list()
            count = 0
            for _, source_imgs in source_iter:
                count += 1
                labels = [step_idx] * self.batch_size
                generated, = self.sess.run([eval_handle.generator],
                                           feed_dict={
                                               input_handle.real_data: source_imgs,
                                               input_handle.ebdd_weights: labels
                                           })
                merged_fake_images = merge(scale_back(generated), [self.batch_size, 1])
                batch_buffer.append(merged_fake_images)
            if len(batch_buffer):
                save_concat_images(batch_buffer,
                                   os.path.join(save_dir, "frame_%02d_%02d_step_%02d.png" % (
                                       between[0], between[1], step_idx)))
        # restore the ebdd variables
        print("restore ebdd values")
        for var, val in ebdd_snapshot:
            op = tf.assign(var, val, validate_shape=False)
            self.sess.run(op)


    def train_procedures(self):
        g_vars, d_vars, all_vars = self.retrieve_trainable_vars(freeze_encoder=self.freeze_encoder,
                                                      freeze_decoder=self.freeze_decoder,
                                                      freeze_discriminator=self.freeze_discriminator,
                                                      freeze_ebdd_weights=self.freeze_ebdd_weights)
        input_handle, loss_handle, _, summary_handle,debug_handle = self.retrieve_handles()

        if not self.sess:
            raise Exception("no session registered")

        learning_rate = tf.placeholder(tf.float32, name="learning_rate")


        if self.optimization_method=='adam':
            d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.d_loss, var_list=d_vars)
            g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(loss_handle.g_loss, var_list=g_vars)
        elif self.optimization_method=='gradient_descent':
            d_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_handle.d_loss, var_list=d_vars)
            g_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_handle.g_loss, var_list=g_vars)


        tf.global_variables_initializer().run()

        # filter by one type of labels
        if self.running_mode==0:
            full_train_mark=True
        else:
            full_train_mark=False
        data_provider = TrainDataProvider(self.data_dir, train_name=self.train_obj_name, val_name=self.val_obj_name,
                                          filter_by=self.fine_tune,full_train_mark=full_train_mark,sub_train_set_num=self.sub_train_set_num)
        self.epoch = data_provider.get_total_epoch_num(self.itrs,self.batch_size)

        total_batches = data_provider.compute_total_batch_num(self.batch_size)
        val_batch_iter = data_provider.get_val_iter(self.batch_size)

        saver = tf.train.Saver(max_to_keep=1,var_list=all_vars)




        # restore model from previous fully trained one
        if self.running_mode==1 or self.running_mode == 2:
            self.restore_model(saver, self.base_trained_model_dir)


        # restore model from previous trained one (identical mrunning mode)
        if self.resume_training==1:
            self.restore_model(saver, self.checkpoint_dir)


        summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        current_lr = self.lr

        start_time = time.time()
        print(self.print_separater)
        print(self.print_separater)
        print(self.print_separater)
        for ei in range(self.epoch):
            # this_itr_start=time.time()
            train_batch_iter = data_provider.get_train_iter(self.batch_size)

            if (ei + 1) % self.schedule == 0:
                update_lr = current_lr / 2.0
                # minimum learning rate guarantee
                update_lr = max(update_lr, 0.0002)
                print("decay learning rate from %.5f to %.5f" % (current_lr, update_lr))
                current_lr = update_lr

            for bid, batch in enumerate(train_batch_iter):
                self.counter += 1

                this_itr_start=time.time()
                labels, batch_images = batch
                labels = self.dense_to_one_hot(labels,self.font_num_for_train)




                # Optimize D
                _, batch_d_loss, d_summary = self.sess.run([d_optimizer, loss_handle.d_loss, summary_handle.d_merged],
                                                           feed_dict=self.feed_dictionary_generation_for_d(batch_images=batch_images,
                                                                                                           labels=labels,
                                                                                                           current_lr=current_lr,
                                                                                                           learning_rate=learning_rate))

                # Optimize G
                _, batch_g_loss = self.sess.run([g_optimizer, loss_handle.g_loss],
                                                feed_dict=self.feed_dictionary_generation_for_g(batch_images=batch_images,
                                                                                                labels=labels,
                                                                                                current_lr=current_lr,
                                                                                                learning_rate=learning_rate))



                # magic move to Optimize G again
                # according to https://github.com/carpedm20/DCGAN-tensorflow
                # collect all the losses along the way
                _, batch_g_loss, category_loss, real_category_loss, fake_category_loss, \
                cheat_loss, \
                const_loss, l1_loss, tv_loss, ebdd_weight_loss, \
                g_summary = self.sess.run([g_optimizer,
                                           loss_handle.g_loss,
                                           loss_handle.category_loss,
                                           loss_handle.real_category_loss,
                                           loss_handle.fake_category_loss,
                                           loss_handle.cheat_loss,
                                           loss_handle.const_loss,
                                           loss_handle.l1_loss,
                                           loss_handle.tv_loss,
                                           loss_handle.ebdd_weight_loss,
                                           summary_handle.g_merged],
                                          feed_dict=self.feed_dictionary_generation_for_g(batch_images=batch_images,
                                                                                          labels=labels,
                                                                                          current_lr=current_lr,
                                                                                          learning_rate=learning_rate))



                passed_full = time.time() - start_time
                passed_itr = time.time() - this_itr_start
                current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
                log_format = "Time:%s,Epoch/Itr:[%d/%d]/[%d/%d];\n" +\
                             "dis:%.2f,category:%.2f,category_real:%.2f,category_fake:%.2f;\n" +\
                             "gen:%.2f,l1:%.2f,const:%.2f,cheat:%.2f,g_fake_category:%.2f;\nebdd_weight_loss:%.2f;\n" +\
                             "from_start:%4.2fhrs, this_itr:%4.2fsecs"

                print(log_format % (current_time,ei,self.epoch, bid, total_batches,
                                    batch_d_loss, category_loss, real_category_loss, fake_category_loss,
                                    batch_g_loss,l1_loss,const_loss,cheat_loss,fake_category_loss,ebdd_weight_loss,
                                    passed_full / 3600,passed_itr))
                print(self.print_separater)

                summary_writer.add_summary(d_summary, self.counter)
                summary_writer.add_summary(g_summary, self.counter)
                summary_writer.flush()





                if self.counter % self.sample_steps == 0:
                    # sample the current model states with val data
                    merged_pair = self.validate_model(val_batch_iter, ei)
                    summary_image = self.sess.run(summary_handle.valiadte_image_merged,feed_dict={
                                                                                    input_handle.validate_image: np.reshape(merged_pair,
                                                                                    (1, merged_pair.shape[0], merged_pair.shape[1], merged_pair.shape[2]))})
                    summary_writer.add_summary(summary_image,self.counter)
                    summary_writer.flush()



                    if self.freeze_ebdd_weights==0:
                        weight_bar_path_org = self.weight_plot_and_save(prefix="org_",weight_to_plot=debug_handle.ebdd_weights_org.eval()[0, :], epoch=ei)
                        weight_bar_org = self.png_read(weight_bar_path_org)
                        weight_org_fig_out = self.sess.run(summary_handle.weight_org_fig, feed_dict={input_handle.weight_org_fig_placeholder: weight_bar_org})
                        summary_writer.add_summary(weight_org_fig_out, self.counter)

                        weight_bar_path_net = self.weight_plot_and_save(prefix="net_",weight_to_plot=debug_handle.ebdd_weights_net.eval()[0, :], epoch=ei)
                        weight_bar_net = self.png_read(weight_bar_path_net)
                        weight_net_fig_out = self.sess.run(summary_handle.weight_net_fig, feed_dict={input_handle.weight_net_fig_placeholder: weight_bar_net})
                        summary_writer.add_summary(weight_net_fig_out, self.counter)

                        weight_bar_path_loss = self.weight_plot_and_save(prefix="loss_",weight_to_plot=debug_handle.ebdd_weights_loss.eval()[0, :], epoch=ei)
                        weight_bar_loss = self.png_read(weight_bar_path_loss)
                        weight_loss_fig_out = self.sess.run(summary_handle.weight_loss_fig, feed_dict={input_handle.weight_loss_fig_placeholder: weight_bar_loss})
                        summary_writer.add_summary(weight_loss_fig_out, self.counter)

                        summary_writer.flush()


                if self.counter % self.checkpoint_steps == 0:
                    current_time = time.strftime('%Y-%m-%d @ %H:%M:%S', time.localtime())
                    print("Time:%s,Checkpoint:SaveCheckpoint@step:%d" % (current_time,self.counter))
                    print(self.print_separater)
                    self.checkpoint(saver)


            # save the last checkpoint of current epoch
            print(self.print_separater)
            print("CheckpointSaved:LastCheckpointStep@%d of epoch:%d" % (self.counter, ei))
            print(self.print_separater)
            self.checkpoint(saver)




    def dense_to_one_hot(self,input_label,label_length):
        output_one_hot_label=np.zeros((len(input_label),label_length),dtype=np.float32)
        output_one_hot_label[np.arange(len(input_label)),input_label]=1
        return output_one_hot_label


    def weight_plot_and_save(self,prefix,weight_to_plot,epoch):
        y_pos=np.arange(len(weight_to_plot))
        plt.figure(figsize=(12,9),dpi=100)

        plt.bar(y_pos,weight_to_plot,align='center',alpha=0.5)
        plt.xticks(y_pos)

        fig_save_path = os.path.join(self.weight_bar_dir, prefix+"weight_bar_%02d_%04d.png" % (epoch, self.counter))

        plt.savefig(fig_save_path,format='png')

        plt.close()

        return fig_save_path

    def png_read(self,path):
        image = img.imread(path)
        image_shape=image.shape
        shape0=int(image_shape[0])
        shape1 = int(image_shape[1])
        shape2 = int(image_shape[2])
        image=image.reshape(1,shape0,shape1,shape2)
        return image




    def feed_dictionary_generation_for_d(self,batch_images,labels,current_lr,learning_rate):

        input_handle, _, _, _, _ = self.retrieve_handles()
        real_data = input_handle.real_data
        ebdd_weights_static = input_handle.ebdd_weights_static
        if self.freeze_ebdd_weights == True:
            output_dict={
                real_data: batch_images,
                ebdd_weights_static: labels,
                learning_rate: current_lr
            }
        else:
            output_dict = {
                real_data: batch_images,
                learning_rate: current_lr
            }

        return output_dict

    def feed_dictionary_generation_for_g(self,batch_images,labels,current_lr,learning_rate):
        input_handle, _, _, _, _ = self.retrieve_handles()
        real_data = input_handle.real_data
        ebdd_weights_static = input_handle.ebdd_weights_static
        targeted_label = input_handle.targeted_label
        if self.freeze_ebdd_weights == True:
            output_dict = {
                real_data: batch_images,
                ebdd_weights_static: labels,
                learning_rate: current_lr,
            }
        else:
            output_dict = {
                real_data: batch_images,
                learning_rate: current_lr,
                targeted_label: labels
            }


        return output_dict

