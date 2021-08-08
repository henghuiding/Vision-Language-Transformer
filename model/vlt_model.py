"""YOLO_v3 Model Defined in Keras."""

import tensorflow as tf
from keras import backend as K
from keras import layers as L
from keras.models import Model

from model.language_backbone import build_nlp_model
from model.transfromer_model import ref_tf, lang_tf_enc
from model import utils as utils
from model import visual_backbone as V


def simple_fusion(F_v, f_q, dim=1024):
    """
    :param F_v: visual features (N,w,h,d)
    :param f_q: GRU output (N,d_q)
    :param dim: project dimensions default: 1024
    :return: F_m: simple fusion of Fv and fq (N,w,h,d)
    """
    out_size = K.int_shape(F_v)[1]
    # Fv project (use darknet_resblock get better performance)
    F_v_proj = V.darknet_resblock(F_v, dim//2)
    # fq_project
    f_q_proj = L.Dense(dim, activation='linear')(f_q)
    f_q_proj = L.advanced_activations.LeakyReLU(alpha=0.1)(
        L.normalization.BatchNormalization()(f_q_proj)
        )
    f_q_proj = L.Lambda(utils.expand_and_tile, arguments={'outsize': out_size})(f_q_proj)
    # simple elemwise multipy
    F_m = L.Multiply()([F_v_proj, f_q_proj])
    F_m = L.advanced_activations.LeakyReLU(alpha=0.1)(
        L.normalization.BatchNormalization()(F_m)
        )
    return F_m


def up_proj_cat_proj(x, y, di=256, do=256):
    x = L.UpSampling2D()(x)
    y = V.DarknetConv2D_BN_Leaky(di, (1, 1))(y)
    out = L.Concatenate()([x, y])
    out = V.DarknetConv2D_BN_Leaky(do, (1, 1))(out)
    return out


def proj_cat(x, y, di=256):
    if K.int_shape(x)[-1] > di:
        x = V.DarknetConv2D_BN_Leaky(di, (1, 1))(x)
    x = V.DarknetConv2D_BN_Leaky(di // 2, (1, 1))(x)
    x = V.DarknetConv2D_BN_Leaky(di, (3, 3))(x)
    out = L.Concatenate()([x, y])
    return out


def pool_proj_cat_proj(x, y, di=256, do=256):
    y = L.AveragePooling2D((2, 2))(y)
    y = V.DarknetConv2D_BN_Leaky(di, (1, 1))(y)
    out = L.Concatenate()([x, y])
    out = V.DarknetConv2D_BN_Leaky(do, (1, 1))(out)
    return out


def make_multitask_braches(Fv, fq, fq_word, config):
    # fq: bs, 1024
    # fq_word: bs, 15, 1024
    Fm = simple_fusion(Fv[0], fq, config.jemb_dim)  # 13, 13, 1024

    Fm_mid_query = up_proj_cat_proj(Fm, Fv[1], K.int_shape(Fv[1],)[-1], K.int_shape(Fm)[-1]//2)  # 26, 26, 512
    Fm_query = pool_proj_cat_proj(Fm_mid_query, Fv[2], K.int_shape(Fv[2])[-1], K.int_shape(Fm)[-1]//2)  # 26, 26, 512

    Fm_mid_tf = proj_cat(Fm_query, Fm_mid_query, K.int_shape(Fm)[-1]//2)  # 26, 26, 1024
    F_tf = up_proj_cat_proj(Fm, Fm_mid_tf, K.int_shape(Fm)[-1] // 2)

    F_tf = V.DarknetConv2D_BN_Leaky(config.hidden_dim, (1, 1))(F_tf)

    # Fm_query:  bs, Hm, Wm, C  (None, 26, 26, 512)
    # Fm_top_tf :  bs, Hc, Wc, C  (None, 26, 26, 512)
    query_out = vlt_querynet(Fm_query, config)
    mask_out = vlt_transformer(F_tf, fq_word, query_out, config)
    mask_out = vlt_postproc(mask_out, Fm_query, config)

    return mask_out


def vlt_transformer(F_tf, fq_word, query_out, config):
    # F_tf:       (None, 26, 26, 512)
    # query_out:  (None, 26, 26,  Nq)
    flatten_length = K.int_shape(F_tf)[1] * K.int_shape(F_tf)[2]
    F_tf_flat = L.Reshape([flatten_length, config.hidden_dim])(F_tf)                      # (None, 676, 512)

    query_flat = L.Reshape([flatten_length, config.num_query])(query_out)                 # (None, 676, Nq)
    query_flat = L.Permute((2, 1), input_shape=(676, config.num_query))(query_flat)       # (None, Nq, 676)
    query_flat = L.Dense(config.hidden_dim, activation='relu')(query_flat)                # (None, Nq, 512)

    lang_feat = L.Dense(config.hidden_dim, activation='relu')(fq_word)                    # (None, Nq, 512)

    query_input = lang_tf_enc(vision_input=query_flat,
                              lang_input=lang_feat,
                              hidden_dim=config.transformer_hidden_dim,
                              head_num=config.transformer_head_num)

    tf_out = ref_tf(encoder_input=F_tf_flat,
                    decoder_input=query_input,
                    feat_size=26,
                    encoder_num=config.transformer_encoder_num,
                    decoder_num=config.transformer_decoder_num,
                    hidden_dim=config.transformer_hidden_dim,
                    head_num=config.transformer_head_num,
                    num_query=config.num_query)

    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)

    return tf_out  # (None, 26, 26, 1)


def vlt_postproc(tf_out, Fm_query, config):
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    # if config.concate_lower_feat:
    #     tf_out = L.Concatenate()([tf_out, Fm_query])
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    if config.seg_out_stride <= 8:
        tf_out = L.UpSampling2D()(tf_out)
    tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    if config.seg_out_stride <= 4:
        tf_out = L.UpSampling2D()(tf_out)
        tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    if config.seg_out_stride <= 2:
        tf_out = L.UpSampling2D()(tf_out)
        tf_out = V.DarknetConv2D_BN_Leaky(config.hidden_dim, [3, 3])(tf_out)
    tf_out = V.DarknetConv2D(1, [3, 3])(tf_out)

    return tf_out


def vlt_querynet(x, config):
    x = L.Lambda(utils.concat_coord)(x)
    x = V.DarknetConv2D_BN_Leaky(K.int_shape(x)[-1], (3, 3))(x)
    x = V.DarknetConv2D_BN_Leaky(K.int_shape(x)[-1], (3, 3))(x)
    x = V.DarknetConv2D_BN_Leaky(K.int_shape(x)[-1], (3, 3))(x)
    x = V.DarknetConv2D(config.num_query, (1, 1))(x)
    return x  # (bs, H, W, n_query)  (None, 52, 52, config.num_query)


def yolo_body(inputs, q_input, config):
    """

    :param inputs:  image
    :param q_input:  word embeding
    :return:  regresion , attention map
    """
    """Create Multi-Modal YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, V.darknet_body(inputs))
    Fv = [darknet.output, darknet.layers[152].output, darknet.layers[92].output]

    fq, fq_word = build_nlp_model(q_input=q_input,
                                  rnn_dim=config.rnn_hidden_size,
                                  bidirection=config.rnn_bidirectional,
                                  dropout=config.rnn_drop_out,
                                  lang_att=config.lang_att,
                                  return_raw=True)

    mask_out = make_multitask_braches(Fv, fq, fq_word, config)

    return Model([inputs, q_input], [mask_out])


def yolo_loss(args,
              batch_size,
              print_loss=False):

    mask_out = args[0]
    mask_gt = args[1]

    loss = 0
    m = K.shape(mask_out)[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(mask_out))

    mask_loss = K.binary_crossentropy(mask_gt, mask_out, from_logits=True)
    mask_loss = K.sum(mask_loss) / mf

    loss += mask_loss

    if print_loss:
        loss = tf.Print(loss, ['mask: ', mask_loss])

    return K.expand_dims(loss, axis=0)
