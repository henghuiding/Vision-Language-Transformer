from keras import layers as L
from keras_transformer import get_encoder_component, get_decoder_component
from keras_multi_head import MultiHeadAttention
from keras_pos_embd import TrigPosEmbedding


def ref_tf(encoder_input,
           decoder_input,
           feat_size,
           encoder_num=2,
           decoder_num=2,
           head_num=8,
           hidden_dim=256,
           num_query=32,
           attention_activation='relu',
           feed_forward_activation='relu',
           dropout_rate=0.1,
           trainable=True,
           balance=True):

    spatial_size = feat_size * feat_size

    encoder_embed = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Encoder-Embedding',
    )(encoder_input)
    encoded_layer = get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    decoder_embed = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Decoder-Embedding',
    )(decoder_input)
    decoded_layer = get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
        encoded_layer=encoded_layer,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )

    if balance:
        query_proj = L.Dense(hidden_dim, activation='relu')(decoder_input)
        output_proj = L.Dense(hidden_dim, activation='relu')(decoded_layer)
        output_proj = L.Concatenate()([output_proj, query_proj])
        output_proj = L.Dense(hidden_dim, activation='relu')(output_proj)
        query_confident = L.Dense(1, activation='sigmoid')(output_proj)

        weighted_output = L.Multiply()([query_confident, decoded_layer])
    else:
        weighted_output = decoded_layer

    output_layer = L.Dense(spatial_size, activation='relu')(weighted_output)
    output_layer = L.Reshape((num_query, feat_size, feat_size))(output_layer)
    output_layer = L.Permute((2, 3, 1))(output_layer)

    return output_layer


def lang_tf_enc(vision_input,
                lang_input,
                head_num=8,
                hidden_dim=256):
    decoder_embed_lang = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Fusion-Lang-Decoder-Embedding',
    )(lang_input)
    decoder_embed_vis = TrigPosEmbedding(
        mode=TrigPosEmbedding.MODE_ADD,
        name='Fusion-Vis-Decoder-Embedding',
    )(vision_input)
    q_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_vis)
    k_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_lang)
    v_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_lang)
    decoded_layer = MultiHeadAttention(head_num=head_num)(
        [q_inp, k_inp, v_inp])
    add_layer = L.Add(name='Fusion-Add')([decoded_layer, vision_input])

    return add_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 name_prefix=''):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name=name_prefix+'Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer


def get_decoders(decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 name_prefix=''):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name=name_prefix+'Decoder-%d' % (i + 1),
            input_layer=last_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer
