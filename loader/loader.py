import numpy as np
import cv2
import os
import spacy
import keras


class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """

    def __init__(
        self,
        data,
        config,
        shuffle=True,
        train_mode=True,
    ):
        self.shuffle = shuffle
        self.data = data
        self.config = config
        self.train_mode = train_mode
        self.batch_size = config.batch_size
        self.embed = spacy.load(config.word_embed)
        self.input_shape = (config.input_size, config.input_size)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)
        self.group()

    def size(self):
        return len(self.data)

    def group(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # divide into groups, one group = one batch
        self.groups = [[self.data[x % len(self.data)] for x in range(i, i + self.batch_size)] for i in range(0, len(self.data), self.batch_size)]

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def get_batch(self, datas):
        size = len(datas)
        image_data = np.empty([size, self.input_shape[0], self.input_shape[1], 3])
        word_data = np.empty([size, self.config.word_len, self.config.embed_dim])
        seg_data = np.empty([size, self.input_shape[0] // self.config.seg_out_stride,
                             self.input_shape[1] // self.config.seg_out_stride, 1])
        for (i, data) in enumerate(datas):
            image, word_vec, seg_map = get_random_data(data,
                                                       self.input_shape,
                                                       self.embed,
                                                       self.config,
                                                       train_mode=self.train_mode)
            word_data[i] = word_vec
            image_data[i] = image
            seg_data[i] = seg_map

        return image_data, word_data, seg_data

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """

        group = self.groups[index]
        image_data, word_data, seg_data = self.get_batch(group)
        # print(np.shape(inputs))
        return [image_data, word_data, seg_data], np.zeros(self.batch_size)


def qlist_to_vec(max_length, q_list, embed, emb_size=300):
    '''
    note: 2018.10.3
    use for process sentences
    '''
    q_list = q_list.split()
    glove_matrix = np.zeros((max_length, emb_size), dtype=float)
    q_len = min(max_length, len(q_list))

    for i in range(q_len):
        glove_matrix[i, :] = embed(u'%s' % q_list[i]).vector

    return glove_matrix


def get_random_data(ref, input_shape, embed, config, train_mode=True, max_boxes=1):
    '''random preprocessing for real-time data augmentation'''
    SEG_DIR = config.seg_gt_path
    h, w = input_shape
    seg_id = ref['segment_id']
    # box = ref['bbox']
    sentences = ref['sentences']
    choose_index = np.random.choice(ref['sentences_num'])

    word_vec = qlist_to_vec(config.word_len, sentences[choose_index]['sent'], embed)
    image = cv2.imread(os.path.join(config.image_path, ref['img_name']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape
    if not train_mode:
        ori_image = image.copy()

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image_data = np.full((w, h, 3), (0.5, 0.5, 0.5))
    image_data[dy:dy+nh, dx:dx+nw, :] = image / 255.

    seg_map = cv2.imread(os.path.join(SEG_DIR, str(seg_id)+'.png'), flags=cv2.IMREAD_GRAYSCALE)
    if train_mode:
        seg_map = cv2.resize(seg_map, (nw, nh), interpolation=cv2.INTER_CUBIC)
        seg_map_data = np.zeros((w, h))
        seg_map_data[dy:dy+nh, dx:dx+nw] = seg_map / 255
        seg_map_data = cv2.resize(seg_map_data, (
            w // config.seg_out_stride, h // config.seg_out_stride), interpolation=cv2.INTER_NEAREST)
        seg_map_data = seg_map_data[:, :, None]

    if not train_mode:
        word_vec = [qlist_to_vec(config.word_len, sent['sent'], embed) for sent in sentences]
        return image_data, word_vec, ori_image, sentences, seg_map[:, :, None]
    return image_data, word_vec, seg_map_data
