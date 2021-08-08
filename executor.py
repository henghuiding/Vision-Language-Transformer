import json
import os
from abc import abstractmethod
from datetime import datetime

import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam

from callbacks.common import RedirectModel
from callbacks.eval import Evaluate
from callbacks.learning_scheduler import LearningRateScheduler, lr_step_decay
from loader.loader import Generator
from model.vlt_model import yolo_body, yolo_loss


class Executor(object):
    def __init__(self, config, GPUS=1, debug=False):
        # settings
        self.config = config
        self.debug = debug
        self.GPUS = GPUS
        self.input_shape = (self.config.input_size, self.config.input_size, 3)  # multiple of 32, hw
        self.word_len = self.config.word_len
        self.embed_dim = self.config.embed_dim
        self.seg_out_stride = self.config.seg_out_stride
        self.start_epoch = self.config.start_epoch
        self.n_freeze = 185 + 12

        # data init
        self.dataset = {}
        self.dataset_len = {}
        self.load_data()

        # create model
        self.yolo_model, self.yolo_body, self.yolo_body_single = self.create_model()

        # call_back_init
        self.callbacks = self.build_callbacks()

    def create_model(self):
        print('Creating model...')
        K.clear_session()  # get a new session
        image_input = Input(shape=(self.input_shape))
        q_input = Input(shape=[self.word_len, self.embed_dim], name='q_input')
        h, w, _ = self.input_shape

        seg_gt = Input(shape=(h//self.seg_out_stride, w//self.seg_out_stride, 1))
        # mask_size = self.config.input_size // self.config.seg_out_stride

        model_body = yolo_body(image_input, q_input, self.config)
        print('Loading model...')
        self.load_model(model_body)

        if self.GPUS > 1:
            print("Using {} GPUs".format(self.GPUS))
            model_body_para = keras.utils.multi_gpu_model(model_body, gpus=self.GPUS)
        else:
            print("Using SINGLE GPU Only")
            model_body_para = model_body

        model_loss = Lambda(yolo_loss,
                            output_shape=(1,),
                            name='yolo_loss',
                            arguments={'batch_size': self.config.batch_size})(
            [model_body_para.output, seg_gt])

        model = Model([model_body_para.input[0],
                       model_body_para.input[1],
                       seg_gt], model_loss)
        print('Model created.')

        return model, model_body_para, model_body

    def load_dataset(self, split):
        with open(self.config[split], 'rb') as f:
            data_lines = json.load(f)
        if self.debug:
            data_lines = data_lines[:50]
        set_num = len(data_lines)
        print('Dataset Loaded: %s,  Len: %d' % (split, set_num))
        return data_lines, set_num

    @abstractmethod
    def build_callbacks():
        pass

    @abstractmethod
    def load_model():
        pass

    @abstractmethod
    def load_data():
        pass


class Trainer(Executor):
    def __init__(self, config, log_path, verbose=False, **kwargs):
        self.load_path = config.pretrained_weights
        self.log_path = log_path
        self.verbose = verbose
        self.model_path = os.path.join(self.log_path, 'models')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        json.dump(config, open(os.path.join(self.model_path, 'config.json'), 'w'))

        timestr = datetime.now().strftime('%m_%d_%H_%M_%S')
        self.tb_path = os.path.join(self.log_path, timestr)
        if not os.path.exists(self.tb_path):
            os.makedirs(self.tb_path)
        json.dump(config, open(os.path.join(self.tb_path, 'config.json'), 'w'))

        super(Trainer, self).__init__(config, **kwargs)

    def load_model(self, model_body):
        path = self.config.pretrained_weights
        model_body.load_weights(path, by_name=True, skip_mismatch=True)
        print('Loading weights from {}.'.format(path))
        if self.config.free_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (self.n_freeze, len(model_body.layers) - 3)[self.config.free_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    def load_data(self):
        self.dataset['val'], self.dataset_len['val'] = self.load_dataset('evaluate_set')
        self.dataset['train'], self.dataset_len['train'] = self.load_dataset('train_set')
        self.train_generator = Generator(self.dataset['train'], self.config)
        # self.train_generator.__getitem__(1)

    def build_callbacks(self):
        call_backs = []
        logging = TensorBoard(log_dir=self.tb_path)
        call_backs.append(logging)

        model_evaluate = Evaluate(self.dataset['val'], self.config, tensorboard=logging)
        call_backs.append(RedirectModel(model_evaluate, self.yolo_body))

        checkpoint_map = ModelCheckpoint(self.log_path + '/models/best_map.h5',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         monitor="seg_iou",
                                         mode='max')
        call_backs.append(RedirectModel(checkpoint_map, self.yolo_body_single))

        lr_schedue = LearningRateScheduler(lr_step_decay(self.config.lr, self.config.steps),
                                           logging, verbose=1,
                                           init_epoch=self.config.start_epoch)
        call_backs.append(lr_schedue)

        return call_backs

    def train(self):
        # Yolo Compile
        print('Compiling model... ')
        self.yolo_model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred},
                                optimizer=Adam(lr=self.config.lr))

        if self.config.workers > 0:
            use_multiprocessing = True
        else:
            use_multiprocessing = False

        print('Starting training:')
        self.yolo_model.fit_generator(self.train_generator,
                                      callbacks=self.callbacks,
                                      epochs=self.config.epoches,
                                      initial_epoch=self.config.start_epoch,
                                      verbose=True,
                                      workers=self.config.workers,
                                      use_multiprocessing=use_multiprocessing,
                                      max_queue_size=self.config.max_queue_size
                                      )


class Tester(Executor):
    def __init__(self, config, **kwargs):
        super(Tester, self).__init__(config, **kwargs)

    def build_callbacks(self):
        self.evaluator = RedirectModel(Evaluate(self.dataset['val'], self.config, phase='test'), self.yolo_body)
        self.evaluator.on_train_begin()

    def load_data(self):
        self.dataset['val'], self.dataset_len['val'] = self.load_dataset('evaluate_set')

    def load_model(self, model_body):
        model_body.load_weights(self.config.evaluate_model, by_name=False, skip_mismatch=False)
        print('Load weights {}.'.format(self.config.evaluate_model))

    def eval(self):
        results = dict()
        self.evaluator.on_epoch_end(-1, results)
        seg_iou = results['seg_iou']
        seg_prec = results['seg_prec']
        # dump results to text file
        if not os.path.exists('result/'):
            os.mkdir('result/')
        from datetime import datetime
        timestr = datetime.now().strftime('%m_%d_%H_%M_%S')

        with open('result/result_%s.txt' % (timestr), 'w') as f_w:
            f_w.write('segmentation result:' + '\n')
            f_w.write('seg_iou: %.4f\n' % (seg_iou))
            for item in seg_prec:
                f_w.write('prec@%.2f: %.4f' % (item, seg_prec[item])+'\n')
            f_w.write('\n')


class Debugger(Executor):
    def __init__(self, config, **kwargs):
        self.config = config
        self.debug = True
        self.dataset, self.dataset_len = self.load_dataset('train_set')
        self.train_generator = Generator(self.dataset, self.config)
        self.train_generator.__getitem__(1)
        kwargs.update({'GPUS': 1, 'debug': True})
        super(Debugger, self).__init__(config, **kwargs)

    def build_callbacks(self):
        return None

    def load_data(self):
        return None

    def load_model(self, model_body):
        return None

    def run(self):
        self.yolo_model.summary()
        self.yolo_model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred},
                                optimizer=Adam(lr=self.config.lr))
