import keras
import numpy as np
from loader.loader import get_random_data
import cv2
import keras.backend as K
from matplotlib.pyplot import cm
import spacy
import progressbar


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        data,
        config,
        tensorboard=None,
        verbose=1,
        phase='train'
    ):
        self.val_data = data
        self.tensorboard = tensorboard
        self.verbose = verbose
        self.vis_id = [i for i in np.random.randint(0, len(data), 200)]
        self.batch_size = max(config.batch_size//2, 1)
        self.colors = np.array(cm.hsv(np.linspace(0, 1, 10)).tolist()) * 255
        self.input_shape = (config.input_size, config.input_size)  # multiple of 32, hw
        self.config = config
        self.word_embed = spacy.load(config.word_embed)
        self.word_len = config.word_len
        self.seg_min_overlap = config.segment_thresh
        if phase == 'test':
            self.log_images = config.log_images
            self.multi_thres = config.multi_thres
        else:
            self.log_images = 0
            self.multi_thres = False
        self.input_image_shape = K.placeholder(shape=(2,))
        self.sess = K.get_session()
        self.eval_save_images_id = [i for i in np.random.randint(0, len(self.val_data), 200)]
        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # run evaluation
        self.seg_iou, self.seg_prec = self.evaluate()

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.seg_iou
            summary_value.tag = "seg_iou"
            for item in self.seg_prec:
                summary_value = summary.value.add()
                summary_value.simple_value = self.seg_prec[item]
                summary_value.tag = "map@%.2f" % item
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['seg_iou'] = self.seg_iou
        logs['seg_prec'] = self.seg_prec

        if self.verbose == 1:
            print('seg_iou: {:.4f}'.format(self.seg_iou))

    def evaluate(self):
        prec_all = dict()
        img_id = 0
        iou_all = 0.

        test_batch_size = self.batch_size
        for start in progressbar.progressbar(range(0, len(self.val_data), test_batch_size), prefix='evaluation: '):
            end = start + test_batch_size
            batch_data = self.val_data[start:end]
            images = []
            images_ori = []
            files_id = []
            word_vecs = []
            sentences = []
            gt_segs = []

            for data in batch_data:
                image_data, word_vec, image, sentence, seg_map = get_random_data(data, self.input_shape,
                                                                                 self.word_embed, self.config,
                                                                                 train_mode=False)  # box is [1,5]
                sentences.extend(sentence)
                word_vecs.extend(word_vec)
                # evaluate each sentence corresponding to the same image
                for ___ in range(len(sentence)):
                    images.append(image_data)
                    images_ori.append(image)
                    files_id.append(img_id)
                    gt_segs.append(seg_map)
                    img_id += 1

            images = np.array(images)
            word_vecs = np.array(word_vecs)
            mask_outs = self.model.predict_on_batch([images, word_vecs])
            mask_outs = self.sigmoid_(mask_outs)  # logit to sigmoid
            batch_size = mask_outs.shape[0]
            for i in range(batch_size):
                ih = gt_segs[i].shape[0]
                iw = gt_segs[i].shape[1]
                w, h = self.input_shape
                scale = min(w / iw, h / ih)
                nw = int(iw * scale)
                nh = int(ih * scale)
                dx = (w - nw) // 2
                dy = (h - nh) // 2

                pred_seg = mask_outs[i, :, :, 0]

                pred_seg = cv2.resize(pred_seg, self.input_shape)
                pred_seg = pred_seg[dy:nh + dy, dx:nw + dx, ...]
                pred_seg = cv2.resize(pred_seg, (gt_segs[i].shape[1], gt_segs[i].shape[0]))
                pred_seg = np.reshape(pred_seg, [pred_seg.shape[0], pred_seg.shape[1], 1])

                # segmentation eval
                iou, prec = self.cal_seg_iou(gt_segs[i], pred_seg, self.seg_min_overlap)
                iou_all += iou
                for item in prec:
                    if prec_all.get(item):
                        prec_all[item] += prec[item]
                    else:
                        prec_all[item] = prec[item]

                if self.log_images:
                    sent = sentences[i]['sent']
                    wstatus1 = cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_pred.png', pred_seg * 255)
                    wstatus2 = cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_gt.png', gt_segs[i])
                    wstatus3 = cv2.imwrite('log/out_img/'+str(files_id[i])+'_'+sent+'_img.png', images[i][dy:nh + dy, dx:nw + dx, ...] * 255)
                    if not (wstatus1 and wstatus2 and wstatus3):
                        import pdb
                        pdb.set_trace()

        pred_seg = iou_all / img_id
        for item in prec_all:
            prec_all[item] /= img_id
        return pred_seg, prec_all

    def cal_seg_iou(self, gt, pred, thresh=0.5):
        t = np.array(pred > thresh)
        p = gt > 0.
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)

        prec = dict()
        thresholds = np.arange(0.5, 1, 0.05)
        for thresh in thresholds:
            prec[thresh] = float(iou > thresh)
        return iou, prec

    def sigmoid_(self, x):
        return (1. + 1e-9) / (1. + np.exp(-x) + 1e-9)
