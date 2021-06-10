import numpy as np
import caffe


class ComputeH(caffe.Layer):
    def __init__(self, p_object, *args, **kwargs):
        super(ComputeH, self).__init__(p_object, *args, **kwargs)
        self.n_classes = -1

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need (only) one input to compute H matrix.")

        params = eval(self.param_str)
        if 'n_classes' in params:
            self.n_classes = int(params['n_classes'])
        else:
            raise Exception('The number of classes (n_classes) must be specified.')

    def reshape(self, bottom, top):
        top[0].reshape(1, 1, self.n_classes, self.n_classes)

    def forward(self, bottom, top):
        classes, cls_num = np.unique(bottom[0].data, return_counts=True)

#        if np.size(classes) != self.n_classes or self.n_classes == -1:
        if self.n_classes == -1:
            raise Exception("Invalid number of classes")

        missing = []
        if np.size(classes) < self.n_classes:
            missing = list(set(np.arange(self.n_classes)).difference(classes))

        cls_num = cls_num.astype(float)
        cls_num = cls_num.max() / cls_num

        weights = cls_num / np.sum(cls_num)

        if (len(missing) > 0):
          for i in missing:
            weights = np.insert(weights, i, 0)

        top[0].data[...] = np.diag(weights)

    def backward(self, top, propagate_down, bottom):
        pass