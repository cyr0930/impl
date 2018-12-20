from tensorflow.python.keras import backend as K


def total_loss(cls_output_len, reg_output_len):
    def total_loss_closure(y_true, y_pred):
        num_of_outputs = cls_output_len + reg_output_len
        a = K.reshape(y_true, (-1, num_of_outputs))
        b = K.reshape(y_pred, (-1, num_of_outputs))
        cls_loss = K.sum(K.binary_crossentropy(a[:, :cls_output_len], b[:, :cls_output_len]))
        diff = K.abs(a[:, cls_output_len:] - b[:, cls_output_len:])
        reg_loss1 = K.sum(K.square(diff * K.cast(K.less(diff, 1), K.floatx())) * 0.5)
        reg_loss2 = K.sum((diff - 0.5) * K.cast(K.greater_equal(diff, 1), K.floatx()))
        reg_loss = reg_loss1 + reg_loss2
        return cls_loss + reg_loss
    return total_loss_closure
