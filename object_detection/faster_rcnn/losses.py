from tensorflow.python.keras import backend as K


def total_loss(num_of_outputs):
    def total_loss_closure(y_true, y_pred):
        K.reshape(y_true, (-1, num_of_outputs))
        K.reshape(y_pred, (-1, num_of_outputs))
        K.binary_crossentropy(y_true[:, :, :, :9], y_pred[:, :, :, :9])
        return K.square(y_pred[:, :, :, :1] - y_true[:, :, :, :1])
    return total_loss_closure
