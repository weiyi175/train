import tensorflow as tf
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="CNN_BiLSTM")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma_variable: Optional[tf.Variable] = None, from_logits: bool = False, name: Optional[str] = "focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self._gamma_variable = gamma_variable
        self.from_logits = from_logits

    @property
    def gamma_variable(self):
        return self._gamma_variable

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if self.from_logits:
            pred_prob = tf.nn.sigmoid(y_pred)
        else:
            pred_prob = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        gamma = 0.0
        if self._gamma_variable is not None:
            try:
                gamma = tf.cast(self._gamma_variable.read_value(), pred_prob.dtype)
            except Exception:
                gamma = tf.cast(self._gamma_variable, pred_prob.dtype)
        p_t = tf.where(tf.equal(y_true, 1), pred_prob, 1 - pred_prob)
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        if self.from_logits:
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        else:
            bce = - (y_true * tf.math.log(pred_prob + 1e-7) + (1 - y_true) * tf.math.log(1 - pred_prob + 1e-7))
        loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "gamma_variable": None, "from_logits": self.from_logits})
        return config
