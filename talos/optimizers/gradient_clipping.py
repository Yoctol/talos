import tensorflow as tf


class GradientClipping(tf.train.Optimizer):

    _ALLOWED_CLIP_BY = {'value', 'norm'}

    def __init__(
            self,
            optimizer,
            value: float,
            clip_by: str = 'value',
            use_locking: bool = False,
            name: str = 'GradientClipping',
        ):
        super().__init__(use_locking, name)
        self.optimizer = optimizer
        if clip_by not in self._ALLOWED_CLIP_BY:
            raise ValueError(f"`clip_by` should be in {self._ALLOWED_CLIP_BY}! Found {clip_by}")
        if value <= 0.:
            raise ValueError("`value` should > 0.!")

        self.value = value
        self.value_tensor = tf.convert_to_tensor(value)
        self.clip_by = clip_by

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        processed_gvs = [
            (self._process_grad(g), v) for g, v in grads_and_vars
            if g is not None
        ]
        return self.optimizer.apply_gradients(
            processed_gvs,
            global_step=global_step,
            name=name,
        )

    def _process_grad(self, grad):
        value = tf.cast(self.value_tensor, grad.dtype.base_dtype)
        if self.clip_by == 'value':
            return tf.clip_by_value(grad, -value, value)
        elif self.clip_by == 'norm':
            return tf.clip_by_norm(grad, value)
        else:
            raise AssertionError("Invalid `clip_by` should be raised in `__init__`!")
