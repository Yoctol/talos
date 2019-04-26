import numpy as np
import tensorflow as tf

from talos.layers import Layer


_LARGE_BIAS = 1e4


class RelativeAttentionCell(Layer):
    """Layer to perform attention with relative positional encoding on a single block.

    ref: https://arxiv.org/pdf/1901.02860.pdf

    Arguments:
        units: Positive integer, dimensionality of each heads.
        output_dim: Positive integer, dimensionality of the output space.
        heads: Positive integer, number of heads for attention.
            Default: 1.
        use_forward_mask: Bool, whether to mask out future information.
            Default: False.
    """

    def __init__(
            self,
            units: int,
            output_dim: int,
            heads: int = 1,
            use_forward_mask: bool = False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.units = units
        self.output_dim = output_dim
        self.heads = heads
        self.use_forward_mask = use_forward_mask

        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.supports_masking = True

        self._computed_rel = {}
        self._computed_triu = {}

    def build(self, input_shape: tf.TensorShape):
        fan_in = input_shape[2].value
        self.query_W, self.key_W, self.rel_W, self.value_W = [
            self.add_weight(
                name=name,
                shape=[fan_in, self.heads, self.units],
                initializer=self._get_glorot_uniform_initializer(fan_in, self.units),
            )
            for name in ['query_kernel', 'key_kernel', 'rel_kernel', 'value_kernel']
        ]
        self.u, self.v = [
            self.add_weight(
                name=name,
                shape=[self.heads, self.units],
                initializer=tf.truncated_normal_initializer(stddev=np.sqrt(1. / self.units)),
            )
            for name in ['u_vector', 'v_vector']
        ]
        self.output_W = self.add_weight(
            name='output_kernel',
            shape=[self.heads, self.units, self.output_dim],
            initializer=self._get_glorot_uniform_initializer(
                fan_in=self.heads * self.units,
                fan_out=self.output_dim,
            ),
        )
        self.input_spec.axes = {2: fan_in}
        self.built = True

    def _get_glorot_uniform_initializer(self, fan_in, fan_out):
        limit = np.sqrt(6. / (fan_in + fan_out))  # glorot uniform
        return tf.keras.initializers.uniform(-limit, limit)

    def call(
            self,
            inputs: tf.Tensor,
            state: tf.Tensor = None,
            mask: tf.Tensor = None,
            state_mask: tf.Tensor = None,
        ) -> tf.Tensor:
        """
        Args:
            inputs: a float tensor with shape (batch_size, timesteps, input_dim)
            state: a float tensor with shape (batch_size, memory_timesteps, input_dim)
            mask: None or a boolean tensor with shape (batch_size, timesteps)
            state_mask: None or a boolean tensor with shape (batch_size, memory_timesteps)
        Return:
            a float tensor with shape (batch_size, timesteps, output_dim)
        """
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # shape (N, T)

        if state is not None:
            concated = tf.concat([state, inputs], axis=1)
            if state_mask is not None:
                if mask is None:
                    raise TypeError("Invalid input!")
                state_mask = tf.cast(state_mask, inputs.dtype)
                concated_mask = tf.concat([state_mask, mask], axis=1)
            else:
                concated_mask = None
        else:
            concated = inputs
            concated_mask = mask

        query = tf.tensordot(inputs, self.query_W, axes=[2, 0])
        key, value = [
            tf.tensordot(concated, kernel, axes=[2, 0])
            for kernel in (self.key_W, self.value_W)
        ]  # shape (N, T, h, U)

        if mask is not None:
            query *= mask[:, :, tf.newaxis, tf.newaxis]

        rel = self._get_positional_matrix(
            q_length=query.shape[1].value,
            kv_length=key.shape[1].value,
            fan_in=inputs.shape[-1].value,
        )  # shape (T, t, h, U)

        attended_vec = self._multihead_attention(
            query=query,
            key=key,
            value=value,
            rel=rel,
            value_mask=concated_mask,
        )  # shape (N, h, U, T')
        outputs = tf.tensordot(
            attended_vec,
            self.output_W,
            axes=[[2, 3], [0, 1]],
        )  # shape (N, T', O)
        return outputs

    def _get_positional_matrix(self, q_length, kv_length, fan_in, amplitude=1., base=1e4):
        if (q_length, kv_length) in self._computed_rel:
            return self._computed_rel[(q_length, kv_length)]

        q_range = np.arange(start=0, stop=q_length, dtype=np.float32)  # shape (T,)
        kv_range = np.arange(
            start=q_length - kv_length,
            stop=q_length,
            dtype=np.float32,
        )  # shape (T,)
        relative_pos = q_range[:, np.newaxis] - kv_range  # shape (T, t)
        dim_range = np.arange(fan_in, dtype=np.float32)  # shape (D,)
        wave_length = np.power(base, 2. * dim_range / fan_in)  # shape (D,)

        offset = (np.pi / 2.) * (dim_range % 2)
        # [0, pi / 2, ...] for convert sin to cos on odd dim, shape (D,)

        theta = relative_pos[:, :, np.newaxis] / wave_length + offset
        outputs_np = amplitude * np.sin(theta)
        sin_wave_matrix = tf.constant(outputs_np, dtype=tf.float32)  # shape (T, t, D)

        rel = tf.tensordot(sin_wave_matrix, self.rel_W, axes=[2, 0])  # shape (T, t, h, U)
        self._computed_rel[(q_length, kv_length)] = rel
        return rel

    def _multihead_attention(self, query, key, value, rel, value_mask=None):
        """Multihead Attention with Relative Positional Encoding

        We define extended_timesteps = memory_timesteps + timesteps

        Args:
            query: a float tensor with shape (batch_size, timesteps, heads, units)
            key: a float tensor with shape (batch_size, extended_timesteps, heads, units)
            value: a float tensor with shape (batch_size, extended_timesteps, heads, units)
            rel: a relative positional encoding tensor (heads, units, extended_timesteps, timesteps)
            value_mask: a boolean tensor with shape (batch_size, timesteps)
        Return:
            float tensor with shape:  (batch_size, heads, units, timesteps)
        """
        # n: batch_size
        # T: timesteps
        # t: extended_timesteps
        # u: units
        logits = tf.einsum('nThu, nthu -> nTht', query, key)
        logits += tf.einsum('nThu, Tthu -> nTht', query + self.v, rel)
        logits += tf.einsum('nthu, hu -> nht', key, self.u)[:, tf.newaxis]  # (n, 1, h, t)
        # TODO, simpler implementation for heads = 1

        logits = self._mask_logits(logits / np.sqrt(self.units), value_mask)
        weights = tf.nn.softmax(logits)  # shape (N, T, h, t)

        attended_vec = tf.einsum('nTht, nthu -> nThu', weights, value)
        return attended_vec

    def _mask_logits(self, logits, mask):
        """Mask Logits

        Args:
            logits: a float tensor with shape (batch_size, timesteps, heads, extended_timesteps)
            mask: a boolean tensor with shape (batch_size, extended_timesteps)

        Return:
            a float tensor with shape (batch_size, timesteps, heads, extended_timesteps)

        """
        if self.use_forward_mask:
            q_length, _, kv_length = logits.shape.as_list()[1:]
            if (q_length, kv_length) in self._computed_triu:
                triu_tensor = self._computed_triu[(q_length, kv_length)]
            else:
                triu_tensor = tf.constant(
                    np.triu(
                        np.full([q_length, kv_length], _LARGE_BIAS),
                        k=kv_length - q_length + 1,
                    )[:, np.newaxis],  # shape (T, 1, t), 1 to broadcast on heads
                    dtype=logits.dtype,
                )
                self._computed_triu[(q_length, kv_length)] = triu_tensor
                # example if (q_length, kv_length) = (3, 5)
                # [[0, 0, 0, 1e4, 1e4],
                #  [0, 0, 0, 0,   1e4],
                #  [0, 0, 0, 0,     0]]

            logits -= triu_tensor

        if mask is not None:
            bias = (1. - mask) * _LARGE_BIAS
            logits -= bias[:, tf.newaxis, tf.newaxis, :]  # shape (N, 1, 1, t)
            # TODO, simpler implementation for heads = 1

        return logits

    def compute_mask(self, inputs, mask):
        # use same mask
        return mask

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.as_list()
        output_shape[2] = self.output_dim
        return tf.TensorShape(output_shape)
