import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Dropout, Add, LayerNormalization, GlobalAveragePooling1D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def gated_attention(inputs, attn_units=None, gate_type='sigmoid', name='gated_attn'):
    """Gated attention over time dimension.

    inputs: (batch, time, channels)
    returns: (batch, channels) weighted sum
    """
    x = inputs
    channels = int(x.shape[-1])
    if attn_units is None:
        attn_units = max(32, channels // 2)

    # u: attention hidden representation (batch, time, attn_units)
    u = Dense(attn_units, activation='tanh', name=f'{name}_u')(x)

    # choose gate implementation
    if gate_type == 'sigmoid':
        # elementwise sigmoid gate (per time, per unit)
        g = Dense(attn_units, activation='sigmoid', name=f'{name}_g')(x)
        h = Lambda(lambda t: t[0] * t[1], name=f'{name}_h')([u, g])

    elif gate_type == 'scalar':
        # per-time scalar gate, broadcasted to attn_units
        g_scalar = Dense(1, activation='sigmoid', name=f'{name}_g_scalar')(x)  # (batch, time, 1)
        g = Lambda(lambda s: K.repeat_elements(s, attn_units, axis=-1), name=f'{name}_g_broadcast')(g_scalar)
        h = Lambda(lambda t: t[0] * t[1], name=f'{name}_h')([u, g])

    elif gate_type == 'vector':
        # trainable vector gate (one vector of length attn_units), broadcast across batch and time
        class _VectorGateLayer(tf.keras.layers.Layer):
            def __init__(self, units, name=None, **kw):
                super().__init__(name=name, **kw)
                self.units = units

            def build(self, input_shape):
                self.v = self.add_weight(shape=(self.units,), initializer='ones', trainable=True, name=(self.name or 'vector_gate') + '_v')

            def call(self, inputs):
                batch = tf.shape(inputs)[0]
                time = tf.shape(inputs)[1]
                v = tf.sigmoid(self.v)
                v = tf.reshape(v, (1, 1, self.units))
                return tf.tile(v, (batch, time, 1))

        g = _VectorGateLayer(attn_units, name=f'{name}_g_vector')(x)
        h = Lambda(lambda t: t[0] * t[1], name=f'{name}_h')([u, g])

    elif gate_type == 'vector_sigmoid':
        # channel-wise trainable gate + residual combine
        class _ChannelVectorGateLayer(tf.keras.layers.Layer):
            def __init__(self, channels, name=None, **kw):
                super().__init__(name=name, **kw)
                self.channels = channels

            def build(self, input_shape):
                self.v = self.add_weight(shape=(self.channels,), initializer='ones', trainable=True, name=(self.name or 'channel_vector_gate') + '_v')

            def call(self, inputs):
                batch = tf.shape(inputs)[0]
                time = tf.shape(inputs)[1]
                v = tf.sigmoid(self.v)
                v = tf.reshape(v, (1, 1, self.channels))
                return tf.tile(v, (batch, time, 1))

        g = _ChannelVectorGateLayer(channels, name=f'{name}_g_channel')(x)
        # gated part and residual combine
        gated_part = Lambda(lambda t: t[0] * t[1], name=f'{name}_gated_part')([x, g])
        residual_out = Lambda(lambda t: t[0] + t[1], name=f'{name}_residual_out')([x, gated_part])
        # map residual (B,T,C) to attn_units for scoring
        h = Dense(attn_units, activation='tanh', name=f'{name}_h_from_residual')(residual_out)

    elif gate_type == 'vector_softmax':
        # vector per-attn-unit scaling combined with time-wise softmax over u
        class _VectorGateLayer(tf.keras.layers.Layer):
            def __init__(self, units, name=None, **kw):
                super().__init__(name=name, **kw)
                self.units = units

            def build(self, input_shape):
                self.v = self.add_weight(shape=(self.units,), initializer='ones', trainable=True, name=(self.name or 'vector_gate') + '_v')

            def call(self, inputs):
                batch = tf.shape(inputs)[0]
                time = tf.shape(inputs)[1]
                v = tf.sigmoid(self.v)
                v = tf.reshape(v, (1, 1, self.units))
                return tf.tile(v, (batch, time, 1))

        v_broadcast = _VectorGateLayer(attn_units, name=f'{name}_g_vector')(x)
        # time-wise softmax logits from u
        logits = Dense(1, name=f'{name}_g_logits')(u)
        logits = Lambda(lambda t: K.squeeze(t, axis=-1), name=f'{name}_g_squeeze_vecsoft')(logits)
        weights = Lambda(lambda s: K.softmax(s, axis=1), name=f'{name}_g_softmax_vecsoft')(logits)
        weights_exp = Lambda(lambda w: K.expand_dims(w, axis=-1), name=f'{name}_g_expand_vecsoft')(weights)
        weights_broad = Lambda(lambda t: K.repeat_elements(t, attn_units, axis=-1), name=f'{name}_g_repeat_vecsoft')(weights_exp)
        g = Lambda(lambda t: t[0] * t[1], name=f'{name}_g_mul_vecsoft')([v_broadcast, weights_broad])
        h = Lambda(lambda t: t[0] * t[1], name=f'{name}_h')([u, g])

    elif gate_type == 'softmax':
        # time-wise softmax gate derived from u, broadcast to attn_units
        logits = Dense(1, name=f'{name}_g_logits')(u)
        logits = Lambda(lambda t: K.squeeze(t, axis=-1), name=f'{name}_g_squeeze')(logits)
        weights = Lambda(lambda s: K.softmax(s, axis=1), name=f'{name}_g_softmax')(logits)
        weights_exp = Lambda(lambda w: K.expand_dims(w, axis=-1), name=f'{name}_g_expand')(weights)
        g = Lambda(lambda t: K.repeat_elements(t, attn_units, axis=-1), name=f'{name}_g_broadcast_softmax')(weights_exp)
        h = Lambda(lambda t: t[0] * t[1], name=f'{name}_h')([u, g])

    elif gate_type == 'vector_sigmoid_softmax':
        # channel sigmoid gate combined with time softmax gate (double gating)
        class _ChannelVectorGateLayer(tf.keras.layers.Layer):
            def __init__(self, channels, name=None, **kw):
                super().__init__(name=name, **kw)
                self.channels = channels

            def build(self, input_shape):
                self.v = self.add_weight(shape=(self.channels,), initializer='ones', trainable=True, name=(self.name or 'channel_vector_gate') + '_v')

            def call(self, inputs):
                batch = tf.shape(inputs)[0]
                time = tf.shape(inputs)[1]
                v = tf.sigmoid(self.v)
                v = tf.reshape(v, (1, 1, self.channels))
                return tf.tile(v, (batch, time, 1))

        channel_gate = _ChannelVectorGateLayer(channels, name=f'{name}_g_channel')(x)
        time_logits = Dense(1, name=f'{name}_g_time_logits')(u)
        time_logits = Lambda(lambda t: K.squeeze(t, axis=-1), name=f'{name}_g_time_squeeze')(time_logits)
        time_weights = Lambda(lambda s: K.softmax(s, axis=1), name=f'{name}_g_time_softmax')(time_logits)
        time_weights_exp = Lambda(lambda w: K.expand_dims(w, axis=-1), name=f'{name}_g_time_expand')(time_weights)
        time_weights_broad = Lambda(lambda t: K.repeat_elements(t, channels, axis=-1), name=f'{name}_g_time_repeat')(time_weights_exp)
        gated = Lambda(lambda t: t[0] * t[1] * t[2], name=f'{name}_gated_triple')([x, channel_gate, time_weights_broad])
        h = Dense(attn_units, activation='tanh', name=f'{name}_h_from_g')(gated)

    else:
        raise ValueError(f'Unknown gate_type: {gate_type}')

    # score and softmax over time (use h)
    scores = Dense(1, name=f'{name}_score')(h)  # (batch, time, 1)
    scores = Lambda(lambda s: K.squeeze(s, axis=-1), name=f'{name}_squeeze')(scores)  # (batch, time)
    weights = Lambda(lambda s: K.softmax(s, axis=1), name=f'{name}_softmax')(scores)  # (batch, time)
    weights_exp = Lambda(lambda w: K.expand_dims(w, axis=-1), name=f'{name}_expand')(weights)
    # weighted sum over time of original attn output x
    weighted = Lambda(lambda t: K.sum(t[0] * t[1], axis=1), name=f'{name}_weighted')([x, weights_exp])
    return weighted


def residual_tcn_block(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.2, activation='relu', use_layernorm=True, padding='causal'):
    # 主路徑
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
    conv1 = Activation(activation)(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(conv1)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    # shortcut（必要時用 1x1 conv 調整通道）
    residual = x
    if int(x.shape[-1]) != filters:
        residual = Conv1D(filters=filters, kernel_size=1, padding='same')(residual)

    out = Add()([residual, conv2])

    if use_layernorm:
        out = LayerNormalization()(out)
    return out


def residual_tcn_block_scaled(x, filters, kernel_size=3, dilation_rate=1, dropout_rate=0.2, activation='relu', use_layernorm=True, padding='causal', residual_scale=0.1, trainable_scale=False):
    """Residual TCN block with residual scaling: out = x + alpha * F(x)

    residual_scale: float initial value for alpha; if trainable_scale True, alpha is a trainable scalar.
    """
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(x)
    conv1 = Activation(activation)(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate)(conv1)
    conv2 = Activation(activation)(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    residual = x
    if int(x.shape[-1]) != filters:
        residual = Conv1D(filters=filters, kernel_size=1, padding='same')(residual)

    if trainable_scale:
        # scalar variable (no explicit name to avoid duplicates when building multiple blocks)
        alpha = tf.Variable(initial_value=residual_scale, trainable=True, dtype=tf.float32)
        scaled = Lambda(lambda t: t * alpha)(conv2)
    else:
        # use anonymous lambda layer so Keras will auto-assign a unique name per layer
        scaled = Lambda(lambda t: t * residual_scale)(conv2)

    out = Add()([residual, scaled])
    if use_layernorm:
        out = LayerNormalization()(out)
    return out


def build_tcn_residual(input_shape, num_classes=1, num_filters=64, kernel_size=3, dilations=None, dropout_rate=0.2, use_layernorm=True, use_causal=True):
    """Build a TCN with residual blocks and dropout.

    Args:
        input_shape: (timesteps, features)
        num_classes: 1 for binary
        num_filters: base filters
        kernel_size: conv kernel size
        dilations: list of dilation rates per block
        dropout_rate: dropout rate applied after activations in main path
        use_layernorm: whether to apply LayerNormalization after residual add
        use_causal: whether to use causal padding (recommended for time-series)
    Returns:
        keras Model
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    padding = 'causal' if use_causal else 'same'

    inp = Input(shape=input_shape, name='input')
    x = inp

    # 如果輸入通道數與 num_filters 不同，先用 1x1 Conv 做一次映射（保持時間長度）
    if int(x.shape[-1]) != num_filters:
        x = Conv1D(filters=num_filters, kernel_size=1, padding='same')(x)

    for d in dilations:
        x = residual_tcn_block(x, filters=num_filters, kernel_size=kernel_size, dilation_rate=d, dropout_rate=dropout_rate, use_layernorm=use_layernorm, padding=padding)

    x = GlobalAveragePooling1D()(x)

    if num_classes == 1:
        out = Dense(1, activation='sigmoid', name='out')(x)
    else:
        out = Dense(num_classes, activation='softmax', name='out')(x)

    model = Model(inputs=inp, outputs=out)
    return model


def build_tcn_residual_gated(input_shape, num_classes=1, num_filters=64, kernel_size=3, dilations=None, dropout_rate=0.2, use_layernorm=True, use_causal=True, residual_scale=0.1, trainable_scale=False, attn_units=None, gate_type='sigmoid'):
    """Build a TCN with residual blocks + gated attention.

    - residual_scale: initial alpha multiplier for residual connection (out = x + alpha*F(x))
    - trainable_scale: if True, alpha is trainable
    - gated attention applied after final TCN stack, returns vector then dense output
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    padding = 'causal' if use_causal else 'same'

    inp = Input(shape=input_shape, name='input')
    x = inp

    if int(x.shape[-1]) != num_filters:
        x = Conv1D(filters=num_filters, kernel_size=1, padding='same')(x)

    for d in dilations:
        x = residual_tcn_block_scaled(x, filters=num_filters, kernel_size=kernel_size, dilation_rate=d, dropout_rate=dropout_rate, use_layernorm=use_layernorm, padding=padding, residual_scale=residual_scale, trainable_scale=trainable_scale)

    # apply gated attention over time
    attn_vec = gated_attention(x, attn_units=attn_units, gate_type=gate_type, name='gated_attn')

    if num_classes == 1:
        out = Dense(1, activation='sigmoid', name='out')(attn_vec)
    else:
        out = Dense(num_classes, activation='softmax', name='out')(attn_vec)

    model = Model(inputs=inp, outputs=out)
    return model
