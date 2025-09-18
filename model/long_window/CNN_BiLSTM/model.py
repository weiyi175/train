import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Permute, Dense, GlobalAveragePooling1D, Bidirectional, LSTM, Lambda, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def mlp_attention(x, attn_units=None, name='attn', mask=None, mask_mode='soft', mask_threshold=0.6):
    """
    Additive MLP attention with optional time mask.
    x: (B, T, C)
    mask: (B, T) with values in [0,1] meaning occlusion degree (1 = fully occluded).
    mask_mode:
      - 'soft': scores *= (1 - mask)
      - 'hard': scores[mask > threshold] = -inf (very negative)
    Returns: (context_vector, attn_weights)
    """
    if attn_units is None:
        attn_units = max(32, int(x.shape[-1]) // 2)
    u = Dense(attn_units, activation='tanh', name=f'{name}_u')(x)
    scores = Dense(1, name=f'{name}_score')(u)  # (B, T, 1)
    scores = Lambda(lambda t: K.squeeze(t, axis=-1), name=f'{name}_squeeze')(scores)  # (B, T)

    if mask is not None:
        # ensure mask shape (B,T)
        def apply_mask(args):
            s, m = args
            m = K.cast(m, K.floatx())
            if mask_mode == 'soft':
                # weight down occluded frames: multiply by (1 - m)
                return s * (1.0 - m)
            else:
                # hard: set very negative where m > threshold
                neg_inf = -1e9
                cond = K.cast(K.greater(m, mask_threshold), K.floatx())
                return s + cond * neg_inf
        scores = Lambda(apply_mask, name=f'{name}_apply_mask')([scores, mask])

    weights = Lambda(lambda s: K.softmax(s, axis=1), name=f'{name}_softmax')(scores)  # (B, T)
    weights_exp = Lambda(lambda w: K.expand_dims(w, axis=-1), name=f'{name}_expand')(weights)  # (B, T, 1)
    weighted = Lambda(lambda t: K.sum(t[0] * t[1], axis=1), name=f'{name}_weighted')([x, weights_exp])
    return weighted, weights


def build_cnn_bilstm(input_shape, num_filters=64, kernel_sizes=(3,5,3), conv_dropout=0.2, lstm_units=64, lstm_dropout=0.2, attn_units=None, use_mask=False, mask_mode='soft', mask_threshold=0.6):
    """Build a small CNN + BiLSTM + Attention model.

    input_shape: (features, timesteps) matching project NPZ (36,75)
    returns: Keras Model
    """
    inp = Input(shape=input_shape, name='input')  # (F, T)
    mask_inp = None
    if use_mask:
        # Expect mask shaped (T,) or (T,1). We'll squeeze/expand inside attention as needed.
        mask_inp = Input(shape=(input_shape[1],), name='attn_mask')
    # permute to (T, F)
    x = Permute((2,1), name='permute_time_features')(inp)

    # CNN blocks along time dimension
    for i, k in enumerate(kernel_sizes):
        x = Conv1D(filters=num_filters, kernel_size=k, padding='same', name=f'conv{i}')(x)
        x = BatchNormalization(name=f'bn{i}')(x)
        x = Activation('relu', name=f'act{i}')(x)
        x = Dropout(conv_dropout, name=f'conv_dropout{i}')(x)

    # BiLSTM
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout), name='bilstm')(x)

    # attention over time
    attn_vec, attn_weights = mlp_attention(x, attn_units=attn_units, name='mlp_attn', mask=mask_inp if use_mask else None, mask_mode=mask_mode, mask_threshold=mask_threshold)

    out = Dense(64, activation='relu', name='fc1')(attn_vec)
    out = Dropout(0.3, name='fc_dropout')(out)
    out = Dense(1, activation='sigmoid', name='out')(out)

    if use_mask:
        model = Model(inputs=[inp, mask_inp], outputs=out, name='cnn_bilstm_attn_masked')
    else:
        model = Model(inputs=inp, outputs=out, name='cnn_bilstm_attn')
    return model
