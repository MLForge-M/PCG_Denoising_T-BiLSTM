# T-BiLSTM Model for PCG Signal Denoising
# --------------------------------------------------
# This implementation defines the T-BiLSTM model, combining
# Conv1D, Bidirectional LSTM, and skip connections for effective
#denoising of heart sound signals.

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Bidirectional, LSTM, UpSampling1D,
    Concatenate, Add, Dense, LayerNormalization
)


class TBiLSTMModel:
    def __init__(self, input_shape, loss_function):
        self.input = input_shape
        self.loss = loss_function
        self.model = self.build_model()

    def build_model(self):
        def transformer_block(x, num_heads=4, key_dim=16):
            x_norm = LayerNormalization()(x)
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=key_dim)(x_norm, x_norm)
            x = Add()([x, attn_output])
            x_norm = LayerNormalization()(x)
            ffn = Dense(key_dim * 2, activation='relu')(x_norm)
            ffn = Dense(x.shape[-1])(ffn)
            return Add()([x, ffn])

        input_sig = Input(shape=(self.input, 1), name="input_signal")

        # Encoder
        x = Conv1D(16, 31, strides=1, activation='relu', padding='same')(input_sig)
        rnn_1 = Bidirectional(LSTM(8, return_sequences=True, dropout=0.2))(x)

        x1 = Conv1D(32, 31, strides=2, activation='relu', padding='same')(rnn_1)
        rnn_2 = Bidirectional(LSTM(16, return_sequences=True, dropout=0.2))(x1)
        transformer_1 = transformer_block(rnn_2)

        x2 = Conv1D(32, 31, strides=2, activation='relu', padding='same')(transformer_1)
        rnn_3 = Bidirectional(LSTM(16, return_sequences=True, dropout=0.2))(x2)
        transformer_2 = transformer_block(rnn_3)

        x3 = Conv1D(64, 31, strides=2, activation='relu', padding='same')(transformer_2)
        rnn_4 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(x3)
        transformer_3 = transformer_block(rnn_4)

        x4 = Conv1D(64, 31, strides=2, activation='relu', padding='same')(transformer_3)
        rnn_5 = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2))(x4)
        transformer_4 = transformer_block(rnn_5)

        x5 = Conv1D(128, 31, strides=2, activation='relu', padding='same')(transformer_4)

        # Decoder
        d5 = Conv1D(64, 31, activation='relu', padding='same')(x5)
        d5 = UpSampling1D(2)(d5)
        d5 = Concatenate()([transformer_4, d5])

        d6 = Conv1D(64, 31, activation='relu', padding='same')(d5)
        d6 = UpSampling1D(2)(d6)
        d6 = Concatenate()([transformer_3, d6])

        d7 = Conv1D(32, 31, activation='relu', padding='same')(d6)
        d7 = UpSampling1D(2)(d7)
        d7 = Concatenate()([transformer_2, d7])

        d8 = Conv1D(32, 31, activation='relu', padding='same')(d7)
        d8 = UpSampling1D(2)(d8)
        d8 = Concatenate()([transformer_1, d8])

        d9 = Conv1D(32, 31, activation='relu', padding='same')(d8)
        d9 = UpSampling1D(2)(d9)
        d9 = Concatenate()([rnn_1, d9])

        output = Conv1D(1, 31, activation='tanh', padding='same')(d9)

        model = Model(inputs=input_sig, outputs=output, name="T-BiLSTM")
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=self.loss)
        return model