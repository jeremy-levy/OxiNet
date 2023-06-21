from tensorflow.keras import regularizers
from tensorflow.keras.layers import TimeDistributed, Dropout, Dense, LeakyReLU, Reshape, Conv1D, MaxPool1D, Add, \
    Input, Flatten, Bidirectional, LSTM, BatchNormalization, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, GaussianNoise
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal, HeUniform, Orthogonal


class WrongParameter(Exception):
    pass


def get_initializer(params):
    if params['initializer'] == 'GlorotNormal':
        return GlorotNormal(seed=params['seed'])
    if params['initializer'] == 'GlorotUniform':
        return GlorotUniform(seed=params['seed'])
    if params['initializer'] == 'HeNormal':
        return HeNormal(seed=params['seed'])
    if params['initializer'] == 'HeUniform':
        return HeUniform(seed=params['seed'])
    if params['initializer'] == 'Orthogonal':
        return Orthogonal(seed=params['seed'])


def get_metadata_input(params):
    nb_pobm_features = 176

    if params['all_metadata'] == 'meta_only':
        input_shape = 4
    elif params['all_metadata'] == 'meta_pobm':
        input_shape = 4 + nb_pobm_features
    elif params['all_metadata'] == 'meta_psg':
        input_shape = 16
    elif params['all_metadata'] == 'pobm_only':
        input_shape = nb_pobm_features
    else:
        input_shape = 0

    if params['features_ss'] is True:
        input_shape += 7

    return Input(shape=input_shape)


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def lstm_part(input_lstm, params):
    shape = params['shape']

    X = Reshape(target_shape=(shape[0] * shape[1], 1))(input_lstm)
    for i in range(params['nb_conv_lstm']):
        X = Conv1D(params['n_filters_lstm'], params['kernel_size_lstm'], activation='relu',
                   kernel_regularizer=regularizers.l2(l2=params['regularizer']))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.15)(X)
        X = MaxPool1D(2, strides=2)(X)

    X = Bidirectional(LSTM(units=int(params['num_features'] / 2), return_sequences=True))(X)
    X = Dropout(rate=params['lstm_dropout'])(X)
    X = attention()(X)

    X_prediction = Dense(units=1, name='lstm_output', kernel_regularizer=regularizers.l2(l2=params['regularizer']))(X)

    return X, X_prediction


def residual_convolution(X_in, params, it, kernel_initializer):
    X = X_in

    for i in range(params['residual_convolution_n_convs']):
        X = TimeDistributed(Conv1D(params['first_conv_n_filters'], params['residual_convolution_kernel_size'],
                                   dilation_rate=1, padding='same'),
                            name='conv_residual_convolution_' + str(it) + '_' + str(i))(X)
        X = TimeDistributed(BatchNormalization(), name='bn_residual_convolution_' + str(it) + '_' + str(i))(X)
        X = TimeDistributed(LeakyReLU(alpha=0.15), name='lr_residual_convolution_' + str(it) + '_' + str(i))(X)
        # X = Dropout(rate=params['cnn_dropout'])(X)

    X = TimeDistributed(MaxPool1D(2, strides=2), name='maxpool_residual_convolution' + str(it))(X)

    X_residual = TimeDistributed(MaxPool1D(2, strides=2))(X_in)
    X = TimeDistributed(Add(), name='add_residual_convolution_' + str(it))([X, X_residual])

    return X


def cnn_part(input_cnn, params, initializer):
    first_conv_n_filters = params['first_conv_n_filters']
    first_conv_kernel_size = params['first_conv_kernel_size']
    shape = params['shape']

    X = Reshape(target_shape=(shape[0], shape[1], 1))(input_cnn)
    X = TimeDistributed(Conv1D(first_conv_n_filters, first_conv_kernel_size, activation='relu',
                               kernel_regularizer=regularizers.l2(l2=params['regularizer'])), name="InputConv")(X)

    for i in range(params['num_residual_convolution']):
        X = residual_convolution(X, params=params, it=i, kernel_initializer=initializer)

    # Reshape from Window/Time/Feature to Time/Feature
    X = Reshape(target_shape=(X.shape[1] * X.shape[2], X.shape[3]))(X)

    dilation = params['dilation']
    for i in range(params['dilations_num']):
        X = Conv1D(params['num_features'], kernel_size=params['dilated_block_kernel_size'],
                   dilation_rate=dilation, kernel_regularizer=regularizers.l2(l2=params['regularizer']))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU(alpha=0.15, )(X)

        dilation = (dilation[0] * 2, )

    if params['pooling'] == 'average':
        X = GlobalAveragePooling1D()(X)
    elif params['pooling'] == 'max':
        X = GlobalMaxPooling1D()(X)
    else:
        raise WrongParameter('params[pooling] must be in {average, max}')
    X_prediction = Dense(units=1, name='cnn_output', kernel_regularizer=regularizers.l2(l2=params['regularizer']))(X)

    return X, X_prediction


def metadata_part(metadata_input, params):
    meta_osa = Dense(units=params['dense_meta'], kernel_regularizer=regularizers.l2(l2=params['regularizer']),
                     )(metadata_input)
    meta_osa = BatchNormalization()(meta_osa)
    meta_osa = LeakyReLU(alpha=0.15)(meta_osa)
    return meta_osa


def classifier_part(all_feats, params):
    def block_dense(input_block, units):
        output_block = Dense(units=units, kernel_regularizer=regularizers.l2(l2=params['regularizer']))(input_block)
        output_block = BatchNormalization()(output_block)
        output_block = LeakyReLU(alpha=0.15)(output_block)
        output_block = Dropout(rate=params['regressor_dropout'])(output_block)

        return output_block

    output_block_1 = block_dense(input_block=all_feats, units=params['first_dim'])
    output_block_2 = block_dense(input_block=output_block_1, units=int(params['first_dim'] / 2))
    output_block_3 = block_dense(input_block=output_block_2, units=int(params['first_dim'] / 4))

    X = Dense(units=1, name='final_output',
              kernel_regularizer=regularizers.l2(l2=params['regularizer']))(output_block_3)
    return X


class WeightAdjuster(callbacks.Callback):
    def __init__(self, weights: dict, change_epoch: int):
        """
        Args:
        weights (list): list of loss weights
        change_epoch (int): epoch number for weight change
        """
        super().__init__()

        self.weights = weights
        self.change_epoch = change_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.change_epoch == 0:
            # Updated loss weights

            K.set_value(self.weights['cnn_output'], self.weights['cnn_output'] / 1.5)
            K.set_value(self.weights['lstm_output'], self.weights['lstm_output'] / 1.5)


def get_duplo(params):
    shape = params['shape']

    # Batch_size, Num_window, Window_length
    oximetry_inputs = Input(shape=(shape[0], shape[1]))
    if params['gaussian_layer']:
        oximetry_inputs_2 = GaussianNoise(stddev=params['stddev'])(oximetry_inputs)
    else:
        oximetry_inputs_2 = oximetry_inputs

    initializer = get_initializer(params)
    metadata_input = get_metadata_input(params)

    feat_cnn, prediction_cnn = cnn_part(oximetry_inputs_2, params, initializer)
    feat_lstm, prediction_lstm = lstm_part(oximetry_inputs_2, params)

    if params['use_metadata'] is True:
        feat_meta = metadata_part(metadata_input, params)
        all_feats = Concatenate(name='final_concatenate')([feat_cnn, feat_meta, feat_lstm])
    else:
        all_feats = Concatenate(name='final_concatenate')([feat_cnn, feat_lstm])

    final_output = classifier_part(all_feats, params)

    model_osa = Model(inputs=[oximetry_inputs, metadata_input],
                      outputs=[prediction_cnn, prediction_lstm, final_output])

    losses = {
        "cnn_output": "MAE",
        "lstm_output": "MAE",
        "final_output": 'MAE'
    }

    if 'cnn_loss_weight' in params.keys():
        lossWeights = {"cnn_output": K.variable(params['cnn_loss_weight']),
                       "lstm_output": K.variable(params['lstm_loss_weight']),
                       'final_output': 1}
    else:
        lossWeights = {"cnn_output": K.variable(1.0), "lstm_output": K.variable(1.0), 'final_output': 1}

    model_osa.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss=losses, loss_weights=lossWeights)

    return model_osa
