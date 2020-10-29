from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)
from keras import regularizers

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    #simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    simp_rnn = LSTM(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='tanh', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print("cnn_rnn_model output length: {}".format(model.output_length))
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    simp_rnn = LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_0')(input_data)
    bn_rnn = BatchNormalization(name='batch_norm_0')(simp_rnn)
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(1, recur_layers):
        rnn_name = "rnn_{}".format(i)
        bn_name = "batch_norm_{}".format(i)
        simp_rnn = LSTM(units, activation='tanh', return_sequences=True, implementation=2, name=rnn_name)(bn_rnn)
        bn_rnn = BatchNormalization(name=bn_name)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units, activation='tanh', return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model_1(input_dim, units, output_dim=29):
    """ Build a deep network for speech

    True transcription:
    her father is a most remarkable person to say the least
    --------------------------------------------------------------------------------
    Predicted transcription:
    fathers a mosre markle prsn to sa the l........................................

    """
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'

    dropout_rate = .20

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='tanh',
                     name='conv1d')(input_data)
    bnn_conv_1d = BatchNormalization(name='conv_1d_bn')(conv_1d)
    bnn_conv_1d = Dropout(dropout_rate)(bnn_conv_1d)

    rnn_1 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_1'))(bnn_conv_1d)
    rnn_1 = Dropout(dropout_rate)(rnn_1)
    bn_rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)

    rnn_2 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_2'))(bn_rnn_1)
    rnn_2 = Dropout(dropout_rate)(rnn_2)
    bn_rnn_2 = BatchNormalization(name='bn_rnn_2')(rnn_2)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_2)
    time_dense = Dropout(dropout_rate)(time_dense)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = ...
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)

    print("final_model output length: {}".format(model.output_length))
    print(model.summary())
    return model


def final_model_2(input_dim, units, output_dim=29):
    """ Build a deep network for speech
    """
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'

    dropout_rate = .20

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer

    #rnn_1 = Bidirectional(LSTM(units, activation='relu', return_sequences=True, implementation=2, kernel_regularizer=regularizers.l2(0.00000001), activity_regularizer=regularizers.l2(0.00000001), name='rnn_1'))(input_data)
    rnn_1 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_1'))(input_data)
    rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)
    rnn_1 = Dropout(dropout_rate)(rnn_1)


    #rnn_2 = Bidirectional(LSTM(units, activation='relu', return_sequences=True, implementation=2, kernel_regularizer=regularizers.l2(0.00000001), activity_regularizer=regularizers.l2(0.00000001), name='rnn_2'))(rnn_1)
    rnn_2 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_2'))(rnn_1)
    rnn_2 = BatchNormalization(name='bn_rnn_2')(rnn_2)
    rnn_2 = Dropout(dropout_rate)(rnn_2)

    time_dense = TimeDistributed(Dense(output_dim))(rnn_2)
    #time_dense = Dropout(dropout_rate)(time_dense)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    # model.output_length = ...
    model.output_length = lambda x: x

    print("final_model output length: {}".format(model.output_length))
    print(model.summary())
    return model

def final_model_3(input_dim, units, output_dim=29):
    #Epoch   20 / 20 step - loss: 212.0707 - val_loss: 184.9844
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'

    dropout_rate = .20

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='tanh',
                     name='conv1d')(input_data)
    bnn_conv_1d = BatchNormalization(name='conv_1d_bn')(conv_1d)
    bnn_conv_1d = Dropout(dropout_rate)(bnn_conv_1d)

    rnn_1 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_1'))(bnn_conv_1d)
    rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)
    rnn_1 = Dropout(dropout_rate)(rnn_1)

    rnn_2 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_2'))(rnn_1)
    rnn_2 = BatchNormalization(name='bn_rnn_2')(rnn_2)
    rnn_2 = Dropout(dropout_rate)(rnn_2)

    time_dense = TimeDistributed(Dense(output_dim))(rnn_2)
    time_dense = Dropout(dropout_rate)(time_dense)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = ...
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)

    print("final_model output length: {}".format(model.output_length))
    print(model.summary())
    return model

def final_model(input_dim, units, output_dim=29):
    """ Build a deep network for speech
    """
    filters = 200
    kernel_size = 11
    conv_stride = 2
    conv_border_mode = 'valid'

    dropout_rate = .20

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add convolutional layer

    #rnn_1 = Bidirectional(LSTM(units, activation='relu', return_sequences=True, implementation=2, kernel_regularizer=regularizers.l2(0.00000001), activity_regularizer=regularizers.l2(0.00000001), name='rnn_1'))(input_data)
    rnn_1 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_1'))(input_data)
    rnn_1 = BatchNormalization(name='bn_rnn_1')(rnn_1)
    rnn_1 = Dropout(dropout_rate)(rnn_1)


    #rnn_2 = Bidirectional(LSTM(units, activation='relu', return_sequences=True, implementation=2, kernel_regularizer=regularizers.l2(0.00000001), activity_regularizer=regularizers.l2(0.00000001), name='rnn_2'))(rnn_1)
    rnn_2 = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, implementation=2, name='rnn_2'))(rnn_1)
    rnn_2 = BatchNormalization(name='bn_rnn_2')(rnn_2)
    rnn_2 = Dropout(dropout_rate)(rnn_2)


    time_dense = TimeDistributed(Dense(output_dim))(rnn_2)
    #time_dense = Dropout(dropout_rate)(time_dense)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    # model.output_length = ...
    model.output_length = lambda x: x

    print("final_model output length: {}".format(model.output_length))
    print(model.summary())
    return model