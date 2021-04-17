
# coding: utf-8


from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Conv2D, ConvLSTM2D, concatenate

# # # DEFINE DIFFERENT MODEL ARCHITECTURES # # #
input_shape_one = 11


def stacked_1_model():   
    model = Sequential(name='model_stacked_2')
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(input_shape_one, 91, 134, 4), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    print(model.summary())
    
    return model



def stacked_2_model():   
    model = Sequential(name='model_stacked_2')
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(input_shape_one, 91, 134, 4), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    print(model.summary())
    
    return model



def stacked_3_model():
    model = Sequential(name='model_stacked_3')
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(input_shape_one, 91, 134, 4), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    print(model.summary())
    return model




def stacked_4_model():
    model = Sequential(name='model_stacked_3')
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(input_shape_one, 91, 134, 4), padding='same', return_sequences=True, 
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=134, kernel_size=(3, 3), padding='same', return_sequences=False, 
                         activation='tanh', recurrent_activation='hard_sigmoid', 
                         kernel_initializer='glorot_uniform', unit_forget_bias=True, 
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False ))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last'))
    print(model.summary())
    return model




def stacked_sep_1_model():
    evap_in = Input(shape = (input_shape_one, 91, 134, 1))
    evap_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_in)
    evap_out = BatchNormalization()(evap_2)
    
    
    temp_in = Input(shape = (input_shape_one, 91, 134, 1))
    temp_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_in)
    temp_out = BatchNormalization()(temp_2)
    
    
    precip_in = Input(shape = (input_shape_one, 91, 134, 1))
    precip_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_in)
    precip_out = BatchNormalization()(precip_2)
    
    
    q_in = Input(shape = (input_shape_one, 91, 134, 1))
    q_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_in)
    q_out = BatchNormalization()(q_2)
    
    
    merge_1 = concatenate([evap_out, temp_out, precip_out, q_out])
    merge = BatchNormalization()(merge_1)
    
    out = Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last')(merge)
    
    
    model = Model(inputs = [evap_in, temp_in, precip_in, q_in], outputs = [out])
    print(model.summary())
    return model




def stacked_sep_2_model():
    evap_in = Input(shape = (input_shape_one, 91, 134, 1))
    evap = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_in)
    evap_out_1 = BatchNormalization()(evap)
    
    evap_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_out_1)
    evap_out = BatchNormalization()(evap_2)
    
    
    temp_in = Input(shape = (input_shape_one, 91, 134, 1))
    temp_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_in)
    
    temp_out_1 = BatchNormalization()(temp_1)
    
    temp_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_out_1)
    temp_out = BatchNormalization()(temp_2)
    
    
    precip_in = Input(shape = (input_shape_one, 91, 134, 1))
    precip_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_in)
    precip_out_1 = BatchNormalization()(precip_1)
    
    precip_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_out_1)
    precip_out = BatchNormalization()(precip_2)
    
    
    q_in = Input(shape = (input_shape_one, 91, 134, 1))
    q_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_in)
    q_out_1 = BatchNormalization()(q_1)
    
    q_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_out_1)
    q_out = BatchNormalization()(q_2)
    
    
    merge_1 = concatenate([evap_out, temp_out, precip_out, q_out])
    merge = BatchNormalization()(merge_1)
    
    out = Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last')(merge)
    
    
    model = Model(inputs = [evap_in, temp_in, precip_in, q_in], outputs = [out])
    print(model.summary())
    return model




def stacked_sep_3_model():
    evap_in = Input(shape = (input_shape_one, 91, 134, 1))
    evap = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_in)
    evap_out_1 = BatchNormalization()(evap)
    
    evap_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_out_1)
    evap_out_2 = BatchNormalization()(evap_2)
    
    evap_3 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(evap_out_2)
    evap_out = BatchNormalization()(evap_3)
    
    
    
    temp_in = Input(shape = (input_shape_one, 91, 134, 1))
    temp_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_in)
    
    temp_out_1 = BatchNormalization()(temp_1)
    
    temp_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_out_1)
    temp_out_2 = BatchNormalization()(temp_2)
    
    temp_3 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(temp_out_2)
    temp_out = BatchNormalization()(temp_3)
    
    
    
    precip_in = Input(shape = (input_shape_one, 91, 134, 1))
    precip_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_in)
    precip_out_1 = BatchNormalization()(precip_1)
    
    precip_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_out_1)
    precip_out_2 = BatchNormalization()(precip_2)
    
    precip_3 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(precip_out_2)
    precip_out = BatchNormalization()(precip_3)
    
    
    
    q_in = Input(shape = (input_shape_one, 91, 134, 1))
    q_1 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_in)
    q_out_1 = BatchNormalization()(q_1)
    
    q_2 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(4, 91, 134, 1), padding='same', return_sequences=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_out_1)
    q_out_2 = BatchNormalization()(q_2)
    
    q_3 = ConvLSTM2D(filters=134, kernel_size=(3, 3),
                         input_shape=(4, 91, 134, 1), padding='same', return_sequences=False,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=False)(q_out_2)
    q_out = BatchNormalization()(q_3)
    
    
    merge_1 = concatenate([evap_out, temp_out, precip_out, q_out])
    merge = BatchNormalization()(merge_1)
    
    out = Conv2D(filters=1, kernel_size=(1, 1),
                   activation='relu',
                   padding='same', data_format='channels_last')(merge)
    
    
    model = Model(inputs = [evap_in, temp_in, precip_in, q_in], outputs = [out])
    print(model.summary())
    return model



# # # COMPILE MODELS # # #

stacked_1 = stacked_1_model()
stacked_2 = stacked_2_model()
stacked_3 = stacked_3_model()
stacked_4 = stacked_4_model()
stacked_sep_1 = stacked_sep_1_model()
stacked_sep_2 = stacked_sep_2_model()
stacked_sep_3 = stacked_sep_3_model()

# # # SAVE MODEL ARCHITECTURES # # #

stacked_1.save('model_stacked_1')
stacked_2.save('model_stacked_2')
stacked_3.save('model_stacked_3')
stacked_4.save('model_stacked_4')
stacked_sep_1.save('model_stacked_sep_1')
stacked_sep_2.save('model_stacked_sep_2')
stacked_sep_3.save('model_stacked_sep_3')