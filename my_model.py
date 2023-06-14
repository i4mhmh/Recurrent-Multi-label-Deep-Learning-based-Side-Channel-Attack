from abc import ABC

import tensorflow as tf
from tensorflow import keras
from keras import Model


class MyModel_ASCAD_Nmax_0(Model, ABC):
    def __init__(self):
        super(MyModel_ASCAD_Nmax_0, self).__init__()
        self.conv_1_left = keras.layers.Conv1D(4, 1, activation='selu', padding='same')
        self.conv_1_middle = keras.layers.Conv1D(4, 7, activation='selu', padding='same')
        self.conv_1_right = keras.layers.Conv1D(4, 11, activation='selu', padding='same')
        # 三个卷积合并为一层
        self.conv_2 = keras.layers.Conv1D(4, 1, activation='selu')
        self.bn_1 = keras.layers.BatchNormalization()
        self.pool_1 = keras.layers.AveragePooling1D() 
        self.flatten_1 = keras.layers.Flatten()
        
        # 全连接
        self.fc_1 = keras.layers.Dense(10, activation='selu')
        self.fc_2 = keras.layers.Dense(10, activation='selu')
        self.fc_3 = keras.layers.Dense(8, activation='sigmoid')

    def call(self, input):
        conv_1_left = self.conv_1_left(input)
        conv_1_middle = self.conv_1_middle(input)
        conv_1_right = self.conv_1_right(input)
        x = keras.layers.concatenate([conv_1_left, conv_1_middle, conv_1_right])
        x = self.conv_2(x)
        x = self.bn_1(x)
        x = self.pool_1(x)
        x = self.flatten_1(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        output = self.fc_3(x)
        return output

# model = MyModel_ASCAD_Nmax_0()
# model.build(input_shape=(None, 700, 1))
# print(model.summary())
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             multiple                  8         
                                                                 
 conv1d_1 (Conv1D)           multiple                  32        
                                                                 
 conv1d_2 (Conv1D)           multiple                  48        
                                                                 
 conv1d_3 (Conv1D)           multiple                  52        
                                                                 
 batch_normalization (BatchN  multiple                 16        
 ormalization)                                                   
                                                                 
 average_pooling1d (AverageP  multiple                 0         
 ooling1D)                                                       
                                                                 
 flatten (Flatten)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  14010     
                                                                 
 dense_1 (Dense)             multiple                  110       
                                                                 
 dense_2 (Dense)             multiple                  88        
                                                                 
=================================================================
Total params: 14,364
Trainable params: 14,356
Non-trainable params: 8
_________________________________________________________________
None
'''

class MyModel_ASCAD_Nmax_50(keras.Model, ABC):
    def __init__(self):
        super(MyModel_ASCAD_Nmax_50, self).__init__()
        self.conv_1_left = keras.layers.Conv1D(4, 1, activation='selu', padding='same')
        self.conv_1_middle = keras.layers.Conv1D(4, 7, activation='selu', padding='same')
        self.conv_1_right = keras.layers.Conv1D(4, 11, activation='selu', padding='same')

        #concatenate
        self.conv_2 = keras.layers.Conv1D(8, 1, activation='selu')
        self.bn_1= keras.layers.BatchNormalization()
        self.pool_1 = keras.layers.AveragePooling1D(2, 2)
        self.conv_3 = keras.layers.Conv1D(16, 25, activation='selu')
        self.bn_2 = keras.layers.BatchNormalization()
        self.pool_2 = keras.layers.AveragePooling1D(25, 25)
        self.conv_4 = keras.layers.Conv1D(32, 3, activation='selu')
        self.bn_3 = keras.layers.BatchNormalization()
        self.pool_3 = keras.layers.AveragePooling1D(4, 4)
        self.flatten_1 = keras.layers.Flatten()

        # fc
        self.fc_1 = keras.layers.Dense(15, 'selu')
        self.fc_2 = keras.layers.Dense(15, 'selu')
        self.fc_3 = keras.layers.Dense(15, 'selu')
        self.fc_4 = keras.layers.Dense(8, activation='sigmoid')
        
    def call(self, input):
        conv_1_left = self.conv_1_left(input)
        conv_1_middle =self.conv_1_middle(input)
        conv_1_right = self.conv_1_right(input)
        
        #concatenate
        concate = keras.layers.concatenate([conv_1_left, conv_1_middle, conv_1_right])
        x = self.conv_2(concate)
        x = self.bn_1(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.bn_2(x)
        x = self.pool_2(x)
        x = self.conv_4(x)
        x = self.bn_3(x)
        x = self.pool_3(x)
        x = self.flatten_1(x)

        # fc
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        output = self.fc_4(x)
        return output
# model = MyModel_ASCAD_Nmax_50()
# model.build(input_shape=(None, 700, 1))
# model.summary()
'''
Model: "my_model_ascad__nmax_50"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             multiple                  8         
                                                                 
 conv1d_1 (Conv1D)           multiple                  32        
                                                                 
 conv1d_2 (Conv1D)           multiple                  48        
                                                                 
 conv1d_3 (Conv1D)           multiple                  104       
                                                                 
 batch_normalization (BatchN  multiple                 32        
 ormalization)                                                   
                                                                 
 average_pooling1d (AverageP  multiple                 0         
 ooling1D)                                                       
                                                                 
 conv1d_4 (Conv1D)           multiple                  3216      
                                                                 
 batch_normalization_1 (Batc  multiple                 64        
 hNormalization)                                                 
                                                                 
 average_pooling1d_1 (Averag  multiple                 0         
 ePooling1D)                                                     
                                                                 
 conv1d_5 (Conv1D)           multiple                  1568      
                                                                 
 batch_normalization_2 (Batc  multiple                 128       
 hNormalization)                                                 
                                                                 
 average_pooling1d_2 (Averag  multiple                 0         
 ePooling1D)                                                     
                                                                 
 flatten (Flatten)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  975       
                                                                 
 dense_1 (Dense)             multiple                  240       
                                                                 
 dense_2 (Dense)             multiple                  240       
                                                                 
 dense_3 (Dense)             multiple                  128       
                                                                 
=================================================================
Total params: 6,783
Trainable params: 6,671
Non-trainable params: 112
'''



class MyModel_ASCAD_Nmax_100(keras.Model, ABC):
    def __init__(self):
        super(MyModel_ASCAD_Nmax_100, self).__init__()
        self.conv_1_left = keras.layers.Conv1D(4, 1, activation='selu', padding='same')
        self.conv_1_middle = keras.layers.Conv1D(4, 7, activation='selu', padding='same')
        self.conv_1_right = keras.layers.Conv1D(4, 11, activation='selu', padding='same')

        #concatenate
        self.conv_2 = keras.layers.Conv1D(8, 1, activation='selu')
        self.bn_1= keras.layers.BatchNormalization()
        self.pool_1 = keras.layers.AveragePooling1D(2, 2)
        self.conv_3 = keras.layers.Conv1D(16, 50, activation='selu')
        self.bn_2 = keras.layers.BatchNormalization()
        self.pool_2 = keras.layers.AveragePooling1D(50, 50)
        self.conv_4 = keras.layers.Conv1D(32, 3, activation='selu')
        self.bn_3 = keras.layers.BatchNormalization()
        self.pool_3 = keras.layers.AveragePooling1D(4, 4)
        self.flatten_1 = keras.layers.Flatten()

        # fc
        self.fc_1 = keras.layers.Dense(20, 'selu')
        self.fc_2 = keras.layers.Dense(20, 'selu')
        self.fc_3 = keras.layers.Dense(20, 'selu')
        self.fc_4 = keras.layers.Dense(8, activation='sigmoid')
        
    def call(self, input):
        conv_1_left = self.conv_1_left(input)
        conv_1_middle =self.conv_1_middle(input)
        conv_1_right = self.conv_1_right(input)
        
        #concatenate
        concate = keras.layers.concatenate([conv_1_left, conv_1_middle, conv_1_right])
        x = self.conv_2(concate)
        x = self.bn_1(x)
        x = self.pool_1(x)
        x = self.conv_3(x)
        x = self.bn_2(x)
        x = self.pool_2(x)
        x = self.conv_4(x)
        x = self.bn_3(x)
        x = self.pool_3(x)
        x = self.flatten_1(x)

        # fc
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        output = self.fc_4(x)
        return output
# model = MyModel_ASCAD_Nmax_100()
# model.build(input_shape=(None, 700, 1))
# model.summary()


class MyModelAES_HD(keras.Model, ABC):
    def __init__(self):
        super(MyModelAES_HD, self).__init__()
        self.conv_1 = keras.layers.Conv1D(2, 1, activation='selu', padding='same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.pool_1 = keras.layers.AveragePooling1D(2, 2)
        self.fn_1 = keras.layers.Flatten()
        self.fc_1 = keras.layers.Dense(2, activation='selu')
        self.fc_2 = keras.layers.Dense(8, activation='selu')
    def call(self, input):
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.pool_1(x)
        x = self.fn_1(x)
        x = self.fc_1(x)
        output = self.fc_2(x)
        return output
# model = MyModelAES_HD()
# model.build(input_shape=(None, 700, 1))
# model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             multiple                  4         
                                                                 
 batch_normalization (BatchN  multiple                 8         
 ormalization)                                                   
                                                                 
 average_pooling1d (AverageP  multiple                 0         
 ooling1D)                                                       
                                                                 
 flatten (Flatten)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  1402      
                                                                 
 dense_1 (Dense)             multiple                  24        
                                                                 
=================================================================
Total params: 1,438
Trainable params: 1,434
Non-trainable params: 4
'''

class MyModelAES_RD(keras.Model):
    def __init__(self):
        super(MyModelAES_RD, self).__init__()
        self.conv_1 = keras.layers.Conv1D(8, 1, activation='selu', padding='same')
        self.bn_1 = keras.layers.BatchNormalization()
        self.pool_1 = keras.layers.AveragePooling1D(2, 2)
        self.conv_2 = keras.layers.Conv1D(16, 50, activation='selu', padding='same')
        self.bn_2 = keras.layers.BatchNormalization()
        self.pool_2 = keras.layers.AveragePooling1D(50, 50)
        self.conv_3 = keras.layers.Conv1D(32, 3, activation='selu', padding='same')
        self.bn_3 = keras.layers.BatchNormalization()
        self.pool_3 = keras.layers.AveragePooling1D(7, 7)
        self.fn_1 = keras.layers.Flatten()
        self.fc_1 = keras.layers.Dense(10, activation='selu')
        self.fc_2 = keras.layers.Dense(10, activation='selu')
        self.fc_3 = keras.layers.Dense(8, activation='sigmoid')

    def call(self, input):
        x = self.conv_1(input)
        x = self.bn_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.pool_3(x)
        x = self.fn_1(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        output = self.fc_3(x)
        return output

# model = MyModelAES_RD()
# model.build(input_shape=(None, 700, 1))
# model.summary()