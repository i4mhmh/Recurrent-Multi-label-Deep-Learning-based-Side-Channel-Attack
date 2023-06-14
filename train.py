import tensorflow as tf
from tensorflow import keras
import h5py
import numpy as np
import my_model
import csv


def byte_to_bits(label):
    new_label = []
    for num in range(len(label)):
        bin_y = list(f"{label[num]:08b}")
        bin_y = list(map(float, bin_y))
        new_label.append(bin_y)
    new_label = np.reshape(new_label, (len(new_label), -1))
    return new_label



def data_processing(dataset):
    if dataset == 'ASCAD':
        in_file = h5py.File('../datasets/ASCAD/ASCAD_data/ASCAD_databases/ASCAD.h5', "r")
        X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float32)
        # Load profiling labels
        Y_profiling = np.array(in_file['Profiling_traces/labels'])
        new_profiling_label = byte_to_bits(Y_profiling)
        # Load attacking traces
        X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float32)
        # Load attacking labels
        Y_attack = np.array(in_file['Attack_traces/labels'])
        new_attack_label = byte_to_bits(Y_attack)
        # 为卷积层做预处理
        X_profiling = np.reshape(X_profiling, [50000, 700, 1])
        return [X_profiling, new_profiling_label, X_attack, new_attack_label]
    if dataset == 'AES_HD':
        # csv 100,000条 
        with open("../datasets/AES_HD_Dataset/traces_1.csv") as f:
            data = csv.reader(f)
            data = list(data)
        data = np.array(data)
        print(len(data))
def train(data, model_name):
    if 'ASCAD' in model_name:
        [X_profiling, Y_profiling, X_attack, Y_attack] = data
        if model_name == "ASCAD_Nmax_0":   # 这里对未加对抗措施的dataset进行攻击
            model = my_model.MyModel_ASCAD_Nmax_0()
        if model_name == "ASCAD_Nmax_50":
            model = my_model.MyModel_ASCAD_Nmax_50()
        if model_name == 'ASCAD_Nmax_100':
            model = my_model.MyModel_ASCAD_Nmax_100()
        model.compile(optimizer=keras.optimizers.Adam(0.005), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        history = model.fit(X_profiling, Y_profiling, epochs=100, batch_size=256, verbose=1, validation_split=0.2)
        model.save("models/" + model_name)  

if __name__ == '__main__':
    data = data_processing("AES_HD")
    # train(data=data, model_name="ASCAD_Nmax_100")