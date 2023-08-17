import random
import ast
import numpy as np
import tensorflow as tf
from tqdm import trange

with open('data.txt', 'r' , encoding='UTF-8') as f:
    datas = f.readlines()

dataset = []; app = dataset.append
for data in datas:
    attributes = [community_card, human_hand, computer_hand, action, betting, pot_size] = data.split(' / ')
    app([ast.literal_eval(attribute) for attribute in attributes])


def one_hot_encoding(cards:list) -> list:
    frame = [0 for i in range(52)] # 0 52개 리스트
    def card_to_index(card:tuple) -> int:
        num = card[0]*13 + card[1] - 1 # 2~A(14) # 13진법 -> 10진법
        return num - 1 # index는 0부터 시작
    
    for card in cards: frame[card_to_index(card)] = 1
    return frame


for n,  [cc, hh, ch, action, betting, pot_size] in enumerate(dataset):
    data = one_hot_encoding(cc) + one_hot_encoding(ch) + [action for _ in range(52)] + [betting for _ in range(52)] + [pot_size for _ in range(52)]
    tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    # tensor_data = tf.reshape(tensor, [5,52])
    tensor_data = tf.reshape(tensor, [1,-1])
    
    label = one_hot_encoding(hh)
    tensor = tf.convert_to_tensor(label, dtype=tf.float32)
    tensor_label = tf.reshape(tensor, [1,52])
    dataset[n] = [tensor_data, tensor_label]

print(dataset[-1])
train_x, train_y = [], []
for data, label in dataset:
    train_x.append(data)
    train_y.append(label)

index_list = [i for i in range(len(train_x))]
random.shuffle(index_list)
x_train = [train_x[i] for i in index_list]
y_train = [train_y[i] for i in index_list]

# train_x = tf.convert_to_tensor(train_x)
# train_y = tf.convert_to_tensor(train_y)

model = tf.keras.models.load_model('Model')

mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

epochs = 32
for _ in trange(epochs):
    for n, x in enumerate(x_train):
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            prediction = model(x)
            loss = mae(y_train[n], prediction)        
                
        # print('Loss : {}'.format(loss))
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

model.save('Trained_Model_08_17')
