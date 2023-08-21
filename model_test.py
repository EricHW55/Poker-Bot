import tensorflow as tf
import numpy as np
import ast


model = tf.keras.models.load_model('Trained_Model_08_21')

def one_hot_encoding(cards:list) -> list:
    frame = [0 for i in range(52)] # 0 52개 리스트
    def card_to_index(card:tuple) -> int:
        num = card[0]*13 + card[1] - 1 # 2~A(14) # 13진법 -> 10진법
        return num - 1 # index는 0부터 시작
    
    for card in cards: frame[card_to_index(card)] = 1
    return frame

dataset = '[(3, 8), (0, 2), (0, 9)] / [(2, 8), (1, 4)] / [(0, 3), (3, 9)] / 1 / 2 / 2'
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]   
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
# [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   


attributes = [community_card, human_hand, computer_hand, action, betting, pot_size] = dataset.split(' / ')
dataset = [[ast.literal_eval(attribute) for attribute in attributes[:4]]]

for n,  [cc, hh, ch, action] in enumerate(dataset):
    c_c = [i[1]-2 for i in cc]
    h_h = [i[1]-2 for i in hh]
    c_h = [i[1]-2 for i in ch]
    community_card = tf.one_hot(c_c, depth=13)
    if len(community_card) == 3:
        tensor = tf.constant([[0 for i in range(13)] for p in range(2)], dtype=tf.float32)
        community_card = tf.concat([community_card, tensor], axis=0)
    human_hand = tf.one_hot(h_h, depth=13)
    computer_hand = tf.one_hot(c_h, depth=13)
    action = tf.one_hot([action], depth=2)
    data = tf.concat([tf.reshape(community_card, (1, -1)), tf.reshape(computer_hand, (1, -1)), action], axis=1)
    label = tf.reshape(human_hand, shape=[1,-1])
    dataset[n] = [data, label]


train_x, train_y = [], []
for data, label in dataset:
    train_x.append(data)
    train_y.append(label)

x_train = tf.reshape(tf.convert_to_tensor(train_x[0]), [-1, 93])
y_train = tf.reshape(tf.convert_to_tensor(train_y[0]), [-1, 26])

# x_train = tf.reshape(tf.convert_to_tensor([train_x[i] for i in index_list]), [-1, 3, 52])
# y_train = tf.reshape(tf.convert_to_tensor([train_y[i] for i in index_list]), [-1, 52])

print(x_train)

prediction = tf.constant(model.predict(x_train), dtype=tf.float32).numpy().tolist()[0]
print('prediction: ', prediction)
# prediction = list(prediction)


def return_index(num:int, value:list) -> int:
    return_list = []
    for i in range(num):
        return_list.append(value.index(max(value)))
        # print(return_list[-1])
        value[return_list[-1]] = -100
    return return_list

_list = return_index(2, prediction)


# def return_index(num:int, value:tf.Tensor) -> int:
#     return_list = []
#     for i in range(num):
#         return_list.append(int(tf.argmax(value, axis=1)))
#         print(return_list[-1])
#         value[return_list[-1]] = -100
#     return return_list

# _list = return_index(3, prediction)

print(y_train.numpy().tolist()[0])
print([1.0 if i in _list else 0.0 for i in range(26)])
