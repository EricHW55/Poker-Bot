import random
import ast
import numpy as np
import tensorflow as tf
from tqdm import trange

# 손실함수 cross entropy 로

with open('data.txt', 'r' , encoding='UTF-8') as f:
    datas = f.readlines()

dataset = []; app = dataset.append
for data in datas:
    attributes = [community_card, human_hand, computer_hand, action, betting, pot_size] = data.split(' / ')
    if not ast.literal_eval(action) == 2:
        app([ast.literal_eval(attribute) for attribute in attributes])


 
for n,  [cc, hh, ch, action, betting, pot_size] in enumerate(dataset):
    # print(action)
    # data = [one_hot_encoding(cc), one_hot_encoding(ch), action_list[action]]
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
    # print(community_card, human_hand, computer_hand, action)
    data = tf.concat([tf.reshape(community_card, (1, -1)), tf.reshape(computer_hand, (1, -1)), action], axis=1)
    # data = [one_hot_encoding(cc), one_hot_encoding(ch), [action for _ in range(52)]]
    # print(data)
    # data = one_hot_encoding(cc) + one_hot_encoding(ch) + [action for _ in range(52)] + [betting for _ in range(52)] + [pot_size for _ in range(52)]
    # tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    # tensor_data = tf.reshape(tensor, [5,52])
    # tensor_data = tf.reshape(tensor, [3,-1])
    
    label = tf.reshape(human_hand, shape=[1,-1])
    # print(data, label)
    # tensor = tf.convert_to_tensor(label, dtype=tf.float32)
    # tensor_label = tf.reshape(tensor, [1,52])
    # dataset[n] = [tensor_data, tensor_label]
    dataset[n] = [data, label]

# print(dataset[-1])
train_x, train_y = [], []
for data, label in dataset:
    train_x.append(data)
    train_y.append(label)

index_list = [i for i in range(len(train_x))]
random.shuffle(index_list)
# x_train = tf./
x_train = tf.reshape(tf.convert_to_tensor([train_x[i] for i in index_list], dtype=tf.float32), [-1, 93])
y_train = tf.reshape(tf.convert_to_tensor([train_y[i] for i in index_list], dtype=tf.float32), [-1, 26])

# x_train, x_test = x_train[:1000], x_train[1000:]
# y_train, y_test = y_train[:1000], y_train[1000:]

# trian_x = tf.constant(x_train)
# print(x_train, y_train)

# train_x = tf.convert_to_tensor(train_x)
# train_y = tf.convert_to_tensor(train_y)

model = tf.keras.models.load_model('Model')
history = model.fit(
    x_train, y_train,
    # validation_data=(x_test,y_test),
    batch_size=32, epochs=12,
)

history.history
# # mae = tf.keras.losses.MeanAbsoluteError()
# categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# epochs = 32
# for _ in trange(epochs):
#     for n, x in enumerate(x_train):
#         with tf.GradientTape() as tape:
#             tape.watch(model.trainable_weights)
#             prediction = model(x)
#             # loss = mae(y_train[n], prediction)        
#             loss = categorical_cross_entropy(y_train[n], prediction) 
                
#         # print('Loss : {}'.format(loss))
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))

model.save('Trained_Model_08_21_test')
