from simulation_fold import Simulation 
from Texasholdem import env
import tensorflow as tf
from tqdm import trange
import random
from collections import deque


mae = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

env = env(True)
sim = Simulation()
simulation_num = 3000 #600

def one_hot_encoding(cards:list) -> list:
    frame = [0 for i in range(52)] # 0 52개 리스트
    def card_to_index(card:tuple) -> int:
        num = card[0]*13 + card[1] - 1 # 2~A(14) # 13진법 -> 10진법
        return num - 1 # index는 0부터 시작
    
    for card in cards: frame[card_to_index(card)] = 1
    return frame


model = tf.keras.models.load_model('Trained_Model1')

replaymemory = deque(maxlen=1000)

num = 300
for _ in trange(num):
    env() # 덱 초기화
    dataset = []

    start_money = 30
    my_money = start_money; opponent_money = start_money
    pot_size = 0; betting = 1

    community_card, human_hand, computer_hand = [], [] ,[]
    community_card = env.draw(community_card, 3)
    human_hand = env.draw(human_hand, 2); computer_hand = env.draw(computer_hand, 2)

    results = sim(num=simulation_num, community_card=community_card, my_hand=human_hand, opponent_hand=computer_hand, opponent_action_num=0, pot_size=pot_size, loss = 0, shape_to_num=True)
    if results == 1:
        betting *= 2
        pot_size += betting
    loss = -1*betting
    dataset.append([community_card, human_hand, computer_hand, results, betting, pot_size])

    action = random.choice([0, 1])
    if action == 0:  # call
        # loss = -1*betting
        pot_size += betting
    elif action == 1: # raise
        betting *= 2
        # loss = -1*betting
        pot_size += betting
        results = sim(num=simulation_num, community_card=community_card, my_hand=human_hand, opponent_hand=computer_hand, opponent_action_num=action, pot_size=pot_size, loss = loss, shape_to_num=True)
        if results == 0 or results == 1: # call
            loss -= 2 # <<<
            pot_size += 2
            dataset.append([community_card, human_hand, computer_hand, 0, betting, pot_size])

        elif results == 2: # fold
            continue

        # action = random.choice([0, 2])
    # elif action == 2: # fold
    #     continue

    betting = 1
    community_card = env.draw(community_card, 2)
    results = sim(num=simulation_num, community_card=community_card, my_hand=human_hand, opponent_hand=computer_hand, opponent_action_num=0, pot_size=pot_size, loss = 0, shape_to_num=True)
    if results == 1:
        betting *= 2
        pot_size += betting
    dataset.append([community_card, human_hand, computer_hand, results, betting, pot_size])


    for n,  [cc, hh, ch, action, betting, pot_size] in enumerate(dataset):
        data = one_hot_encoding(cc) + one_hot_encoding(ch) + [action for _ in range(52)] + [betting for _ in range(52)] + [pot_size for _ in range(52)]
        tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        tensor_data = tf.reshape(tensor, [1,-1])
        
        label = one_hot_encoding(hh)
        tensor = tf.convert_to_tensor(label, dtype=tf.float32)
        tensor_label = tf.reshape(tensor, [1,52])
        dataset[n] = [tensor_data, tensor_label]

    for i in dataset:
        replaymemory.append(i)

    if len(replaymemory) > 100: # 일정량 이상 데이터 쌓임
        batch_size = 8
        batch_data = random.sample(replaymemory, batch_size)
        
        train_x, train_y = [], []
        for data, label in batch_data:
            train_x.append(data)
            train_y.append(label)

        epochs = 4
        for _ in range(epochs):
            for n, x in enumerate(train_x):
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_weights)
                    prediction = model(x)
                    loss = mae(train_y[n], prediction)        
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))


model.save('Trained_Model1')