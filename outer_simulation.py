from Texasholdem import env
import random
from copy import deepcopy
from tqdm import trange
from typing import Tuple
import os
import tensorflow as tf



class Simulation_simulation:
    def __init__(self):
        from Texasholdem import env
        self.env = env()

        from inner_simulation import Simulation as ss
        self.sim = ss()

        self.model = tf.keras.models.load_model('Trained_Model_08_21')


    def __call__(self, num:int, community_card:list, my_hand:list, opponent_hand:list, opponent_action_num:int, pot_size:int=0, loss:int=0, shape_to_num:bool=False) -> int: # 시뮬레이션 결과 반환
        if loss == 0: loss = -1 # ... 
        self.shape_to_num = shape_to_num
        action_results = [0,0,loss*num] # -1*num : 참가비 1$
        for n,action in enumerate([0,1]): # action
            for i in range(num):
                self.community_card = deepcopy(community_card)
                self.my_hand = deepcopy(my_hand)
                # self.opponent_hand = deepcopy(opponent_hand)
                
                results = self.model_predict(community_card=deepcopy(community_card), computer_hand=deepcopy(my_hand), action=opponent_action_num)
                num1 = results.index(max(results)) + 2
                results[num1-2] = 0
                num2 = results.index(max(results)) + 2
                if num1 > 14: num1 -= 13
                if num2 > 14: num2 -= 13
                error1 = random.choice([-1,0,1]); error2 = random.choice([-1,0,1])
                if num1+error1 == 1 or num1+error1 == 15: error1 = 0
                if num2+error2 == 1 or num2+error2 == 15: error2 = 0
                if shape_to_num :
                    shape1 = random.choice([0, 1, 2, 3])
                    shape2 = random.choice([0, 1, 2, 3])
                else:
                    shape1 = random.choice(['♠', '♡', '♢', '♣'])
                    shape2 = random.choice(['♠', '♡', '♢', '♣'])
                self.opponent_hand = [[shape1, num1+error1], [shape2, num2+error2]]
                if self.opponent_hand[0] == self.opponent_hand[1]:
                    # print(self.opponent_hand)
                    self.opponent_hand[1][1] += 1
                    if self.opponent_hand[1][1] == 15:
                        self.opponent_hand[1][1] = 2
                for _index, p in enumerate(self.opponent_hand):
                    if tuple(p) in self.community_card + self.my_hand:
                        if shape_to_num:
                            p[0] += 1
                            if p[0] == 4: p[0] = 0
                        else :
                            shapes = ['♠', '♡', '♢', '♣']
                            shape_index = shapes.index(p[0])+1
                            # print(shape_index)
                            if shape_index == 4: shape_index = 0
                            p[0] = shapes[shape_index]
                        self.opponent_hand[_index] = p
                        if self.opponent_hand[0] == self.opponent_hand[1]:
                            self.opponent_hand[1][1] += 1
                            if self.opponent_hand[1][1] == 15:
                                self.opponent_hand[1][1] = 2
                self.opponent_hand = [tuple(v) for v in self.opponent_hand]

                _loss, pot_size = self.simulation(opponent_action=opponent_action_num, agent_action=action, pot_size=pot_size, loss=loss, shape_to_num=shape_to_num) # <<<
                action_results[n] += _loss

        # if len(community_card) == 3:
        #     w = 0.3
        #     action_results[0] *= w
        # print(action_results)
        
        # e = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        # print('e :',e)
        # if e:
        #     print('random')
        #     return random.choice([0,1])
        # print(action_results)
        
        return action_results.index(max(action_results))
    
    def model_predict(self, community_card:list, computer_hand:list, action:int) -> list:
        c_c = [i[1]-2 for i in community_card]; c_h = [i[1]-2 for i in computer_hand]
        community_card = tf.one_hot(c_c, depth=13)
        if len(community_card) == 3:
            tensor = tf.constant([[0 for i in range(13)] for p in range(2)], dtype=tf.float32)
            community_card = tf.concat([community_card, tensor], axis=0)
        computer_hand = tf.one_hot(c_h, depth=13); action = tf.one_hot([action], depth=2)
        data = tf.concat([tf.reshape(community_card, (1, -1)), tf.reshape(computer_hand, (1, -1)), action], axis=1)

        prediction = self.model.predict(data)
        return list(prediction)


    def simulation(self, opponent_action:int, agent_action:int, pot_size:int, loss:int, shape_to_num:bool=False) -> int: # 시뮬레이션
        # 상대방 action에서 fold 제외, 남은 공동카드 2장 랜덤으로 action이 목표, 손실률만 계산
        if not shape_to_num:
            shapes = ['♠', '♡', '♢', '♣']
            left_deck = [(s,i) for s in shapes for i in range(2, 15)]
        else:
            left_deck = [(s,i) for s in range(0, 4) for i in range(2, 15)]  # (Shape, Num) # S : 0, H : 1, D : 2, C : 3
        for i in self.community_card + self.my_hand + self.opponent_hand: # + self.opponent_hand 부분
            try: left_deck.remove(i) # 남은 카드들
            except: None

        # if self.community_card == 3: # 첫번째 베팅
        def betting_phase_simulation(action_num:int, loss:int, pot_size:int, _random:bool, agent_action:int) -> tuple:
            action_list = [0, 1] # 0: check, 1 : raise
            opponent_betting = 2 if action_list[action_num] else 0
            my_betting = 0

            # action = random.choice(action_list) if _random else agent_action
            action = self.sim(num=2, community_card=self.community_card, my_hand=self.opponent_hand, opponent_hand=None, opponent_action_num=action_num, pot_size=0, loss=0, shape_to_num=self.shape_to_num)

            if action == 2: # fold
                return loss, pot_size

            if not action: # check (check - check or raise - check)
                loss -= (opponent_betting - my_betting)
                pot_size += (opponent_betting - my_betting)

            elif action:  # raise
                # betting *= 2
                my_betting = opponent_betting*2 if opponent_betting else 2 # 2배 or 2
                loss -= my_betting
                pot_size += my_betting

                if not my_betting == 4: # check - raise
                    opponent_action = random.choice(action_list)
                    if not opponent_action: # check - raise - call
                        # call
                        pot_size += (my_betting - opponent_betting) # 상대방 check 기준
                    elif opponent_action : # check - raise - raise - call
                        # raise
                        opponent_betting = my_betting*2
                        pot_size += opponent_betting

                        # call
                        loss -= (opponent_betting - my_betting)
                        pot_size += (opponent_betting - my_betting)
                else: # raise - raise - call
                    # raise
                    my_betting = opponent_betting*2
                    loss -= my_betting
                    pot_size += my_betting

                    # call
                    pot_size += (my_betting - opponent_betting)
            
            my_betting, opponent_betting = 0, 0
            return loss, pot_size

        loss, pot_size = 0, 0
        _random = False
        if len(self.community_card) == 3:
            loss, pot_size = betting_phase_simulation(action_num=opponent_action, loss=loss, pot_size=pot_size, _random=False, agent_action=agent_action)
            new_community_card = random.sample(left_deck, 2)
            self.community_card += new_community_card
            _random = True
            agent_action = random.choice([0, 1, 2])
        
        if len(self.community_card) == 5:
            loss, pot_size = betting_phase_simulation(action_num=opponent_action, loss=loss, pot_size=pot_size, _random=_random, agent_action=agent_action)
        

        results = self.env.compare_hand(community_card=self.community_card, opponent_hand=self.opponent_hand, my_hand=self.my_hand)
        
        if results == True : # 승
            loss += pot_size
        elif results == False : # 패
            None
        elif results == None : # 무
            loss += (pot_size/2)

        return loss, pot_size

        # 
        
        
        # if self.community_card == 5: # 마지막 베팅
        #     opponent_action = random.sample(action_list)

        #     if not opponent_action : # check
        #         # pot_size += betting
        #         ...

        #     elif opponent_action : # raise
        #         pot_size += betting
            

        #     action = random.sample(action_list)

        #     if not action: #check


            

        # if action_list[action_num]:
        #     loss -= betting_size
        #     pot_size += betting_size
        #     betting_size *= 2
            
        #     opponent_action = random.sample(action_list)
        #     if opponent_action: # raise
        #         pot_size += betting_size

        
        

        



if __name__ == '__main__':
    def save_data(_dir:str, txt:str):
        with open(_dir, 'r', encoding='UTF-8') as f:
            data = f.readlines()
        data.append(txt+'\n')
        with open(_dir, 'w', encoding='UTF-8') as f:
            for i in data:
                f.write(i)

    env = env()
    sim = Simulation_simulation()
    simulation_num = 16 #600

    start_money = 30
    my_money = start_money; opponent_money = start_money
    pot_size = 0#; betting = 1
    
    while True:
        env() # 덱 초기화
        print('\n\tHolding Funds \nMe : {}$\tComputer : {}$\n{}'.format(my_money, opponent_money, '-'*35))
        my_money -= 1; opponent_money -= 1; pot_size += 2 # 참가비
        print('\tHolding Funds \nMe : {}$\tComputer : {}$\n\tPot : {}$\n'.format(my_money, opponent_money, pot_size))

        community_card = env.draw(input_deck=[], num=3)
        print('Community Card :', *community_card)
        
        hand = env.draw(input_deck=[], num=2) # 컴퓨터
        # print('Opponent Hand :', *hand)

        hand2 = env.draw(input_deck=[], num=2) # 사람
        print('My Hand :', *hand2)

        action_list = ['Check', 'Raise', 'Fold']

        def betting_phase(my_money:int, opponent_money:int, pot_size:int) -> any:
            betting = 1
            my_action = int(input('\nAction(0:check, 1:raise, 2:fold) : '))
            save_data('log.txt', txt=f'{community_card} / {hand2} / {hand} / {my_action} / {betting} / {pot_size}') # <<<

            if my_action == 2: # fold
                opponent_money += pot_size; pot_size = 0
                return my_money, opponent_money, pot_size, True
            elif my_action == 1: # raise
                betting *= 2; my_money -= betting; pot_size += betting
                print('\n\tHolding Funds \nMe : {}$\tComputer : {}$\n\tPot : {}$\n'.format(my_money, opponent_money, pot_size))

            action_num = sim(num=simulation_num, community_card=community_card, my_hand=hand, opponent_hand=hand2, opponent_action_num=my_action, pot_size=0, loss=0)
            print("Computer Action : {}".format(action_list[action_num]))
            if action_num == 2: # fold
                my_money += pot_size; pot_size = 0
                return my_money, opponent_money, pot_size, True
            elif action_num == 1: # raise
                betting *= 2; opponent_money -= betting; pot_size += betting
                print('\n\tHolding Funds \nMe : {}$\tComputer : {}$\n\tPot : {}$\n'.format(my_money, opponent_money, pot_size))
                my_action = int(input('\nAction(0:call, 2:fold) : '))
                save_data('log.txt', txt=f'{community_card} / {hand2} / {hand} / {my_action} / {betting} / {pot_size}') # <<<

                if my_action == 0: # call
                    my_money -= betting/2; pot_size += betting/2
                    print('\n\tHolding Funds \nMe : {}$\tComputer : {}$\n\tPot : {}$\n'.format(my_money, opponent_money, pot_size))
                elif my_action == 2: # fold
                    opponent_money += pot_size; pot_size = 0
                    return my_money, opponent_money, pot_size, True
            elif action_num == 0: # check or call
                if not betting == 1:
                    opponent_money -= betting/2; pot_size += betting/2
                print('\n\tHolding Funds \nMe : {}$\tComputer : {}$\n\tPot : {}$\n'.format(my_money, opponent_money, pot_size))

            return my_money, opponent_money, pot_size, False
        my_money, opponent_money, pot_size, _ = betting_phase(my_money, opponent_money, pot_size)
        if _ : continue

        community_card = env.draw(input_deck=community_card, num=2)

        print('Community Card :', *community_card)
        # print('Opponent Hand :', *hand)
        print('My Hand :', *hand2)

        my_money, opponent_money, pot_size, _ = betting_phase(my_money, opponent_money, pot_size)
        if _ : continue

        # my_action = int(input('Action(0:check, 1:raise, 2:fold) : '))

        # if my_action == 2: # fold
        #     opponent_money += pot_size
        #     continue
        # elif my_action == 1: # raise
        #     my_money -= betting; pot_size += betting; betting *= 2
        #     print('\tHolding Funds \nMe : {}$\tComputer : {}$\nPot : {}$'.format(my_money, opponent_money, pot_size))

        # action_num = sim(num=simulation_num, community_card=community_card, my_hand=hand, opponent_hand=hand2, opponent_action_num=my_action, pot_size=0, loss=0)
        # print(action_list[action_num])

        # action_num = sim(num=simulation_num, community_card=community_card, my_hand=hand, opponent_hand=hand2, opponent_action_num=my_action, pot_size=0, loss=0)
        # print(action_list[action_num])
        # if action_num == 2: # fold
        #     my_money += pot_size
        #     continue
        # elif action_num == 1: # raise
        #     opponent_money -= betting; pot_size += betting; betting *= 2
        #     print('\tHolding Funds \nMe : {}$\tComputer : {}$\nPot : {}$'.format(my_money, opponent_money, pot_size))
        #     my_action = int(input('Action(0:call, 1:fold) : '))
        #     if my_action == 0: # call
        #         my_money -= betting/2; pot_size -= betting/2
        #         print('\tHolding Funds \nMe : {}$\tComputer : {}$\nPot : {}$'.format(my_money, opponent_money, pot_size))
        #     elif my_action == 1: # fold
        #         opponent_money += pot_size
        #         continue
        # elif action_num == 0: # check or call
        #     opponent_money -= betting/2; pot_size += betting/2
        #     print('\tHolding Funds \nMe : {}$\tComputer : {}$\nPot : {}$'.format(my_money, opponent_money, pot_size))

        
        results = env.compare_hand(community_card=community_card, opponent_hand=hand, my_hand=hand2)
        if results == True:
            print('Win')
            my_money += pot_size
            pot_size = 0
        elif results == None:
            print('Draw')
            my_money += (pot_size/2); opponent_money += (pot_size/2)
            pot_size = 0
        else :
            print('Lose')
            opponent_money += pot_size
            pot_size = 0

        # print('\tHolding Funds \nMe : {}$\tComputer : {}$'.format(my_money, opponent_money))
        print('Computer Hand :', *hand[:2],'\nCommunity Card :', *hand[2:])

        if my_money <= 0 or opponent_money <= 0:
            break
    print('\tHolding Funds \nMe : {}$\tComputer : {}$\n'.format(my_money, opponent_money))
    

"""

"""