from copy import deepcopy
import random


class env:
    def __init__(self, shape_to_num:bool=False):
        self.shape_to_num = shape_to_num
        if shape_to_num:
            self.deck = [(s,i) for s in range(0, 4) for i in range(2, 15)]  # (Shape, Num) # S : 0, H : 1, D : 2, C : 3
        else:
            shapes = ['♠', '♡', '♢', '♣']
            self.deck = [(s,i) for s in shapes for i in range(2, 15)]
        random.shuffle(self.deck)
    
    def __call__(self): # 덱 초기화
        if self.shape_to_num:
            self.deck = [(s,i) for s in range(0, 4) for i in range(2, 15)]  # (Shape, Num) # S : 0, H : 1, D : 2, C : 3
        else:
            shapes = ['♠', '♡', '♢', '♣']
            self.deck = [(s,i) for s in shapes for i in range(2, 15)]
        random.shuffle(self.deck)
        
    def draw(self, input_deck:list, num:int=1) -> list:
        input_deck += self.deck[0:num]
        self.deck = self.deck[num:]
        return input_deck
    
    def check_hand(self, hand:list) -> int:
        hand_num = [n for (s,n) in hand]  # 숫자 리스트
        hand_num_list = list(set(hand_num))   # 중복 제외
        hand_num_count_list = [hand_num.count(n) for n in hand_num_list]  # 숫자 중복 개수 리스트
        print(hand_num_count_list)
        print(hand_num_list[hand_num_count_list.index(2)])

    def hand_ranking_check(self, hand) -> int:
        self.ranking_list = []
        # num = 0  # 족보 숫자

        def check_pair(count_num:list):  # 1
            if 2 in count_num :
                pair_num = hand[count_num.index(2)][-1]
                self.ranking_list.append([(i,n) for i,n in hand if n == pair_num])
                return 1
            return 0

        def check_two_pair(count_num:list):  # 2
            if count_num.count(2) >= 2:
                pair_pos = [n for n,i in enumerate(count_num) if i == 2]
                pair_num_list = [hand[i][-1] for i in pair_pos]
                pair_num_list.sort(reverse=True)
                self.ranking_list.append([(i,n) for i,n in hand if n == pair_num_list[0]])
                self.ranking_list.append([(i,n) for i,n in hand if n == pair_num_list[1]])
                # pair_num = hand[pair_pos[0]][-1]
                # self.ranking_list.append([(i,n) for i,n in hand if n == pair_num])
                # pair_num = hand[pair_pos[1]][-1]
                # self.ranking_list.append([(i,n) for i,n in hand if n == pair_num])
                return 1
            return 0

        def check_triple(count_num:list):  # 3
            if count_num.count(3) >= 1:
                triple_num = hand[count_num.index(3)][-1]
                self.ranking_list.append([(i,n) for i,n in hand if n == triple_num])
                return 1
            return 0
            
        def check_straight(sort_num:list):   # 4
            sort_num = list(set(sort_num))[::-1]
            if len(sort_num) < 5:
                return 0
            # _return = 0
            def check_five_continuous(num_list:list) -> bool:
                continuous_list = [num_list[0]-i for i in range(5)]
                if continuous_list == num_list:
                    num_hand_list = [_n[-1] for _n in hand]
                    index_ranking_list = [num_hand_list.index(_n) for _n in num_list]
                    # self.ranking_list = [hand[_n] for _n in index_ranking_list]
                    self.ranking_list.append([hand[_n] for _n in index_ranking_list])
                    return 1
                return 0
            # for i in range(len(sort_num)-4):
            #     print(sort_num[i:i+5])
            #     if check_five_continuous(sort_num[i:i+5]):
            #         return 1
            for i in range(len(sort_num)-4):
                if check_five_continuous(sort_num[i:i+5]):
                    return 1
            #         _return = 1
            # return 1 if _return else 0

        def check_flush(count_shape:list):  # 5
            if count_shape.count(5) >= 1:
                flush_shape = hand[count_shape.index(5)][0]
                # self.ranking_list = [(i,n) for i,n in hand if i == flush_shape]  # <<< 여기부터 코딩
                flush_shape_list = [(i,n) for i,n in hand if i == flush_shape] # 같은 모양의 카드들 (숫자 포함 tuple)
                flush_shape_num_list = [n for i,n in flush_shape_list] # int
                flush_shape_num_list.sort(reverse=True); flush_shape_num_list = flush_shape_num_list[:5] 
                flush_list = []  # ranking_list에 추가할 것
                for num in flush_shape_num_list:
                    flush_list += [i for i in flush_shape_list if i[1]==num]

                self.ranking_list.append(flush_list)
                # self.ranking_list.append([(i,n) for i,n in hand if i == flush_shape])
                return 1
            return 0

        def check_full_house(count_num:list):  # 6
            if check_triple(count_num) and check_two_pair(count_num):  # 트리플 때문에 2가 두개라 투페어 체크
                self.ranking_list.pop(0)  # 투페어로 두번 센거 팝

                if len([i for i in self.ranking_list if len(i) == 3]) == 2:  # 트리플 두번
                    num_list = [i[0][1] for i in self.ranking_list]
                    self.ranking_list = [self.ranking_list[num_list.index(max(num_list))], self.ranking_list[num_list.index(min(num_list))][:2]]
                return 1
            self.ranking_list = []
            return 0

        def check_four_of_a_kind(count_num:list):  # 7
            if count_num.count(4) >= 1:
                quadruple_num = hand[count_num.index(4)][-1]
                self.ranking_list.append([(i,n) for i,n in hand if n == quadruple_num])
                return 1
            return 0

        def check_straight_flush(count_shape:list, sort_num:list):  # 8
            if check_straight(sort_num) and check_flush(count_shape): # 1차 검증
                self.ranking_list = []
                # sort_num = list(set(sort_num))[::-1]
                flush_shape = hand[count_shape.index(5)][0]
                flush_num_list = [n for i,n in hand if i == flush_shape] # 같은 모양의 숫자들
                flush_num_list.sort(reverse=True)
                for i in range(len(flush_num_list)-4):
                    if flush_num_list[i:i+5] == [flush_num_list[i] - n for n in range(5)]:
                        self.ranking_list.append([(flush_shape, number) for number in flush_num_list[i:i+5]])
                        return 1
                return 0
            self.ranking_list = []
            return 0
        
        hand_num = [i[1] for i in hand] #; num = [i for i in range(1,14)]
        hand_shape = [i[0] for i in hand] #; num = [i for i in range(1,14)]
        count_num = [hand_num[n:].count(i) for n,i in enumerate(hand_num)]
        count_shape = [hand_shape[n:].count(i) for n,i in enumerate(hand_shape)]
        sort_num = deepcopy(hand_num); sort_num.sort(reverse=True)# ; continuous_num = []
    
        
        if check_straight_flush(count_shape, sort_num):
            if self.ranking_list in [[(shape, num) for num in range(10, 15)] for shape in ['s', 'h', 'd', 'c']] :
                # print('Royal Straight Flush')
                return 9
            # print('straight flush')
            return 8
        elif check_four_of_a_kind(count_num):
            # print('four of a kind')
            return 7
        elif check_full_house(count_num):
            # print('full house')
            return 6
        elif check_flush(count_shape):
            # print('flush')
            return 5
        elif check_straight(sort_num):
            # print('straight')
            return 4
        elif check_triple(count_num):
            # print('triple')
            return 3
        elif check_two_pair(count_num):
            # print('two pair')
            return 2
        elif check_pair(count_num):
            # print('pair')
            return 1
        else:
            # print('high card')
            # self.ranking_list.append(hand[hand_num.index(max(hand_num))])
            self.ranking_list.append([hand[hand_num.index(max(hand_num))]])
            return 0

    def compare_hand(self, community_card:list, opponent_hand:list, my_hand:list) -> bool:  # my hand를 기준으로 return
        opponent_hand += community_card
        my_hand += community_card

        opponent_rank = self.hand_ranking_check(hand=opponent_hand)
        opponent_ranking_list = self.ranking_list

        my_rank = self.hand_ranking_check(hand=my_hand)
        my_ranking_list = self.ranking_list

        # print(opponent_ranking_list)
        # print(my_ranking_list)

        if my_rank > opponent_rank:
            return True
        elif my_rank < opponent_rank:
            return False
        else: # (my_rank == opponent_rank) # 족보 같음
            if my_rank == 6: # full house
                my_ranking_list = my_ranking_list if len(my_ranking_list[0]) == 3 else my_ranking_list[::-1] # 트리플 먼저
            my_top_list = [i[0][1] for i in my_ranking_list]
            opponent_top_list = [i[0][1] for i in opponent_ranking_list]

            for n,i in enumerate(my_top_list):
                if i > opponent_top_list[n]:
                    return True
                elif i < opponent_top_list[n]:
                    return False

            # kicker 비교
            my_left_hand_list, opponent_left_hand_list = deepcopy(my_hand), deepcopy(opponent_hand)
            for ranking in my_ranking_list:
                for i in ranking: 
                    my_left_hand_list.remove(i)
            for ranking in opponent_ranking_list:
                for i in ranking:
                    opponent_left_hand_list.remove(i)
            my_kicker_list = [n for i,n in my_left_hand_list] # 숫자만 따온다
            opponent_kicker_list = [n for i,n in opponent_left_hand_list] # 숫자만 따온다
            my_kicker_list.sort(reverse=True); opponent_kicker_list.sort(reverse=True) # 내림차순 정렬
            my_kicker_list = my_kicker_list[:-2]; opponent_kicker_list = opponent_kicker_list[:-2] # 뒤에서 2개 제거

            for n,i in enumerate(my_kicker_list):
                if i > opponent_kicker_list[n]:
                    return True # 승
                elif i < opponent_kicker_list[n]:
                    return False # 패
            return None # 무승부

        """
        else: # (my_rank == opponent_rank) # 족보 같음
            my_kicker_list = ...
            opponent_kicker_list = ...
            

            if my_rank == 2 : # 투페, 2 : tp, len : 2
                num_list = [i[0][1] for i in my_ranking_list]
                my_ranking_list = [my_ranking_list[num_list.index(max(num_list))]]

                num_list = [i[0][1] for i in opponent_ranking_list]
                opponent_ranking_list = [opponent_ranking_list[num_list.index(max(num_list))]]
                # <<<<

            elif my_rank == 6:  # 집, 6 : fh, ranking_list 길이 :2  # full house -> triple
                my_ranking_list = [i for i in my_ranking_list if len(i) == 3]
                my_kicker_list = [i for i in my_ranking_list if len(i) == 2]

                opponent_ranking_list = [i for i in opponent_ranking_list if len(i) == 3]
                opponent_kicker_list = [i for i in opponent_ranking_list if len(i) == 2]

                if len(my_ranking_list) == 2:
                    num_list = [i[0][1] for i in my_ranking_list]
                    my_ranking_list = [my_ranking_list[num_list.index(max(num_list))]]
                    my_kicker_list = [my_ranking_list[num_list.index(min(num_list))]]

                    num_list = [i[0][1] for i in opponent_ranking_list]
                    opponent_ranking_list = [opponent_ranking_list[num_list.index(max(num_list))]]
                    opponent_kicker_list = [opponent_ranking_list[num_list.index(min(num_list))]]

                


            # else:  # ranking_list 길이 : 1
            my_num = max([n for i,n in my_ranking_list[0]]) # 족보 탑카드
            opp_num = max([n for i,n in opponent_ranking_list[0]]) # 족보 탑카드

            if my_num > opp_num :
                return True
            elif my_num < opp_num :
                return False
            elif my_num == opp_num and my_rank in [4,5,8,9]: # 플, 스, 스플, 로티플 -> 무승부
                return None # 무승부
            else: # (my_num == opp_num)  # kicker 비교  # 모양 비교 shdc
                ...
                # my_shape = [i for i,n in my_ranking_list[0] if n == my_num]
                # opp_shape = [i for i,n in opponent_ranking_list[0] if n == opp_num]
                
                # shape_dict = {'♠':4, '♡':3, '♢':2, '♣':1}
                # my_shape = [shape_dict[i] for i in my_shape]
                # opp_shape = [shape_dict[i] for i in opp_shape]

                # max_my_shape = max(my_shape)
                # max_opp_shape = max(opp_shape)

                # if max_my_shape > max_opp_shape :
                #     return True
                # elif max_my_shape < max_opp_shape :
                #     return False
                # else : # 모양 같음 == 커뮤니티 카드가 가장 높은 족보
                #     ...
            """




if __name__ == '__main__':
    env = env()
    community_card = env.draw(input_deck=[], num=5)
    print(community_card)
    
    hand = env.draw(input_deck=[], num=2)
    print('Opponent Hand :', hand)

    hand2 = env.draw(input_deck=[], num=2)
    print('My Hand :', hand2)
    
    results = env.compare_hand(community_card=community_card, opponent_hand=hand, my_hand=hand2)
    if results == True:
        print('승')
    elif results == None:
        print('무')
    else :
        print('패')


    # community_card = [('♣', 6), ('♣', 3), ('♣', 8), ('♣', 7), ('♣', 4)]
    # # community_card = [('♣', 14), ('♠', 14), ('♣', 6), ('♠', 6),  ('♣', 5)]
    # # hand = env.draw(input_deck=[('♣', 5)], num=1)
    # # hand2 = env.draw(input_deck=[], num=2)
    # hand = [('♡', 5), ('♡', 2)]
    # hand2 = [('♣', 5), ('♢', 9)]
    # print(community_card)
    # print(hand2)
    # print(hand)

    # print(env.compare_hand(community_card=community_card, opponent_hand=hand2, my_hand=hand))




    # rank = env.hand_ranking_check(hand+community_card) # 족보 int
    # ranking_list = env.ranking_list  # 족보 리스트 list

    # print(rank, ranking_list)


    # rank2 = env.hand_ranking_check(hand2+community_card) # 족보 int
    # ranking_list2 = env.ranking_list  # 족보 리스트 list

    # print(rank2, ranking_list2)


    # community_card = env.draw(input_deck=community_card, num=2)
    # print(community_card)

    # rank = env.hand_ranking_check(hand+community_card) # 족보 int
    # ranking_list = env.ranking_list  # 족보 리스트 list

    # print(rank, ranking_list)


    # rank2 = env.hand_ranking_check(hand2+community_card) # 족보 int
    # ranking_list2 = env.ranking_list  # 족보 리스트 list

    # print(rank2, ranking_list2)




# Traceback (most recent call last):
#   File "c:\Users\82102\Desktop\poker\simulation.py", line 152, in <module>
#     action_num = sim(num=500, community_card=community_card, my_hand=hand, opponent_hand=hand2, opponent_action_num=my_action, pot_size=0, loss=0)
#   File "c:\Users\82102\Desktop\poker\simulation.py", line 19, in __call__
#     _loss, pot_size = self.simulation(opponent_action=opponent_action_num, agent_action=action, pot_size=pot_size, loss=loss) # <<<
#   File "c:\Users\82102\Desktop\poker\simulation.py", line 88, in simulation
#     results = env.compare_hand(community_card=self.community_card, opponent_hand=self.opponent_hand, my_hand=self.my_hand)
#   File "c:\Users\82102\Desktop\poker\Texasholdem.py", line 208, in compare_hand
#     opponent_left_hand_list.remove(i)
# ValueError: list.remove(x): x not in list

