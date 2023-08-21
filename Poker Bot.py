import os
import numpy as np
import pygame
import random
import time
from copy import deepcopy

pygame.init()

from outer_simulation import Simulation_simulation as Simulation
from Texasholdem import env


class Holdem:
    def __init__(self, main_page:bool=True):
        self.env = env(True)
        self.sim = Simulation()
        # simulation_num = 600

        start_money = 30
        self.my_money = start_money; self.opponent_money = start_money
        self.pot_size = 0#; betting = 1
        self.betting_phase = False # 베팅인지 아닌지
    
        if main_page:
            self.main_page()
    

    def main_page(self):
        self.WINDOW_WIDTH =  1400 # 1910/10*7  image_scale : 1910*1100
        self.WINDOW_HEIGHT = 750

        img_path = os.path.abspath('img')
        
        card_size = (100, 150)
        self.card_img = [self.spade_img, self.heart_img, \
         self.diamond_img, self.clover_img] = [ [pygame.transform.scale(   # 크기 조정
            pygame.image.load(  # 이미지 로드
            os.path.join( img_path, str(num)+shape+'.png' )  # 주소 결합
            ), card_size)  
            for num in range(2, 15)]    # 숫자 2~14까지
            for shape in ['S', 'H', 'D', 'C']]   # 모양 스,하,다,클

        button_size = (150,75)
        self.button_img = \
            [self.check_button_img, self.raise_button_img, self.fold_button_img, self.call_button_img] = [pygame.transform.scale(
            pygame.image.load(os.path.join(img_path, f'{i}_button.png'))
            , button_size )
              for i in ['Check','Raise','Fold','Call']]


        self.back_ground_img = pygame.image.load(os.path.join(img_path, "background.jpg"))  # 배경 사진 로드
        self.back_ground_img = pygame.transform.scale(self.back_ground_img, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))  # (self.WINDOW_WIDTH*2, self.WINDOW_HEIGHT*2))
        
        self.back_img = pygame.image.load(os.path.join(img_path, "back.png"))  # 배경 사진 로드
        self.back_img = pygame.transform.scale(self.back_img, card_size)
        
        self.chip_img = pygame.image.load(os.path.join(img_path, "chip.png"))  # 칩 사진 로드
        self.chip_img = pygame.transform.scale(self.chip_img, (100, 100))

        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT)) #화면 크기 설정
        pygame.display.set_caption('Texas Holdem')
        self.clock = pygame.time.Clock() 
        
        
    def main(self):
        self.screen.fill((255, 255, 255)) #단색으로 채워 화면 지우기
        self.screen.blit(self.back_ground_img, (0,0))#(-1*self.WINDOW_WIDTH/2, -1*self.WINDOW_HEIGHT/2) )  # 배경 화면

        event = pygame.event.poll() #이벤트 처리

        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # pos = self.mouse_pos()
            pos = pygame.mouse.get_pos()
            pos = self.check_mouse_pos(pos)
            return pos if not pos == None else None


    # def mouse_pos(self):
    #     return pygame.mouse.get_pos()
    

    def check_mouse_pos(self, pos:tuple) -> tuple:
        x,y = pos; button_width, button_height = (150, 75)
        for n, (_x, _y) in enumerate(self.button_pos):
            if _x < x < _x + button_width and _y < y < _y + button_height:
                if self.betting_phase and n == 1:
                    return n+1
                return n


    def show_cards(self, community_cards:list, my_cards:list, opponent_cards:list):
        cards_width, cards_height = (100, 150)
        my_img_pos = [((self.WINDOW_WIDTH + cards_width)/7*x - cards_width, (self.WINDOW_HEIGHT*2-cards_height)/3) for x in range(1,7)]
        opponent_img_pos = [((self.WINDOW_WIDTH + cards_width)/7*x - cards_width, (self.WINDOW_HEIGHT-cards_height*2)/3-100) for x in range(1,7)]
        community_img_pos = [((self.WINDOW_WIDTH + cards_width)/6*x - cards_width, (self.WINDOW_HEIGHT-cards_height)/2-50) for x in range(1,6)]

        for n,i in enumerate(my_cards):
            self.screen.blit(self.card_img[i[0]][i[1]-2], my_img_pos[n+2] )
        for n,i in enumerate(opponent_cards):
            self.screen.blit(self.card_img[i[0]][i[1]-2], opponent_img_pos[n+2] )
        for n,i in enumerate(community_cards):
            self.screen.blit(self.card_img[i[0]][i[1]-2], community_img_pos[n] )


    def show_buttons(self, betting_phase:bool=False):
        button_width, button_height = (150, 75)
        self.button_pos = [((self.WINDOW_WIDTH+button_width)/4*x - button_width, self.WINDOW_HEIGHT-button_height-30) for x in range(1,4)]
        if betting_phase:
            self.button_pos = [self.button_pos[n] for n in [0,2]]
            for n,i in enumerate(self.button_pos[::-1]):
                self.screen.blit(self.button_img[n+2], i)
        else:
            for n,i in enumerate(self.button_pos):
                self.screen.blit(self.button_img[n], i)


    def hidden_cards(self, opponent_hidden:list):
        cards_width, cards_height = (100, 150)
        opponent_img_pos = [((self.WINDOW_WIDTH + cards_width)/7*x - cards_width, (self.WINDOW_HEIGHT-cards_height*2)/3-100) for x in range(1,7)]

        for i in opponent_hidden:
            self.screen.blit(self.back_img, opponent_img_pos[i[0]])
        

    def money_text(self, my_money:float, opponent_money: float):
        font = pygame.font.SysFont('arial', 45, True, False)
        self.text = font.render(f'{my_money}$ vs {opponent_money}$', True, (255, 255, 255))
        self.screen.blit(self.text, (40,30))


    def pot_text(self, pot_size:float):
        font = pygame.font.SysFont('arial', 45, True, False)
        self.text = font.render(f'{pot_size}$', True, (255, 255, 255))
        self.screen.blit(self.text, (self.WINDOW_WIDTH - 100, 30))


    def show_computer_action_txt(self, action_num:int):
        action_txt = ['Call', 'Raise', 'Fold'][action_num]
        font = pygame.font.SysFont('arial', 100, True, True)
        txt_color = (0, 0, 0)
        action_text = font.render(f'{action_txt}', True, txt_color)
        self.screen.blit(action_text, (self.WINDOW_WIDTH/2 - 100, self.WINDOW_HEIGHT/2 - 100))


    def results_txt(self, results:bool):
        if results == None:
            txt = 'Draw'
        elif results:
            txt = 'Win'
        else :
            txt = 'Lose'
        font = pygame.font.SysFont('arial', 100, True, True)
        txt_color = (0, 0, 0)
        action_text = font.render(f'{txt}', True, txt_color)
        self.screen.blit(action_text, (self.WINDOW_WIDTH/2 - 100, self.WINDOW_HEIGHT/2 - 100))


    def window_update(self):
        pygame.display.flip()


def save_data(_dir:str, txt:str):
    with open(_dir, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    data.append(txt+'\n')
    with open(_dir, 'w', encoding='UTF-8') as f:
        for i in data:
            f.write(i)


if __name__ == "__main__":
    poker = Holdem(main_page=True)
    opponent_hand = poker.env.draw([], 2)
    my_hand = poker.env.draw([], 2)
    community_card = poker.env.draw([], 3)

    print(opponent_hand, my_hand)

    opponent_hidden_pos = [(2,0), (3,0)]
    my_money, opponent_money = 30, 30
    pot_size = 0; my_loss = 0; opponent_loss = 0
    betting_amount = 2
    
    simulation_num = 4

    my_money -= 1; opponent_money -= 1 # 참가비 1$
    my_loss -= 1; opponent_loss -= 1
    pot_size += 2


    poker.betting_phase = False #  raise check
    game_end = False; fold = False; fold_player = None
    
    while True:
        action = poker.main()
        poker.show_cards(community_cards=community_card, my_cards=my_hand, opponent_cards=opponent_hand)
        poker.hidden_cards(opponent_hidden=opponent_hidden_pos)
        poker.show_buttons(poker.betting_phase)
        poker.money_text(my_money, opponent_money)
        poker.pot_text(pot_size)
        poker.window_update()

        if not action == None:
            save_data('log.txt', txt=f'{community_card} / {my_hand} / {opponent_hand} / {action} / {betting_amount} / {pot_size}') # <<<
            poker.main()
            poker.show_cards(community_cards=community_card, my_cards=my_hand, opponent_cards=opponent_hand)
            poker.hidden_cards(opponent_hidden=opponent_hidden_pos)
            poker.money_text(my_money, opponent_money)
            poker.pot_text(pot_size)
            poker.window_update()

            print(action)
            if action == 2: # fold
                game_end = True
                fold = True
                fold_player = True  # 나
                opponent_action = None # <<<

                poker.main()  # <<<
                poker.show_cards(community_cards=community_card, my_cards=my_hand, opponent_cards=opponent_hand) # <<<
                poker.money_text(my_money=my_money, opponent_money=opponent_money) # <<<
                poker.results_txt(results=False) # <<<
                poker.window_update() # <<<
                pygame.time.delay(5000) # <<<
                
            elif action == 1: # raise
                my_money -= betting_amount; pot_size += betting_amount
                my_loss -= betting_amount
                betting_amount *= 2
            elif action == 0: # check
                None
            
            if not poker.betting_phase and not fold:
                opponent_action = poker.sim(num=simulation_num, community_card=community_card, 
                                            my_hand=opponent_hand, opponent_hand=my_hand, 
                                            opponent_action_num=action, 
                                            pot_size=pot_size, loss=opponent_loss,
                                            shape_to_num=True)
                print('Computer Action :', opponent_action)
                poker.show_computer_action_txt(action_num=opponent_action)
                poker.window_update()                
                pygame.time.delay(1500) # 밀리초 1000 -> 1s
                
                if opponent_action == 0: # call 
                    if betting_amount == 4 : # raise - call
                        opponent_money -= betting_amount/2; pot_size += betting_amount/2
                        opponent_loss -= betting_amount/2
                        betting_amount = 2
                    if len(community_card) == 3 :
                        community_card = poker.env.draw(input_deck=community_card, num=2)
                    elif len(community_card) == 5:
                        results = poker.env.compare_hand(community_card=community_card, opponent_hand=opponent_hand, my_hand= my_hand)
                        game_end = True
                elif opponent_action == 1: # raise
                    poker.betting_phase = not poker.betting_phase # raise - raise
                    opponent_money -= betting_amount; pot_size += betting_amount
                    opponent_loss -= betting_amount
                elif opponent_action == 2: # fold
                    game_end = True
                    fold = True
                    fold_player = False # 컴퓨터
            elif poker.betting_phase and not fold: # check(raise) - raise - call
                poker.betting_phase = not poker.betting_phase
                my_money -= 2; pot_size += 2; my_loss -= 2
                if len(community_card) == 3 :
                    community_card = poker.env.draw(input_deck=community_card, num=2)
                    betting_amount = 2
                elif len(community_card) == 5:
                    results = poker.env.compare_hand(community_card=community_card, opponent_hand=opponent_hand, my_hand= my_hand)
                    game_end = True
                    # print(results)
            
            if game_end : 
                if not fold:
                    print(results)
                    if results == True: # 승
                        my_money += pot_size
                    elif results == False: # 패
                        opponent_money += pot_size
                    elif results == None:  # 무
                        my_money += pot_size/2
                        opponent_money += pot_size/2
                    
                    my_hand, opponent_hand = my_hand[:2], opponent_hand[:2]
                    poker.main()  #<<<
                    poker.show_cards(community_cards=community_card, my_cards=my_hand, opponent_cards=opponent_hand)
                    poker.money_text(my_money=my_money, opponent_money=opponent_money)
                    poker.results_txt(results=results)
                    poker.window_update()
                    pygame.time.delay(5000)

                if fold:
                    if fold_player: # 내가 폴드
                        opponent_money += pot_size
                    else:  # 컴퓨터가 폴드
                        my_money += pot_size
                poker = Holdem(main_page=True)
                opponent_hand = poker.env.draw([], 2)
                my_hand = poker.env.draw([], 2)
                community_card = poker.env.draw([], 3)
                pot_size = 0; my_loss = 0; opponent_loss = 0
                betting_amount = 2
                fold, game_end = False, False
                fold_player == None
                print(my_money, opponent_money)

                my_money -= 1; opponent_money -= 1 # 참가비 1$
                my_loss -= 1; opponent_loss -= 1
                pot_size += 2
