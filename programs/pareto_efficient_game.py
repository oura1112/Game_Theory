# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:39:14 2020

@author: ryohe
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy

class pareto_game():
    
    def __init__(self, player_num, action_num):
        self.emp_dist_ndiv = [ [0]*action_num for i in range(player_num) ]
        self.actions = [i for i in range(action_num)]
        self.players = [i for i in range(player_num)]
        self.player_num = player_num
        self.action_num = action_num
        self.dist_history = [[[0] for j in range(self.action_num)] for i in range(self.player_num)]
        self.player_state = np.array( [ [0,0,0] for i in range(self.player_num) ] )
        self.epsilon = 0.01
    
    def Initialize(self):
        
        for i in self.players :
            self.player_state[i][0] = np.random.choice(self.actions)

        for i in self.players :
            self.player_state[i][1] = self.Utility( self.player_state[:, 0],i)
            
        for i in self.players :
            self.player_state[i][2] = 0
    
    def Utility(self,actions,player_i):
        if actions[player_i] == 0:
            return 2/(3*self.player_num) * np.count_nonzero(actions == 0)
        else :
            return 2/(3*self.player_num) * ( np.count_nonzero(actions == 0) + self.player_num/2 )
    
    def policy(self, player_i):
        c = 6
        actions_except = copy.deepcopy(self.player_state[:, 0])
        
        if self.player_state[player_i][2] == 0:
            if np.random.random() < self.epsilon**c :
                actions_except = np.delete(self.player_state[:, 0], player_i)
                return np.random.choice( actions_except )
            else :
                return self.player_state[player_i][0]
        
        else :
            return np.random.choice( actions_except )
    
    def state_update(self, current_actions, current_utils):
        
        for player_i in self.players :
            if self.player_state[player_i][2] != 0 or current_actions[player_i] != self.player_state[player_i][0] or current_utils[player_i] != self.player_state[player_i][1] :
                
                if np.random.random() < self.epsilon**(1 - current_utils[player_i]):
                    self.player_state[player_i][2] = 1
                else : 
                    self.player_state[player_i][2] = 0
                    
            self.player_state[player_i][0] = current_actions[player_i]
            self.player_state[player_i][1] = current_utils[player_i]
                
    def learn_pareto_efficient(self, episode_num = 100):
       
        self.Initialize()   
        
        episode_axis = []
        action_P1 = []
        action_P2 = []
        action_P3 = []
        action_P4 = []
        action_P5 = []
        
        for episode in range(episode_num):
            current_actions = []
            current_utils = []
            
            for player in self.players:
                
                action = self.policy(player)
                current_actions.append(action)
                
            for player in self.players:
                
                util = self.Utility(current_actions, player)
                current_utils.append(util)
                
            self.state_update(current_actions, current_utils)

            if episode % 2 == 0:
                episode_axis.append(episode)
                action_P1.append( self.player_state[0][0] + 1 )
                action_P2.append( self.player_state[1][0] + 1 )
                action_P3.append( self.player_state[2][0] + 1 )
                action_P4.append( self.player_state[3][0] + 1 )
                action_P5.append( self.player_state[4][0] + 1 )
        
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        player1 = fig.add_subplot(5, 1, 1)
        player2 = fig.add_subplot(5, 1, 2)
        player3 = fig.add_subplot(5, 1, 3)
        player4 = fig.add_subplot(5, 1, 4)
        player5 = fig.add_subplot(5, 1, 5)
        
        player1.plot(episode_axis, action_P1, color="blue")
        player2.plot(episode_axis, action_P2, color="red")
        player3.plot(episode_axis, action_P3, color="green")
        player4.plot(episode_axis, action_P4, color="purple")
        player5.plot(episode_axis, action_P5, color="orange")
        
        player1.set_ylabel("player 1")
        player2.set_ylabel("player 2")
        player3.set_ylabel("player 3")
        player4.set_ylabel("player 4")
        player5.set_ylabel("player 5")
        player5.set_xlabel("Learning step")
        
        plt.show()
    
    
if __name__ == "__main__":        
    f_game = pareto_game(player_num = 5, action_num = 2)
    f_game.learn_pareto_efficient()
