# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:56:08 2020

@author: ryohe
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy

class fictitious_game():
    
    def __init__(self, player_num, action_num):
        self.emp_dist_ndiv = [ [0]*action_num for i in range(player_num) ]
        #self.actions_list = [ [action for action in range(action_num)] for player in player_num ]
        self.actions = [i for i in range(action_num)]
        self.players = [i for i in range(player_num)]
        self.player_num = player_num
        self.action_num = action_num
        self.dist_history = [[[0] for j in range(self.action_num)] for i in range(self.player_num)]
        #self.init_actions = [ 0 for i in range(player_num)]
        
        
    def Utility(self,actions):
        one = np.zeros(3)
        two = one + 1 
        #three = two + 1
        #print(two==actions)
        if all(one == actions):
            return 1
        elif all(two == actions):
            return 2
        #elif all(three == actions):
         #   return 3
        else:
            return 0
    def empirical_dist_init(self, init_actions):
        for player in self.players:
            self.emp_dist_ndiv[player][init_actions[player]] += 1
        print(self.emp_dist_ndiv)
    
    def empirical_dist(self, episode, player, action_prev):
        
        #print(1)
        if episode == 0:
            #print(1)
            return self.emp_dist_ndiv
            
        else:
            #print(player)
            #print(action_prev)
            self.emp_dist_ndiv[player][action_prev] += 1
            #print(self.emp_dist_ndiv)
            empirical_dist = np.array(self.emp_dist_ndiv)/(episode)
            return empirical_dist
    
    def policy_fictitious(self, episode, player_i, action_prev):
        players_except = [ j for j in self.players if j != player_i ]
        q = 1
        util = 0
        utils = []
        action_list = []
        
        empirical_dist = self.empirical_dist(episode, player_i, action_prev)
        #print(empirical_dist)
        #print(np.array(self.emp_dist_ndiv)/1)
                
        for action in self.actions:
            #actions_except = [ j for j in self.actions if j != action]
            action_list_all = []
            
            for j in players_except:
                q *= empirical_dist[j][action]
            
            #print(q)
            #残り2プレイヤー分の行動リストの生成
            for action1 in self.actions:
                for action2 in self.actions:
                    action_list.append(action1)
                    action_list.append(action2)
                    action_list.insert(player_i, action)
                    action_list_all.append(action_list)
                    
                    action_list = []
                    
            for actions in action_list_all:      
                util += self.Utility(actions)
                
            utils.append(util)
            #print(len(utils))
            
            util = 0
            q = 1
        #utils = np.array(utils)
        return np.argmax(utils)
    
    def policy_log_linear(self, player, actions_prev, tau_inv):
        action_probs = []

        actions_prev_except = copy.copy(actions_prev)
        for action in self.actions:
            del actions_prev_except[player]
            actions_prev_except.insert(player, action)
            #if actions_prev_ecept == [1,1,1]:
            util = self.Utility(actions_prev_except)
            #print(util)
            action_probs.append( math.exp( tau_inv * util) )
            #print(math.exp( tau_inv * util))
            
        action_probs = np.array(action_probs)
        action_probs = action_probs/sum(action_probs)
        #print(action_probs)
        
        #if action_probs[0] != 0:
         #   print(util)
        
        #action = np.argmax(action_probs)
        np.random.seed(42)
        action = np.random.choice(self.actions, 1, list(action_probs))
        return action, action_probs
    
    def learn_fictitious(self, episode_num = 30, initial_action = [1,1,1]):
        
        #print(len(action_history))
        #dist = np.array(dist_history)
        #print(dist.shape)
        
        #emp_dist = copy.deepcopy(self.emp_dist_ndiv)
        #dist_history.append(np.array(emp_dist))
        
        #init_flag = 1
        
        #for action in self.actions:
         #   self.init_actions[action] = initial_action[action]
        
        self.empirical_dist_init(initial_action)
        
        for player in self.players:
            self.dist_history[player][initial_action[player]] = [1]
        
        count = 0
        
        episode_axis = [0]
        
        action_prev = initial_action
        action = action_prev       
        
        for episode in range(episode_num):
                
            for player in self.players:
                #print(action_prev[player])
                
                action[player] = self.policy_fictitious(episode, player, action_prev[player])  
                #print(action)
                action_prev[player] = action[player]
                #empirical_dist = self.empirical_dist(episode+1, player, action_prev)
        
            if episode % 1 == 0:
                episode_axis.append(episode)
                emp_dist = np.array(self.emp_dist_ndiv) / (episode+1)
                for player_i in self.players:
                    for action_i in self.actions:
                        count += 1
                        #print(player_i)
                        #print(action_i)
                        #print(emp_dist[player][action_i])
                        self.dist_history[player_i][action_i].append(emp_dist[player][action_i])
                        #print(len(self.dist_history[player_i][action_i]))
                
        empirical_dist = np.array(self.emp_dist_ndiv) / episode_num
        #print(empirical_dist)
        #print(2)
        dist = np.array(self.dist_history)
        #print(dist.shape)
        #print(count)
        
        
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        player1 = fig.add_subplot(3, 1, 1)
        player2 = fig.add_subplot(3, 1, 2)
        player3 = fig.add_subplot(3, 1, 3)
        
        #print(len(self.dist_history[1][1]))
        player1.plot(episode_axis, self.dist_history[0][1], color="blue")
        player2.plot(episode_axis, self.dist_history[1][1], color="red")
        player3.plot(episode_axis, self.dist_history[2][1], color="green")
        
        player1.set_ylabel("Pr for player 1")
        player2.set_ylabel("Pr for player 2")
        player3.set_ylabel("Pr for player 3")
        player3.set_xlabel("Learning step")
        
        plt.show()
        
        fig = plt.figure()
        player1 = fig.add_subplot(3, 1, 1)
        player2 = fig.add_subplot(3, 1, 2)
        player3 = fig.add_subplot(3, 1, 3)
        
        #print(len(self.dist_history[1][1]))
        player1.plot(episode_axis, self.dist_history[0][0], color="blue")
        player2.plot(episode_axis, self.dist_history[1][0], color="red")
        player3.plot(episode_axis, self.dist_history[2][0], color="green")
        
        player1.set_ylabel("Pr for player 1")
        player2.set_ylabel("Pr for player 2")
        player3.set_ylabel("Pr for player 3")
        player3.set_xlabel("Learning step")
        
        plt.show()
        
    def learn_log_linear(self, episode_num = 10000):
        
        count = 1
        
        episode_axis = [0]
        
        action_prev = [0,0,0]
        action = action_prev
        for episode in range(episode_num):
                
            for player in self.players:               
                action[player], action_probs = self.policy_log_linear(player, action_prev, tau_inv = 100)
                print(player)
                print(action[player])
                print(action_probs)
                
                action_prev[player] = action[player]
                print(action_prev)
        
                if episode % 1 == 0:
                    if player == 0:
                        episode_axis.append(episode)
                    for action_i in self.actions:
                        self.dist_history[player][action_i].append(action_probs[action_i])
            
        empirical_dist = np.array(self.emp_dist_ndiv) / episode_num
        dist = np.array(self.dist_history)       
        
        plt.style.use('default')
        sns.set()
        sns.set_style('whitegrid')
        sns.set_palette('gray')
        
        fig = plt.figure()
        player1 = fig.add_subplot(3, 1, 1)
        player2 = fig.add_subplot(3, 1, 2)
        player3 = fig.add_subplot(3, 1, 3)
        
        player1.plot(episode_axis, self.dist_history[0][1], color="blue")
        player2.plot(episode_axis, self.dist_history[1][1], color="red")
        player3.plot(episode_axis, self.dist_history[2][1], color="green")
        
        player1.set_ylabel("Pr for player 1")
        player2.set_ylabel("Pr for player 2")
        player3.set_ylabel("Pr for player 3")
        player3.set_xlabel("Learning step")
        
        plt.show()
        
        fig = plt.figure()
        player1 = fig.add_subplot(3, 1, 1)
        player2 = fig.add_subplot(3, 1, 2)
        player3 = fig.add_subplot(3, 1, 3)
        
        #print(len(self.dist_history[1][1]))
        player1.plot(episode_axis, self.dist_history[0][0], color="blue")
        player2.plot(episode_axis, self.dist_history[1][0], color="red")
        player3.plot(episode_axis, self.dist_history[2][0], color="green")
        
        player1.set_ylabel("Pr for player 1")
        player2.set_ylabel("Pr for player 2")
        player3.set_ylabel("Pr for player 3")
        player3.set_xlabel("Learning step")
        
        plt.show()

if __name__ == "__main__":        
    f_game = fictitious_game(player_num = 3, action_num = 2)
    #f_game.learn_fictitious()
    f_game.learn_log_linear(episode_num = 100)         
                
#util = Utility([2,2,2])
#print(util)