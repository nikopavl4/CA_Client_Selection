import random
import torch.nn as nn
import numpy as np

class RandomSelector:
    def __init__(self, fraction):
        self.fraction = fraction


    def sample_clients(self, client_list):  # default parameters
        available_clients = client_list
        if len(available_clients) == 0:
            print(f"Cannot sample clients. The number of available clients is zero.")
            return []
        num_selection = int(self.fraction * len(available_clients))
        if num_selection == 0:
            num_selection = 1
        if num_selection > len(available_clients):
            num_selection = len(available_clients)
        sampled_clients = random.sample(available_clients, num_selection)
        print(f"Parameter c={self.fraction}. Sampled {num_selection} client(s): {[cl.id for cl in sampled_clients]}")
        return sampled_clients

def top_n_indexes(arr, n):
    idx = np.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]    
    
class CASelector:
    def __init__(self, fraction, X, Y):
        self.fraction = fraction
        self.X = X
        self.Y = Y
        self.PF = np.zeros((X, Y))
        #self.SQ = np.zeros((X, Y))
        self.NPC = np.zeros((X, Y))
        #self.IS = np.zeros((X, Y))
        self.TC = np.zeros((X, Y))
        #self.CC = np.zeros((X, Y))
        #self.DQ = np.zeros((X, Y))
        self.label_CA = np.zeros((X, Y))
        self.C = np.zeros((X, Y))
        self.e = 0.75
        self.m = 0.25

    def sample_clients(self, client_list):  # default parameters
        d = random.random()
        new_PF = np.zeros((self.X, self.Y))
        new_NPC = np.zeros((self.X, self.Y))
        new_TC = np.zeros((self.X, self.Y))
        new_C = np.zeros((self.X, self.Y))
        for x in range(self.X):
            for y in range(self.Y):
                # Update NPC
                if self.PF[x,y]==1:
                    new_NPC[x,y] = 0
                else:
                    new_NPC[x,y] = self.NPC[x,y] + 1
                # Update TC
                if x == 0 and y == 0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list))
                elif x==0 and y!=0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list)  + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list) + self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list))
                elif x!=0 and y == 0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list) + self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list))
                
                elif x == self.X and y == self.Y:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                elif x==self.X and y!=self.Y:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list)  + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list) + self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                elif x!=self.X and y == self.Y:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list) + self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                
                else:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list)+ self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list)+self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list)+self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list))
                
                # Update C
                new_C[x,y] = [client_list[self.label_CA[x,y]].IS/len(client_list[self.label_CA[x,y]].vehicle_list) + new_NPC[x,y] + client_list[self.label_CA[x,y]].DQ] * (d*len(client_list[self.label_CA[x,y]].vehicle_list)/new_TC[x,y])

                # Update PF
                ind = top_n_indexes(new_C, int(len(client_list)*self.fraction))
                for i,j in ind:
                    new_PF[i,j] = 1

        self.PF = new_PF
        self.NPC = new_NPC
        self.TC = new_TC
        self.C = new_C

        # Create Sampled Clients List
        sampled_arr = np.multiply(new_PF, self.label_CA)
        sampled_list_id = list(sampled_arr[np.where(sampled_arr!=0)].astype(int))

        sampled_clients = []

        for client in client_list:
            if client.id in sampled_list_id:
                sampled_clients.append(client)
  

        return sampled_clients