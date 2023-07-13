import random
import torch.nn as nn
import numpy as np
import time

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
    def __init__(self, fraction, X, Y, selector_type):
        self.fraction = fraction
        self.X = X
        self.Y = Y
        self.PF = np.ones((X, Y))
        #self.SQ = np.zeros((X, Y))
        self.NPC = np.zeros((X, Y))
        #self.IS = np.zeros((X, Y))
        self.TC = np.zeros((X, Y))
        #self.CC = np.zeros((X, Y))
        #self.DQ = np.zeros((X, Y))
        self.label_CA = np.arange(0, (X*Y)).reshape((X, Y)).astype(int)
        self.C = np.zeros((X, Y))
        self.e = 0.75
        self.m = 0.25
        self.alpha = 0.33
        self.beta = 0.33
        self.gamma = 0.33
        self.selector_type = selector_type

    def sample_clients(self, client_list, busted):  # default parameters
        d = random.random()
        new_PF = np.zeros((self.X, self.Y))
        helper = np.zeros((self.X, self.Y))
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
                
                elif x==0 and y==self.Y-1:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list))
                    
                elif x==self.X-1 and y == 0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list))
                    
                elif x==0 and y!=0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list)  + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list) + self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list))
                
                elif x!=0 and y == 0:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list) + self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list))

                
                
                elif x == self.X-1 and y == self.Y-1:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                
                elif x==self.X-1 and y!=self.Y-1:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list)  + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list)) + \
                            self.m*(self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list) + self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                elif x!=self.X-1 and y == self.Y-1:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list) + self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list))
                
                else:
                    new_TC[x,y] = self.PF[x,y]*len(client_list[self.label_CA[x,y]].vehicle_list) + \
                            self.e*(self.PF[x+1,y]*len(client_list[self.label_CA[x+1,y]].vehicle_list) + self.PF[x,y+1]*len(client_list[self.label_CA[x,y+1]].vehicle_list) + self.PF[x,y-1]*len(client_list[self.label_CA[x,y-1]].vehicle_list) + self.PF[x-1,y]*len(client_list[self.label_CA[x-1,y]].vehicle_list)) + \
                            self.m*(self.PF[x+1,y+1]*len(client_list[self.label_CA[x+1,y+1]].vehicle_list)+ self.PF[x-1,y-1]*len(client_list[self.label_CA[x-1,y-1]].vehicle_list)+self.PF[x-1,y+1]*len(client_list[self.label_CA[x-1,y+1]].vehicle_list)+self.PF[x+1,y-1]*len(client_list[self.label_CA[x+1,y-1]].vehicle_list))
                
                # Update C
                temp1 = self.alpha*client_list[self.label_CA[x,y]].IS/len(client_list[self.label_CA[x,y]].vehicle_list) + self.beta*new_NPC[x,y] + (1*self.gamma/float(client_list[self.label_CA[x,y]].DQ))
                temp2 = d*len(client_list[self.label_CA[x,y]].vehicle_list)/(new_TC[x,y]+0.0001)
                new_C[x,y] =  float(temp1)*temp2

        # Update PF
        ind = top_n_indexes(new_C, int(len(client_list)*self.fraction))
        for i,j in ind:
            new_PF[i,j] = 1

        # Compute Slowed Clients
        ind2 = top_n_indexes(new_TC, int(len(client_list)*0.2))
        for i, j in ind2:
            helper[i,j] = 1

        # Create Slow Clients List
        slowed_arr = np.multiply(helper, self.label_CA)
        slowed_list_id = list(slowed_arr[np.where(slowed_arr!=0)].astype(int))

        slowed_clients = []

        for client in client_list:
            if client.id in slowed_list_id:
                slowed_clients.append(client)


        self.PF = new_PF
        self.NPC = new_NPC
        self.TC = new_TC
        self.C = new_C

        if self.selector_type == "cellular":
            # Create Sampled Clients List
            sampled_arr = np.multiply(new_PF, self.label_CA)
            sampled_list_id = list(sampled_arr[np.where(sampled_arr!=0)].astype(int))

            sampled_clients = []

            for client in client_list:
                if client.id in sampled_list_id:
                    sampled_clients.append(client)
        
        elif self.selector_type == "random":
            available_clients = client_list
            num_selection = int(self.fraction * len(available_clients))
            if num_selection == 0:
                num_selection = 1
            if num_selection > len(available_clients):
                num_selection = len(available_clients)
            sampled_clients = random.sample(available_clients, num_selection)           

        
        # Add time overhead for choosing slowed clients
        for client in sampled_clients:
            if client.id in [cl.id for cl in slowed_clients]:
                print("Buuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuusted")
                busted = busted + 1
                time.sleep(5)
  
        print(f"Parameter c={self.fraction}. Sampled {len(sampled_clients)} client(s): {[cl.id for cl in sampled_clients]}")
        return sampled_clients, busted