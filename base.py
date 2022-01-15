from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Environment(ABC):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def get_name(self):
        return self.name
    
    @abstractmethod
    def reset(self): # return observation
        pass
    
    @abstractmethod
    def step(self, action): # return observation, reward, done, info
        pass 

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def get_nb_states(self):
        pass

    @abstractmethod
    def get_nb_actions(self):
        pass
    


class MDP(Environment):
    def __init__(self, nb_states, nb_actions, initial_state=None):
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.initial_state = initial_state
        self.current_state = initial_state
    
    def get_nb_states(self):
        return self.nb_states

    def get_nb_actions(self):
        return self.nb_actions
    
    def init_all_proba_to_hole(self, states=None):
        if states is None:
            states = range(self.nb_states)
        for s in states:
            for a in range(self.nb_actions):
                for n in range(self.nb_states):
                    if s==n:
                        p = 1
                    else:
                        p = 0
                    self.set_transition(s, a, n, p, 0)
                    
              
    def init_all_proba_to_fixed_value(self, value, states=None):
        if states is None:
            states = range(self.nb_states)
        for s in states:
            for a in range(self.nb_actions):
                for n in range(self.nb_states):
                    self.set_transition(s, a, s, value, 0)
                    
                    
    def reset(self): # return observation
        state = self.initial_state
        while state is None or state<0 or not self.is_valid(state):
            state = np.random.randint(0, self.nb_states)
        self.current_state = state
        return state
        
    
    def step(self, action): # return observation, reward, done, info
        o = self.current_state
        if not self.is_final(o):
            no = np.random.choice(self.nb_states, p=self.p(o, action))
            r = self.r(o, action, no)
            d = self.is_final(no)
            self.current_state = no
        else:
            no = o
            r = 0
            d = True
        return no, r, d, "(action "+str(action)+" - new state "+str(o)+")"
                   
    @abstractmethod
    def is_final(self, state):
        pass

    @abstractmethod
    def is_valid(self, state):
        pass
    
    @abstractmethod
    def p(self, state=None, action=None, next_state=None):
        pass
    
    @abstractmethod
    def r(self, state=None, action=None, next_state=None):
        pass
        
    @abstractmethod
    def set_transition(self, state, action, next_state, proba, reward):
        pass

    @abstractmethod
    def set_transition_proba(self, state, action, next_state, proba):
        pass

    @abstractmethod
    def set_transition_reward(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def render_values(self, values, precision):
        pass
    
    @abstractmethod
    def render_policy(self, policy):   
        pass

    def observe_episode(self, policy, limit):
        obs = self.reset()
        totalreward = 0
        for n in range(limit):
            print("\n----- Step", n, "-----")
            self.render()
            action = np.random.choice(self.get_nb_actions(),p=policy[obs])
            obs, reward, done, _ = self.step(action)                    
            totalreward += reward
            print("Action", action, "is performed")
            print("Reward", reward, "is obtained")
            if done:
                rep = input("- Continuer (y/n) ? ")
                if rep=='n':break
                print("\n----- Terminal state -----")
                self.render()
                break
            if n==limit-1:
                print("\nStep limit reached !")           
                break
            
            rep = input("- Continuer (y/n) ? ")
            if rep=='n':break
        return n+1, totalreward
    
    def perform_episode(self, policy, limit):
        obs = self.reset()
        totalreward = 0
        for n in range(limit):
            action = np.random.choice(self.get_nb_actions(),p=policy[obs])
            obs, reward, done, _ = self.step(action)                    
            totalreward += reward
            if done: break
        return n+1, totalreward

    
class TabularMDP(MDP):
    def __init__(self, nb_states, nb_actions, initial_state=None):
        super().__init__(nb_states, nb_actions, initial_state)
        self.probas = np.zeros((nb_states, nb_actions, nb_states))
        self.rewards = np.zeros((nb_states, nb_actions, nb_states))

    def __str__(self):
        res = ""
        sa, aa, na = np.where(self.probas > 0)
        for s,a,n in zip(sa, aa, na):
            res += str(s)+' - '+str(a)+' - '+str(n)+': '+str(self.probas[s, a, n])+' '+str(self.rewards[s, a, n])+'\n'
        return res
        
    def get_max_reward(self):
        return np.max(self.rewards)
    
    def p(self, state=None, action=None, next_state=None):
        if state is None:
            state = slice(None)
        if action is None:
            action = slice(None)
        if next_state is None:
            next_state = slice(None)
        return self.probas[state, action, next_state]
    
    def r(self, state=None, action=None, next_state=None):
        if state is None:
            state = slice(None)
        if action is None:
            action = slice(None)
        if next_state is None:
            next_state = slice(None)
        return self.rewards[state, action, next_state]
        
    def set_transition(self, state, action, next_state, proba, reward):
        self.probas[state, action, next_state] = proba
        self.rewards[state, action, next_state] = reward

    def set_transition_proba(self, state, action, next_state, proba):
        self.probas[state, action, next_state] = proba
        
    def set_transition_reward(self, state, action, next_state, reward):
        self.rewards[state, action, next_state] = reward
       
    def init_all_proba_to_hole(self, states=None):
        if states is None:
            states = range(self.nb_states)
        self.probas[states, :, :] = 0
        self.probas[states, :, states] = 1
              
    def init_all_proba_to_fixed_value(self, value, states=None, to=None):
        if states is None:
            states = range(self.nb_states)
        if to is None:
            to = slice(None)
        self.probas[states, :, to] = value

class FL_MDP(TabularMDP):
    def __init__(self, env_cells, nline, ncol, slippery=True):
        super().__init__(len(env_cells), 4, 0)
        self.nb_lines = nline
        self.nb_columns = ncol
        self.cells = env_cells
        self.slippery = slippery
        self.init_transitions(env_cells, nline, ncol)
    
    def is_final(self, state):
        return self.cells[state]=='G' or self.cells[state]=='H'

    def is_valid(self, state):
        return self.cells[state]!='G' and self.cells[state]!='H'
        
    def init_transitions(self, env, nline, ncol):
        for i in range(nline):
            for j in range(ncol):
                state = i*ncol+j
                cell = env[state]
                if cell=='H' or cell=='G':
                    self.init_all_proba_to_hole(states=[state])
                else:
                    an = [state-1, state+ncol, state+1, state-ncol]
                    if j==0: 
                        an[0] = state
                    if j==ncol-1:
                        an[2] = state
                    if i==0:
                        an[3] = state
                    if i==nline-1:
                        an[1] = state
                        
                    r = [0]*4
                    for ni in range(4):
                        if env[an[ni]]=='G':
                            r[ni] = 1

                    if not self.slippery:
                        for a in range(4):
                            self.set_transition(state, a, an[a], 1, r[a])
                    else:
                        for a in range(4):
                            p = {}
                            for o in range(4):
                                p[an[o]] = 0
                            for o in [(a-1)%4 ,a, (a+1)%4]:
                                p[an[o]] += 1.0/3.0
                            for n in p:
                                if env[n]=='G':
                                    r = 1
                                else:
                                    r = 0
                                self.set_transition(state, a, n, p[n], r)


    def render(self):   
        res = ""
        for i, v in enumerate(self.cells):
            if self.current_state == i:
                res+='X '
            elif v == 'F':
                res+='_ '
            else:
                res += v+' '
            if (i+1)%self.nb_columns==0:
                res+="\n"
        
        print(res)

    def render_values(self, values, precision=3):   
        res = ""
        for i, v in enumerate(self.cells):
            res += str(round(values[i],precision)).ljust(precision+2, '0')+' '
            if (i+1)%self.nb_columns==0:
                res+="\n"
        
        print(res)

    def render_values_img(self, values):   
        return plt.imshow(values.reshape((self.nb_lines, self.nb_columns))) 
    
    def render_policy(self, policy):   
        res = ""
        for i, v in enumerate(self.cells):
            res += str(np.argmax(policy[i]))+' '
            if (i+1)%self.nb_columns==0:
                res+="\n"
        
        print(res)

    def render_policy_img(self, policy):  
        a = np.argmax(policy, axis=1)
        return plt.imshow(a.reshape((self.nb_lines, self.nb_columns)))
    
class FrozenLake44(FL_MDP):
    def __init__(self, slippering=True):
        super().__init__("SFFFFHFHFFFHHFFG", 4, 4, slippering)
        self.name="FrozenLake_4x4"
        
        
        
class FrozenLake88(FL_MDP):
    def __init__(self, slippering=True):
        super().__init__("SFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG", 8, 8, slippering)
        self.name="FrozenLake_8x8"
        

class GridWorld(TabularMDP):
    def __init__(self, env_cells, nline, ncol, final_cells, cell_rewards, blocked_chars='W', initial_state_char=None, teleporters={}):
        super().__init__(len(env_cells), 4, None if initial_state_char is None else env_cells.find(initial_state_char))
        self.nb_lines = nline
        self.nb_columns = ncol
        self.cells = env_cells
        self.final_cells = final_cells
        self.cell_rewards = cell_rewards
        self.blocked_chars = blocked_chars
        self.teleporters = teleporters
        self.wall_char='\u25A0'
        self.init_transitions(env_cells, nline, ncol)
    
    def is_final(self, state):
        return self.cells[state] in self.final_cells

    def is_valid(self, state):
        return  self.cells[state] not in self.final_cells and self.cells[state] not in self.blocked_chars
        
    def init_transitions(self, env, nline, ncol):
        for i in range(nline):
            for j in range(ncol):
                state = i*ncol+j
                cell = env[state]
                if cell in self.final_cells or cell in self.blocked_chars:
                    self.init_all_proba_to_hole(states=[state])
                else:
                    an = [state-1, state+ncol, state+1, state-ncol]
                    if j==0: 
                        an[0] = state
                    if j==ncol-1:
                        an[2] = state
                    if i==0:
                        an[3] = state
                    if i==nline-1:
                        an[1] = state
                    
                    for ni in range(4):
                        if env[an[ni]] in self.blocked_chars:
                            an[ni] = state

                    for ni in range(4):
                        if env[an[ni]] in self.teleporters:
                            an[ni] = env.find(self.teleporters[env[an[ni]]])
                        
                    r = [0]*4
                    for ni in range(4):
                        r[ni] = self.cell_rewards.get(env[an[ni]], 0)

                    for a in range(4):
                        self.set_transition(state, a, an[a], 1, r[a])




    def render(self):   
        res = (self.wall_char+' ') * (self.nb_columns + 2)+'\n'
        
        for i, v in enumerate(self.cells):
            if i%self.nb_columns==0: res+=self.wall_char+' '
            
            if self.current_state == i:
                res+='X'
            elif v == 'W':
                res+=self.wall_char
            else:
                res += v
            res+=' '
            if (i+1)%self.nb_columns==0:
                res+=self.wall_char+"\n"

        res +=  (self.wall_char+' ') * (self.nb_columns + 2)+'\n'
        print(res)

    def render_values(self, values, precision=3):   
        res = self.wall_char+' '+(self.wall_char*(precision+2)+' ') * (self.nb_columns) +self.wall_char+'\n'
        for i, v in enumerate(self.cells):
            if i%self.nb_columns==0: res+=self.wall_char+' '

            if v == 'W':
                res+=self.wall_char*(precision+2)
            else:
                res += str(round(values[i],precision)).ljust(precision+2, '0')

            res+=' '
        
            if (i+1)%self.nb_columns==0:
                res+=self.wall_char+"\n"

        res += self.wall_char+' '+(self.wall_char*(precision+2)+' ') * (self.nb_columns)  +self.wall_char+'\n'
        print(res)
    
    def render_values_img(self, values):   
        plt.imshow(values.reshape((self.nb_lines, self.nb_columns)))
        plt.show()
        
    def render_policy(self, policy):   
        res = ""
        for i, v in enumerate(self.cells):
            res += str(np.argmax(policy[i]))+' '
            if (i+1)%self.nb_columns==0:
                res+="\n"
        
        print(res)
        
        
    def render_policy_img(self, policy):  
        a = np.argmax(policy, axis=1)
        plt.imshow(a.reshape((self.nb_lines, self.nb_columns)))
        plt.show()

class Maze(GridWorld):
    def __init__(self, initial_state=None):
        super().__init__("A__G_W_W___W_W_F", 4, 4, ['G', 'F'], {'G':400, 'F':1000}, initial_state_char='A')
        self.name="Maze"

class FoorRooms_Key(GridWorld):
    def __init__(self, initial_state=11):
        super().__init__("____W_____A__C________W__G_____W____W_WWWWWWW____W________W_K_______________W____WWWWWWWWW____W_________________W__G_____W____W_WWWWWWW____W________W_T_______________W____", 
                             19, 9, ['G'], {'G':1}, blocked_chars='WC', initial_state_char='A', teleporters={'K':'T'})
        self.name="FoorRooms_Key"    

class CliffWalking(GridWorld):
    def __init__(self, wind=0):
        super().__init__("____________________________________AHHHHHHHHHHG", 4, 12, ['H', 'G'], {'A':-1,'_': -1, 'H':-100}, initial_state_char='A')
        self.wind = wind
        if self.wind>0:
            self.probas[:36,[0,2],:] *= 1-self.wind
            for s in range(36):
                self.set_transition(s, 0, s+12, self.wind, self.cell_rewards.get(self.cells[s+12], 0))
                self.set_transition(s, 2, s+12, self.wind, self.cell_rewards.get(self.cells[s+12], 0))
#                self.set_transition(s, 3, s+12, self.wind, self.cell_rewards.get(self.cells[s+12], 0))
        
            self.probas[36,[2],:] *= 1-self.wind
            self.set_transition(36, 2, 36, self.wind, self.cell_rewards.get(self.cells[36], 0))
        self.name="CliffWalking"
    