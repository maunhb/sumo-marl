import numpy as np 
import random

class VariableElimination():
   
    def __init__(self, local_q_table, ordering, edges):
        '''
        Variables 
        q_functions:  a dictionary of local q tables, given state s
        function_e:  (dict) agents e functions
        functions_e_argmax:   (dict int matrices) best response tables - which action to take dependent on opponents
        functions_e_variables:  (dict int array) which variables the e function include (other than their own action)
        functions_e_used:   (dict bool) has e function been replaced by another e function 
        '''
        self.agents_to_eliminate = ordering
        self.q_functions = local_q_table 
        self.functions_e = {agent: None for agent in ordering} 
        self.functions_e_argmax = {agent: None for agent in ordering} 
        self.functions_e_variables = {agent: None for agent in ordering} 
        self.functions_e_used = {agent: False for agent in ordering}  
        self.elim_ordering = ordering
        self.coord_graph = edges 
        self.opt_action = {} 

    def VariableElimination(self):
        ''' 
        Eliminates agents by replacing their Q function with e functions
        Then agents choose their actions in the opposite order
        Returns dictionary of optimal action
        '''
        i = 0
        # eliminate agents and update functions
        while i < len(self.elim_ordering):
            agent2elim = self.elim_ordering[i] 
            connected_agents, connected_edge_indexes, agent_q_input = self.find_scope(agent2elim)
            self.new_function(agent2elim, connected_agents, connected_edge_indexes, agent_q_input)
            self.agents_to_eliminate = np.delete(self.agents_to_eliminate,0)
            i += 1
        #find optimal joint actions
        while i > 0:
            acting_agent = self.elim_ordering[i-1]
            arg = 0; other_agents = []
            for x in self.functions_e_variables[acting_agent]:
                if x in self.elim_ordering[i::]:
                    other_agents = np.append(other_agents,x)
                    arg +=1 
            if arg == 0:
                action = self.functions_e[acting_agent]
            else:
                action = self.chooseaction(arg, acting_agent, other_agents)
            self.opt_action.update({acting_agent: action})
            i -= 1
        print('Optimal actions:')
        print(self.opt_action)
        return self.opt_action 

    def find_scope(self, agent):
        ''' 
        Returns:
        scope  (list int) of neighbouring agents
        edge_indexes (list int) of agent's coorindation graph edges
        q_input (list bool) of whether the agent in the first or second node in the edge vector
        '''
        scope = []; edge_indexes = []; q_input = []
        for i in range(0,len(self.coord_graph),2):
            if self.coord_graph[i] == agent:
                scope = np.append(scope,self.coord_graph[i+1])
                edge_indexes = np.append(edge_indexes,i) 
                q_input = np.append(q_input,0)
            if self.coord_graph[i+1] == agent:
                scope = np.append(scope,self.coord_graph[i])
                edge_indexes = np.append(edge_indexes,i)
                q_input = np.append(q_input,1)
        return scope, edge_indexes, q_input

    def new_function(self, agent, scope, edge_id, q_function_index):
        '''
        updates e function of the agent
        updates e function variables (list of e function inputs excluding the agent)
        updates e function argmax giving best response conditional on e function variables actions
        '''
        E = []
        for i in range(len(scope)):
            if scope[i] in self.agents_to_eliminate:
                if E == []:
                    if q_function_index[i] == 0:
                        E = self.q_functions[edge_id[i]]
                        variables = np.array([scope[i]])
                    else:
                        E = list(map(list, zip(*self.q_functions[edge_id[i]])))
                        variables = np.array([scope[i]])
                else:
                    old_E = E
                    if q_function_index[i] == 0:
                        new_E = self.q_functions[edge_id[i]]
                    else: 
                        new_E = list(map(list, zip(*self.q_functions[edge_id[i]])))
                    new_shape = np.shape(new_E)
                    old_shape = np.shape(old_E)
                    index = -1
                    for j in range(len(variables)):
                        if scope[i] == variables[j]:
                            index = j
                            break 
                    new_E = np.asarray(new_E)
                    old_E = np.asarray(old_E)
                    if index == -1:
                        variables = np.append(variables,scope[i])
                        new_shape = np.append(old_shape,new_shape[1:])
                        if len(old_shape) > 1:
                            new_E = np.broadcast_to(new_E[:,self.printnewaxis(len(old_shape)-1),:], new_shape)
                        # add one new axis to old E as we know new E is a Q function
                        E = np.broadcast_to(old_E[:,np.newaxis],new_shape) + new_E
                    else:
                        new_E = np.broadcast_to(new_E, old_shape)
                        new_E = np.swapaxes(new_E, 1, index+1)
                        E = old_E + new_E

            # if this local Q function has been replaced by an e:
            else:
                if E == [] and self.functions_e_used[scope[i]] == False:
                    E = self.functions_e[scope[i]]
                    for j in range(len(self.functions_e_variables[scope[i]])):
                        if agent == self.functions_e_variables[scope[i]][j]:
                            index = j
                            break 
                    E = np.swapaxes(E, 0, index)
                    variables = self.functions_e_variables[scope[i]]
                    variables = self.check_variables(variables, agent)
                    self.functions_e_used[scope[i]] = True 

                elif self.functions_e_used[scope[i]] == False:
                    old_E = E
                    new_E = self.functions_e[scope[i]]
                    for j in range(len(self.functions_e_variables[scope[i]])):
                        if agent == self.functions_e_variables[scope[i]][j]:
                            index = j
                            break 
                    new_E = np.swapaxes(new_E, 0, index)
                    self.functions_e_used[scope[i]] = True 
                    if old_E != []:
                        old_shape = np.shape(old_E)
                        new_shape = np.shape(new_E)
                        index = -1
                        for j in range(len(variables)):
                            if scope[i] == variables[j]:
                                index = j
                                break 
                        new_E = np.asarray(new_E)
                        old_E = np.asarray(old_E)
                        if index == -1:
                            variables = np.append(variables,scope[i])
                            variables = self.check_variables(variables, agent)
                            next_shape = np.append(old_shape,new_shape[1:])
                            if len(old_shape) > 1:
                                new_E = np.broadcast_to(new_E[:,self.printnewaxis(len(old_shape)-1),:], next_shape)
                            if len(new_shape) > 1:
                                old_E = np.broadcast_to(old_E[:,self.printnewaxis(len(new_shape)-1)], next_shape)
                            E = old_E + new_E
                        else:
                            if old_shape >= new_shape:
                                new_E = np.broadcast_to(new_E, old_shape)
                                new_E = np.swapaxes(new_E, 1, index+1)
                                E = old_E + new_E
                            else:
                                old_E = np.broadcast_to(old_E, new_shape)
                                old_E = np.swapaxes(old_E, 1, index+1)
                                E = old_E + new_E

        variables = self.check_variables(variables, agent)

        if len(np.shape(E)) == 1:
            best_actions = np.argwhere(E == np.amax(E, axis=0))
            self.functions_e[agent] = int(random.choice(best_actions))
            self.functions_e_variables[agent] = []
        else:     
            self.functions_e[agent] = np.amax(E, axis=0)
            noise = self.make_noise_to_reduce_bias(np.shape(E))
            self.functions_e_argmax[agent] = np.argmax(E+noise, axis=0)
            self.functions_e_variables[agent] = variables

    def check_variables(self, variables, agent):
        '''
        do this when adding e_function variables to the current agent's variables 
        removes multiple values and the agent himself from the neighbours list
        and checks all agents are uneliminated 
        '''
        while True:
            for i in range(len(variables)):
                if variables[i] not in self.agents_to_eliminate:
                    variables = np.append(variables, self.functions_e_variables[variables[i]])
                    variables = np.delete(variables,i)
            variables = np.unique(variables)
            variables = variables[variables != agent]
            if(all(x in self.agents_to_eliminate for x in variables)):
                return variables 

    def make_noise_to_reduce_bias(self, E_shape):
        '''
        This estimates the solution to finding a random
        argmax value rather than the first occurrence
        NEEDS TO BE CHANGED TO REMOVE BIAS IN LARGER ACTION SPACES
        '''
        noise = np.zeros(E_shape)
        if len(E_shape) == 1:
            noise = np.array(range(E_shape[-1]))* 1e-15 * np.random.randint(2, size=E_shape[-1])
        elif len(E_shape) == 2:
            for i in range(E_shape[0]):
                for j in range(E_shape[1]):
                    noise[i][j] = (i+j)* 1e-15 *np.random.randint(2)
        elif len(E_shape) == 3:
            for i in range(E_shape[0]):
                for j in range(E_shape[1]):
                    for k in range(E_shape[2]):
                        noise[i][j][k] = (i+j)* 1e-15 *np.random.randint(2)
        elif len(E_shape) == 4:
            for i in range(E_shape[0]):
                for j in range(E_shape[1]):
                    for k in range(E_shape[2]):
                        for l in range(E_shape[3]):
                            noise[i][j][k][l] = (i+j)* 1e-15*np.random.randint(2)
        return noise 

    def printnewaxis(self,i):
        '''
        used in new_function to reshape E
        '''
        if i ==1:
            return np.newaxis
        else:
            return self.printnewaxis(i-1),np.newaxis

    def chooseaction(self, length, acting_agent, other_agents):
        '''
        Returns optimal action given opponent strategies
        '''
        i = length
        if length == 0:  
            return self.functions_e_argmax[acting_agent]
        else:
            i -= 1
            return self.chooseaction(length-1,acting_agent,other_agents)[self.opt_action[other_agents[i]]]
        

