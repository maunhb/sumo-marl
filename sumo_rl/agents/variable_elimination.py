import numpy as np 
import random

class VariableElimination():
   
    def __init__(self, local_q_table, ordering, edges):
        '''
        Elements 
        ----------------------------------------------------------------------
        q_fns:         (dict) of local q tables, given state s
        function_e:    (dict int matrices) agents e functions
        fn_e_argmax:   (dict int matrices) best response tables - which
                        action to take dependent on opponents
        fn_e_vars:     (dict int array) which variables the e function
                        include (other than their own action)
        fn_e_used:     (dict bool) has e function been replaced by another 
                        e function 
        elim_ordering: (list int) order in which agents should be eliminated - 
                        opposite of action selection
        coord_graph:   (list int) list of vertices connected by edges 
                        eg [v_1(e_1), v_2(e_1), v_1(e_2), v_2(e_2)]
        opt_action:    (dict int) outputs of VE - the optimal action to take 
                        for each player 
        '''
        self.agents_to_eliminate = ordering
        self.q_fns = local_q_table 
        self.fn_e = {agent: None for agent in ordering} 
        self.fn_e_argmax = {agent: None for agent in ordering} 
        self.fn_e_vars = {agent: None for agent in ordering} 
        self.fn_e_used = {agent: False for agent in ordering}  
        self.elim_ordering = ordering
        self.coord_graph = edges 
        self.opt_action = {} 

    def VariableElimination(self):
        ''' 
        Eliminates agents by replacing their Q function with e functions.
        Then agents choose their actions in the opposite order.

        Returns:
        (dict) optimal action
        '''
        i = 0
        # eliminate agents and update functions
        while i < len(self.elim_ordering):
            elim = self.elim_ordering[i] 
            agents, edges, q_input = self.scope(elim)
            self.new_function(elim, agents, edges, q_input)
            self.agents_to_eliminate = np.delete(self.agents_to_eliminate,0)
            i += 1
        #find optimal joint actions
        while i > 0:
            acting_agent = self.elim_ordering[i-1]
            arg = 0; other_agents = []
            for x in self.fn_e_vars[acting_agent]:
                if x in self.elim_ordering[i::]:
                    other_agents = np.append(other_agents,x)
                    arg +=1 
            if arg == 0:
                action = self.fn_e[acting_agent]
            else:
                action = self.choose_a(arg, acting_agent, other_agents)
            self.opt_action.update({acting_agent: action})
            i -= 1

        # print('Optimal actions:')
        # print(self.opt_action)
        return self.opt_action 

    def scope(self, agent):
        ''' 
        Input:
        agent - id of agent

        Returns:
        scope - (list int) of neighbouring agents
        edge_indexes - (list int) of agent's coorindation graph edges
        q_input - (list bool) of whether the agent in the first or second node
                in the edge vector
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
        Updates:
        - e function of the agent
        - e function variables (list) e function inputs excluding agent
        - e function argmax giving BR conditional on e func. variables actions
        '''
        E = []
        for i in range(len(scope)):
            if scope[i] in self.agents_to_eliminate:
                if E == []:
                    if q_function_index[i] == 0:
                        E = self.q_fns[edge_id[i]]
                        variables = np.array([scope[i]])
                    else:
                        E = list(map(list, zip(*self.q_fns[edge_id[i]])))
                        variables = np.array([scope[i]])
                else:
                    old_E = E
                    if q_function_index[i] == 0:
                        new_E = self.q_fns[edge_id[i]]
                    else: 
                        new_E = list(map(list, zip(*self.q_fns[edge_id[i]])))
                    new_shape = np.shape(new_E)
                    old_shape = np.shape(old_E)
                    index = self.index(variables, scope[i])
                    new_E = np.asarray(new_E)
                    old_E = np.asarray(old_E)
                    if index == -1:
                        variables = np.append(variables,scope[i])
                        new_shape = np.append(old_shape,new_shape[1:])
                        if len(old_shape) > 1:
                            new_E = np.broadcast_to(
                                new_E[:,self.new_ax(len(old_shape)-1),:],
                                new_shape)
                        # add one new axis to old_E 
                        # as we know new_E is a Q function
                        E = np.broadcast_to(old_E[:,np.newaxis],
                                            new_shape) + new_E
                    else:
                        new_E = np.broadcast_to(new_E, old_shape)
                        new_E = np.swapaxes(new_E, 1, index+1)
                        E = old_E + new_E

            # if this local Q function has been replaced by an e:
            else:
                if E == [] and self.fn_e_used[scope[i]] == False:
                    E = self.fn_e[scope[i]]
                    E = np.swapaxes(E, 0, 
                                    self.index(self.fn_e_vars[scope[i]],
                                               agent))
                    variables = self.fn_e_vars[scope[i]]
                    variables = self.check_variables(variables, agent)
                    self.fn_e_used[scope[i]] = True 

                elif self.fn_e_used[scope[i]] == False:
                    old_E = E
                    new_E = self.fn_e[scope[i]]
                    new_E = np.swapaxes(new_E, 0, 
                                        self.index(self.fn_e_vars[scope[i]],
                                                   agent))
                    self.fn_e_used[scope[i]] = True 
                    if old_E != []:
                        old_shape = np.shape(old_E)
                        new_shape = np.shape(new_E)
                        index = self.index(variables, scope[i])
                        new_E = np.asarray(new_E)
                        old_E = np.asarray(old_E)
                        if index == -1:
                            variables = np.append(variables,scope[i])
                            variables = self.check_variables(variables, agent)
                            next_shape = old_shape + new_shape[1:]
                            if len(old_shape) > 1:
                                if len(new_shape) > 1:
                                    new_E = np.broadcast_to(
                                        new_E[:,self.new_ax(len(old_shape)-1),:], 
                                        next_shape)
                                else:
                                    new_E = np.broadcast_to(
                                        new_E[:,self.new_ax(len(old_shape)-1)], 
                                        next_shape)
                            if len(new_shape) > 1:
                                old_E = np.broadcast_to(
                                        old_E[:,self.new_ax(len(new_shape)-1)],
                                        next_shape)
                            E = old_E + new_E
                        else:
                            if old_shape >= new_shape:
                                new_E = np.broadcast_to(new_E, old_shape)
                                new_E = np.swapaxes(new_E, 1, index+1)
                            else:
                                old_E = np.broadcast_to(old_E, new_shape)
                                old_E = np.swapaxes(old_E, 1, index+1)
                            E = old_E + new_E

        variables = self.check_variables(variables, agent)
        E_size = np.shape(E)
        if len(E_size) == 1:
            best_actions = np.argwhere(E == np.amax(E, axis=0))
            self.fn_e[agent] = int(random.choice(best_actions))
            self.fn_e_vars[agent] = []
        else:     
            self.fn_e[agent] = np.amax(E, axis=0)
            self.argmax(E, E_size, agent)
            self.fn_e_vars[agent] = variables

    def check_variables(self, variables, agent):
        '''
        Removes multiple values and the agent herself from the neighbours list
        and checks all agents are uneliminated.
        '''
        while True:
            for i in range(len(variables)):
                if variables[i] not in self.agents_to_eliminate:
                    variables = np.append(variables, 
                                          self.fn_e_vars[variables[i]])
                    variables = np.delete(variables, i)
            variables = np.unique(variables)
            variables = variables[variables != agent]
            if(all(x in self.agents_to_eliminate for x in variables)):
                return variables 

    def new_ax(self,i):
        '''
        Prints np.newaxis (used in new_function to reshape E).
        '''
        if i ==1:
            return np.newaxis
        else:
            return self.new_ax(i-1),np.newaxis

    def choose_a(self, length, acting_agent, other_agents):
        '''
        Returns optimal action given opponent strategies
        '''
        i = length
        if length == 0:  
            return self.fn_e_argmax[acting_agent]
        else:
            i -= 1
            return self.choose_a(length-1,
                                 acting_agent,
                                 other_agents
                                 )[self.opt_action[other_agents[i]]]

    def index(self,vector,element):
        '''
        Inputs: 
        - vector 
        - element
        Returns:
        - the first index of the vector where the element exists 
        - -1 if the element doesn't belong in the vector
        '''
        for j in range(len(vector)):
            if element == vector[j]:
                return j
        return -1

    def argmax(self, E, E_size, agent):
        '''
        Randomly selects the argmax of E with respect to the first index. 
        '''
        argmax = np.zeros(E_size[1:])
        argmaxshape = np.shape(argmax)
        indices = self.indices(argmaxshape)
        for ind in indices:
            inputstring = "E[:,{}]".format(str(ind)[1:-1])
            E = np.asarray(E)
            input = eval(inputstring)
            best_actions = np.argwhere(input == np.amax(input))
            argmax[tuple(ind)] = int(random.choice(best_actions))
        argmax = argmax.astype(int)
        self.fn_e_argmax[agent] = argmax

    def indices(self, shape):
        '''
        Generates a list of all possibe integer vectors within a space.
        '''
        indices = [[]]
        for dim in shape:
            new_indices = []
            for ind in indices:
                for n in range(dim):
                    new_indices.append(ind+[n])
            indices = new_indices
        return indices 


# ----------------------------------------------------------------------

def MakeVertexList(coord_graph):
    '''
     Removes any duplicates from coord graph.
     '''
    coord_edges = []
    vertex_list = list(coord_graph.keys())
    for vertex in coord_graph:
        for i in range(0,len(coord_graph[vertex])):
            if coord_graph[vertex][i] in vertex_list:
                coord_edges = np.append(coord_edges, 
                                            int(vertex))
                coord_edges = np.append(coord_edges, 
                                            int(coord_graph[vertex][i]))
        vertex_list.remove(vertex)

    return vertex_list, coord_edges