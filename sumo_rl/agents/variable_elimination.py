import numpy as np 
import random
# Variable Elimination:
# run variable elimination to find e functions
# then it does action selection to determine optimal action profile
class VariableElimination():
   
    def __init__(self, local_q_table, ordering, edges):
        self.agents_to_eliminate = ordering
        self.q_functions = local_q_table # a dictionary of local q tables, given state s
        self.functions_e = {agent: None for agent in ordering} # dict of agents e functions
        self.functions_e_variables = {agent: None for agent in ordering} # compiles which variables the e function include (other than their own action)
        self.functions_e_used = {agent: False for agent in ordering} # has the e function been replaced by another e function 
        self.elim_ordering = ordering
        self.coord_graph = edges 
        self.opt_action = {} 

    def VariableElimination(self):
        i = 0
        # eliminate agents and update functions
        while i < len(self.elim_ordering):
            # find next agent to eliminate
            agent2elim = self.elim_ordering[i] 
            # find their scope
            connected_agents, connected_edge_indexes, agent_q_input = self.find_scope(agent2elim)
            # update their e function
            self.new_function(agent2elim, connected_agents, connected_edge_indexes, agent_q_input)
            # remove them from agents to eliminate
            self.agents_to_eliminate = np.delete(self.agents_to_eliminate,0)
            i += 1
        #find optimal joint actions
        while i > 0:
            # select actions in reverse order
            acting_agent = self.elim_ordering[i-1]
            # find the other actions of those that affect your choice
            arg = 0; other_agents = []
            for x in self.functions_e_variables[acting_agent]:
                if x in self.elim_ordering[i::]:
                    other_agents = np.append(other_agents,x)
                    arg +=1 
            # find the maximum given the action profile
            if arg == 0:
                action = self.functions_e[acting_agent]
            elif arg == 1:
                action = self.functions_e[acting_agent][self.opt_action[other_agents[0]]]
            elif arg == 2:
                action = self.functions_e[acting_agent][self.opt_action[other_agents[0]]][self.opt_action[other_agents[1]]]
            elif arg == 3:
                action = self.functions_e[acting_agent][self.opt_action[other_agents[0]]][self.opt_action[other_agents[1]]][self.opt_action[other_agents[2]]]
            elif arg == 4:
                action = self.functions_e[acting_agent][self.opt_action[other_agents[0]]][self.opt_action[other_agents[1]]][self.opt_action[other_agents[2]]][self.opt_action[other_agents[3]]]
            # update action dictionary with optimal action

            self.opt_action.update({acting_agent: action})
            i -= 1
        # return optimal actions
        return self.opt_action 

    def find_scope(self, agent):
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
        # at the moment it only works if you have a maximum of 3 neighbours each
        # but can write a few more lines of code to make it work for max degree 4
        E = []
        for i in range(len(edge_id)):
            # if this Q function has not been replaced by an e:
            if scope[i] in self.agents_to_eliminate:
                # is it the first element of the E matrix?
                if E == []:
                    # make a new E
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
                    if len(old_shape) == 1:
                        E = [[old_E[i]+new_E[i][k] for k in range(old_shape[1])] for i in range(old_shape[0])]
                        # update e function variable
                        variables = np.append(variables,scope[i])
                    elif len(old_shape) == 2:
                        if scope[i] in variables:  
                            E = [[old_E[i][k]+new_E[i][k] for k in range(old_shape[1])] for i in range(old_shape[0])]
                        else:
                            E = [[[old_E[i][j]+new_E[i][k] for k in range(new_shape[1])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                            # update e function variable
                            variables = np.append(variables,scope[i])
                    elif len(old_shape) == 3:
                        if scope[i] == variables[0]:
                            E = [[[old_E[i][j][k]+new_E[i][j] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                        elif scope[i] == variables[1]:
                            E = [[[old_E[i][j][k]+new_E[i][k] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                        else:
                            E = [[[[old_E[i][j][k]+new_E[i][l] for l in range(new_shape[1])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                            # update e function variables
                            variables = np.append(variables,scope[i])
                    elif len(old_shape) == 4:
                        if scope[i] == variables[0]:
                            E = [[[[old_E[i][j][k][l]+new_E[i][j] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                        elif scope[i] == variables[1]:
                            E = [[[[old_E[i][j][k][l]+new_E[i][k] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                        elif scope[i] == variables[2]:
                            E = [[[[old_E[i][j][k][l]+new_E[i][l] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                        else:
                            E = [[[[[old_E[i][j][k][l]+new_E[i][m] for m in range(new_shape[1])] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                            # update e function variables
                            variables = np.append(variables,scope[i])
                    else:
                       return print('Error: max degree in coordination graph larger than 4.')
            # if this local Q function has been replaced by an e:
            else:
                # is it the first element of the E matrix?
                if E == []:
                    if self.functions_e_used[scope[i]] == False:
                        E = self.functions_e[scope[i]]
                        variables = self.functions_e_variables[scope[i]]
                        self.functions_e_used[scope[i]] = True 
                        variables = self.check_variables(variables, agent)
                else:
                    if self.functions_e_used[scope[i]] == False:
                        old_E = E
                        new_E = self.functions_e[scope[i]]
                        self.functions_e_used[scope[i]] = True 
                        # check sizes of old_E and new_E
                        if new_E != []:
                            old_shape = np.shape(old_E)
                            new_shape = np.shape(new_E)
                            if len(new_shape) > 4:
                                return print('Error: maximum coordination graph degree higher than 4.')
                            if len(old_shape) == 1:
                                if len(new_shape) == 1:
                                    E = [old_E[i]+new_E[i] for i in range(old_shape[0])]
                                elif len(new_shape) ==1:
                                    E = [[old_E[i]+new_E[i][k] for k in range(new_shape[1])] for i in range(old_shape[0])]
                                    variables = self.functions_e_variables[scope[i]]
                                    variables = self.check_variables(variables,agent)
                            elif len(old_shape) == 2:
                                if len(new_shape) == 1:
                                    E = [[old_E[i][k]+new_E[i] for k in range(old_shape[1])] for i in range(old_shape[0])]
                                elif len(new_shape) == 2:
                                    if scope[i] in variables: 
                                        E = [[old_E[i][k]+new_E[i][k] for k in range(old_shape[1])] for i in range(old_shape[0])]
                                    else:
                                        E = [[[old_E[i][j]+new_E[i][k] for k in range(new_shape[1])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                        # update e function variables
                                        variables = np.append(variables, self.functions_e_variables[scope[i]])
                                        variables = self.check_variables(variables, agent)
                            elif len(old_shape) == 3:
                                if len(new_shape) == 1:
                                    E = [[[old_E[i][j][k]+new_E[i] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                elif len(new_shape) == 2:
                                    if scope[i] == variables[0]:
                                        E = [[[old_E[i][j][k]+new_E[i][j] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                    elif scope[i] == variables[1]:
                                        E = [[[old_E[i][j][k]+new_E[i][k] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                    else:
                                        E = [[[[old_E[i][j][k]+new_E[i][l] for l in range(new_shape[1])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                        # update e function variables
                                        variables = np.append(variables, self.functions_e_variables[scope[i]])
                                        variables = self.check_variables(variables, agent)
                                elif len(new_shape) == 3:
                                    return print('Error: not implemented this case yet.')
                                else:
                                    return print('Error: not implemented this case yet.')
                            elif len(old_shape) == 4:
                                if len(new_shape) == 1:
                                    E = [[[[old_E[i][j][k][l]+new_E[i] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                elif len(new_shape) == 2:
                                    if scope[i] == variables[0]:
                                        E = [[[[old_E[i][j][k][l]+new_E[i][j] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                    elif scope[i] == variables[1]:
                                        E = [[[[old_E[i][j][k][l]+new_E[i][k] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                    elif scope[i] == variables[2]:
                                        E = [[[[old_E[i][j][k][l]+new_E[i][l] for l in range(old_shape[3])] for k in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                    else:
                                        E = [[[[[old_E[i][j][l][m]+new_E[i][k] for k in range(new_shape[1])] for m in range(old_shape[3])] for l in range(old_shape[2])] for j in range(old_shape[1])] for i in range(old_shape[0])]
                                        # update e function variables
                                        variables = np.append(variables, self.functions_e_variables[scope[i]])
                                        variables = self.check_variables(variables, agent)
                                elif len(new_shape) == 3:
                                    return print('Error: not implemented this case yet.')
                                else:
                                    return print('Error: not implemented this case yet.')
        # record the e function and its variables
        if len(np.shape(E)) == 1:
            best_actions = np.argwhere(E == np.amax(E))
            self.functions_e[agent] = int(random.choice(best_actions))
            self.functions_e_variables[agent] = []
        else:
            E_shape = np.shape(E)
            noise = self.make_noise_to_reduce_bias(E_shape)       
            self.functions_e[agent] = np.argmax(E+noise, axis=0)
            self.functions_e_variables[agent] = variables

    def check_variables(self, variables, agent):
        # need to do this when adding e_function variables to the current agent's variables 
        while True:
            #check what variables are included in the e function
            for i in range(len(variables)):
                if variables[i] not in self.agents_to_eliminate:
                    variables = np.append(variables, self.functions_e_variables[variables[i]])
                    variables = np.delete(variables,i)
            # remove any duplicates
            variables = np.unique(variables)
            # remove any variables that are the agent
            variables = variables[variables != agent]
            # check that all variables are uneliminated agents
            if(all(x in self.agents_to_eliminate for x in variables)):
                return variables 

    def make_noise_to_reduce_bias(self, E_shape):
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
        elif len(E_shape) == 3:
            for i in range(E_shape[0]):
                for j in range(E_shape[1]):
                    for k in range(E_shape[2]):
                        for l in range(E_shape[3]):
                            noise[i][j][k][l] = (i+j)* 1e-15*np.random.randint(2)
        return noise 

  
