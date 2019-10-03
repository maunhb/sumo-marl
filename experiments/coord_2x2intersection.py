import argparse
import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl.environment.coord_env import SumoEnvironment # also coord_env
from sumo_rl.agents.coord_agent import CoordAgent
from sumo_rl.exploration.coord_epsilon_greedy import EpsilonGreedy
from sumo_rl.agents.variable_elimination import VariableElimination

# THE AIM IS TO HAVE A SEPARATE FUNCTION THAT MAKES A COORD GRAPH WITH VARIOUS PROPERTIES
coord_graph = {
    1:[2,6],
    2:[1,5],
    5:[2,6],
    6:[1,5]
}
# remove any duplicates from coord graph
# coord edges represent the edges as a vector where 
coord_edges = []
vertex_list = list(coord_graph.keys())
for vertex in coord_graph:
    for i in range(0,len(coord_graph[vertex])):
        if coord_graph[vertex][i] in vertex_list:
            coord_edges = np.append(coord_edges, vertex)
            coord_edges = np.append(coord_edges, coord_graph[vertex][i])
    vertex_list.remove(vertex)
# WE ARE WORKING ON AN ALGORITHM TO CHOOSE OPTIMAL ORDERING BASED ON THE NETWORK TYPE
elim_ordering = [1,2,5,6] # implement max-plus algorithm
action_ordering = elim_ordering[::-1]


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='scenarios/my2x2grid/2x2.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=120000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait1', required=False, help="Reward function: [-r av_q] for average queue reward, [-r q] for queue reward, [-r wait1] for waiting time reward,  [-r wait2] for waiting time reward 2,  [-r wait3] for waiting time reward 3.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True, help="Choose a tripinfo output file name (.xml).\n")

    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/my-2x2-grid/coord_q_{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay, args.reward)

    env = SumoEnvironment(net_file='scenarios/my2x2grid/2x2.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          trip_file=args.tripfile,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          min_green=args.min_green,
                          max_green=args.max_green,
                          max_depart_delay=0,
                          time_to_load_vehicles=120,
                          phases=[
                            traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),  
                            traci.trafficlight.Phase(3, "yyrrrryyrrrr"),
                            traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),   
                            traci.trafficlight.Phase(3, "rryrrrrryrrr"),
                            traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),   
                            traci.trafficlight.Phase(3, "rrryyrrrryyr"),
                            traci.trafficlight.Phase(32, "rrrrrGrrrrrG"), 
                            traci.trafficlight.Phase(3, "rrrrryrrrrry")
                            ])

    if args.reward == 'av_q':
        env._compute_rewards = env._queue_average_reward
    elif args.reward == 'q':
        env._compute_rewards = env._queue_reward
    elif args.reward == 'wait1':
        env._compute_rewards = env._waiting_time_reward
    elif args.reward == 'wait2':
        env._compute_rewards = env._waiting_time_reward2
    elif args.reward == 'wait3':
        env._compute_rewards = env._waiting_time_reward3

    for run in range(1, args.runs+1):
        initial_states = env.reset()

        coord_agents = {edge: CoordAgent(joint_starting_state=[env.encode(initial_states['{}'.format(int(coord_edges[edge]))]),env.encode(initial_states['{}'.format(int(coord_edges[edge+1]))])],
                                 joint_state_space=[env.observation_space, env.observation_space],
                                 joint_action_space=[env.action_space,env.action_space],
                                 alpha=args.alpha,
                                 gamma=args.gamma) for edge in range(0,len(coord_edges),2)} 
        strategy = EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)

        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:

                # define q functions for current state for VE algo
                q_functions = {edge: coord_agents[edge].q_table['{}'.format(coord_agents[edge].state)] for edge in coord_agents }
 
                # make VE algo object 
                ve = VariableElimination(q_functions, elim_ordering, coord_edges)
                # do VE algo to find optimal action profile
                opt_actions = ve.VariableElimination()
                # use epsilon greedy strategy to choose actions 
                actions = strategy.choose(opt_actions, env.action_space)
                # find new state and rewards by implementing actions
                s, r, done, _ = env.step(action=actions)
                
                if args.v:
                    print('s=', env.radix_decode(coord_agents['t'].state), 'a=', actions['t'], 's\'=', env.radix_encode(s['t']), 'r=', r['t'])
                # update q tables 
                # rewards are the sum of the rewards of the connected traffic lights
                i = 0
                for edge_id in coord_agents.keys():
                    coord_agents[edge_id].learn(new_state=[env.encode(s['{}'.format(int(coord_edges[i]))]),env.encode(s['{}'.format(int(coord_edges[i+1]))])], reward=r['{}'.format(int(coord_edges[i]))]+r['{}'.format(int(coord_edges[i+1]))])
                    i += 2

        env.save_csv(out_csv, run)
        env.close()




