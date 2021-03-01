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
from sumo_rl.environment.coord_env import SumoEnvironment 
from sumo_rl.environment.collect_p import CollectP2

from sumo_rl.agents.coord_agent import CoordAgent
from sumo_rl.agents.variable_elimination import VariableElimination
from sumo_rl.agents.variable_elimination import MakeVertexList
from sumo_rl.exploration.coord_epsilon_greedy import EpsilonGreedy

coord_graph = {
    1:[2,5],
    2:[1,6],
    5:[1,6],
    6:[2,5]
}

vertex_list, edges = MakeVertexList(coord_graph)

elim_ordering = [1,2,6,5] 

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=
                                  argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Coord Q-Learning 2x2 Grid""")
    prs.add_argument("-route", dest="route", type=str, 
                     default='scenarios/my2x2grid/2x2.rou.xml', 
                                          help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, 
                                required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, 
                                required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, 
                                            required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, 
                                        required=False, help="Min epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, 
                                      required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10,
                                     required=False, help="Min green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, 
                                     required=False, help="Max green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, 
                                     help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, 
                              help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, 
                             required=False, help="No. simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait', 
                required=False, help="Reward function.\n")
    prs.add_argument("-v", action="store_true", default=False, 
                                             help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, 
                                                     help="Number of runs.\n")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True,
                          help="Choose a tripinfo output file name (.xml).\n")
    prs.add_argument("-summaryfile", dest="summaryfile",
                  default='outputs/summarycoord2x2.xml', type=str, 
                  required=False, help="Choose a summary file name (.xml).\n")

    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/my-2x2-grid/coord_q_{}_alpha{}_gamma{}_eps{}_decay{}_'\
                'reward{}'.format(experiment_time, 
                                  args.alpha, 
                                  args.gamma, 
                                  args.epsilon, 
                                  args.decay, 
                                  args.reward)

    env = SumoEnvironment(net_file='scenarios/my2x2grid/2x2.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          trip_file=args.tripfile,
                          summary_file=args.summaryfile,
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


    if args.reward == 'wait':
        env._compute_rewards = env._total_wait
    elif args.reward == 'wait2':
        env._compute_rewards = env._total_wait_2
    else:
        raise "Error: Invalid reward."

    for run in range(1, args.runs+1):
        s = env.reset(run)

        agents = {edge: CoordAgent(
                            joint_starting_state=
                               [env.encode(s['{}'.format(int(edges[edge]))]),
                               env.encode(s['{}'.format(int(edges[edge+1]))])],
                            joint_state_space=[env.observation_space, 
                                               env.observation_space],
                            joint_action_space=[env.action_space,
                                                env.action_space],
                            alpha=args.alpha,
                            gamma=args.gamma) for edge in range(0,len(edges),2)
                 } 
                                 
        strategy = EpsilonGreedy(initial_epsilon=args.epsilon, 
                                 min_epsilon=args.min_epsilon, 
                                 decay=args.decay)

        collect_ts_data =  CollectP2(ts_ids=env.ts_ids, 
                                     phases=env.phases, 
                                     filename='outputs/my-3x3-grid/'\
                                              'coordq{}'.format(run))  

        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:

                # get q functions for current state for VE algo
                q_functions = {edge: agents[edge].q_table['{}'\
                                        .format(agents[edge].state)] 
                                           for edge in agents.keys() 
                               }
     
                # make VE algo object 
                ve = VariableElimination(q_functions, elim_ordering, edges)
                # do VE algo to find optimal action profile
                opt_actions = ve.VariableElimination()

                # use epsilon greedy strategy to choose actions 
                action_profile = strategy.choose(opt_actions, env.action_space)
                # find new state and rewards by implementing actions
                s, r, done, _ = env.step(action=action_profile)
                collect_ts_data._add_p_data(time=
                                        traci.simulation.getCurrentTime()/1000)
                
                if args.v:
                    print('s=', env.radix_decode(agents['t'].state), 
                                                 'a=', 
                                                  action_profile['t'], 
                                                  's\'=', 
                                                  env.radix_encode(s['t']), 
                                                  'r=', 
                                                  r['t'])
                # update q tables 
                i = 0
                for edge_id in agents.keys():
                    reward=r['{}'.format(edges[i])]+r['{}'.format(edges[i+1])]
                    agents[edge_id].learn(
                                        new_state=
                                        [env.encode(s['{}'.format(edges[i])]),
                                        env.encode(s['{}'.format(edges[i+1])])], 
                                        actions=[action_profile[edges[i]],
                                                   action_profile[edges[i+1]]],
                                        reward=reward)
                    i += 2 

        collect_ts_data._write_p_data_file()
        env.save_csv(out_csv, run)
        env.close()





