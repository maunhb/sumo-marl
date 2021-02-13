import argparse
import os
import sys
import pandas as pd
from datetime import datetime

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.agents.ql_agent import QLAgent
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

from sumo_rl.environment.collect_p import CollectP2


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
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait2', required=False, help="Reward function: [-r av_q] for average queue reward, [-r q] for queue reward, [-r wait1] for waiting time reward,  [-r wait2] for waiting time reward 2,  [-r wait3] for waiting time reward 3.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True, help="Choose a tripinfo output file name (.xml).\n")
    prs.add_argument("-summaryfile", dest="summaryfile", default='outputs/summaryq2x2.xml', type=str, required=False, help="Choose a summary file name (.xml).\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/my-2x2-grid/q_{}_alpha{}_gamma{}_eps{}_decay{}_reward{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay, args.reward)

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
    elif args.reward == 'stopped':
        env._compute_rewards = env._total_stopped
    elif args.reward == 'stopped2':
        env._compute_rewards = env._total_stopped_2
    elif args.reward == 'density':
        env._compute_rewards = env._total_density
    elif args.reward == 'density2':
        env._compute_rewards = env._total_density_2
    elif args.reward == 'stoppeddensity':
        env._compute_rewards = env._total_stopped_density
    elif args.reward == 'stoppeddensity2':
        env._compute_rewards = env._total_stopped_density_2

    for run in range(1, args.runs+1):
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts]),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=args.alpha,
                                 gamma=args.gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)) for ts in env.ts_ids}
        collect_ts_data =  CollectP2(ts_ids=env.ts_ids, phases=env.phases, filename='outputs/CollectingP/2x2') 
        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, _ = env.step(action=actions)
                collect_ts_data._add_p_data(time=traci.simulation.getCurrentTime()/1000)
                if args.v:
                    print('s=', env.radix_decode(ql_agents['t'].state), 'a=', actions['t'], 's\'=', env.radix_encode(s['t']), 'r=', r['t'])

                for agent_id in ql_agents.keys():
                    ql_agents[agent_id].learn(new_state=env.encode(s[agent_id]), reward=r[agent_id])

        collect_ts_data._write_p_data_file()
        env.save_csv(out_csv, run)
        env.close()
        




