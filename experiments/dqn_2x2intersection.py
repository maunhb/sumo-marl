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
import traci 

from stable_baselines.deepq import DQN, MlpPolicy

if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""DQN 2x2 grid""")
    prs.add_argument("-route", dest="route", type=str, default='scenarios/my2x2grid/2x2.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait1', required=False, help="Reward function: [-r av_q] for average queue reward, [-r q] for queue reward, [-r wait1] for waiting time reward,  [-r wait2] for waiting time reward 2,  [-r wait3] for waiting time reward 3.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True, help="Choose a tripinfo output file name (.xml).\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/my-2x2-grid/dqn_{}_reward{}'.format(experiment_time, args.reward)

    env = SumoEnvironment(net_file='scenarios/my2x2grid/2x2.net.xml',
                          route_file=args.route,
                          out_csv_name=out_csv,
                          trip_file=args.tripfile,
                          use_gui=args.gui,
                          num_seconds=args.seconds,
                          #single_agent=True, This only changes the first traffic light
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

    model = DQN(
        env=env,
        policy=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )
    model.learn(total_timesteps=100000)




