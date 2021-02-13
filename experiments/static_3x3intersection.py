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


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Static 3x3-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='scenarios/my3x3grid/3x3.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=32, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=True, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait', required=False, help="Reward function: [-r queue] for average queue reward or [-r wait] for waiting time reward.\n")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True, help="Choose a tripinfo output file name (.xml).\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    prs.add_argument("-summaryfile", dest="summaryfile", default='outputs/summarystatic3x3.xml', type=str, required=False, help="Choose a summary file name (.xml).\n")
    args = prs.parse_args()
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/my-single-intersection/static_{}'.format(experiment_time)

    env = SumoEnvironment(net_file='scenarios/my3x3grid/3x3_12links.net.xml',
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
    if args.reward == 'queue':
        env._compute_rewards = env._queue_average_reward
    else:
        env._compute_rewards = env._waiting_time_reward

    for run in range(1, args.runs+1):
        initial_states = env.reset()
        done = {'__all__': False}

        infos = []
        if args.fixed:
            for i in range(args.seconds):
                env._sumo_step()


        env.save_csv(out_csv, run)
        env.close()




