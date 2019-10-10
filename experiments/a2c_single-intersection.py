import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

import argparse
import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
from gym import spaces
import numpy as np
from sumo_rl.environment.env import SumoEnvironment
import traci


if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""A2C Single-Intersection""")
    prs.add_argument("-tripfile", dest="tripfile", type=str, required=True, help="Choose a tripinfo output file name (.xml).\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-r", dest="reward", type=str, default='wait1', required=False, help="Reward function: [-r av_q] for average queue reward, [-r q] for queue reward, [-r wait1] for waiting time reward,  [-r wait2] for waiting time reward 2,  [-r wait3] for waiting time reward 3.\n")
    args = prs.parse_args()

    n_cpu = 1
    env = SubprocVecEnv([lambda: SumoEnvironment(net_file='scenarios/mysingleintersection/single-intersection.net.xml',
                                    route_file='scenarios/mysingleintersection/single-intersection.rou.xml',
                                    out_csv_name='outputs/my-single-intersection/dqn-stable-mlp-bs',
                                    trip_file=args.tripfile,
                                    single_agent=True,
                                    use_gui=False,
                                    num_seconds=args.seconds,
                                    time_to_load_vehicles=120,
                                    max_depart_delay=0,
                                    phases=[
                                        traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(3, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(3, "rryrrrrryrrr"),
                                        traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(3, "rrryyrrrryyr"),
                                        traci.trafficlight.Phase(32, "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(3, "rrrrryrrrrry")
                                        ]) for i in range(n_cpu)])

    model = A2C(
        env=env,
        verbose=1,
        policy=MlpPolicy,
        learning_rate=1e-3,
        lr_schedule='constant'
    )
    model.learn(total_timesteps=100000)
