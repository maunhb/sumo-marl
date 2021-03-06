import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
from gym import Env
import traci.constants as tc
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import pandas as pd

from sumo_rl.environment.traffic_signal import TrafficSignal

movingAverageLength = 50

class SumoEnvironment(MultiAgentEnv):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param trip_file: (str) name of .xml file to collect trip data 
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases 
     definition
    :param out_csv_name: (str) name of the .csv output with simulation results. 
     If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be
     inserted after max_depart_delay seconds
    :param time_to_load_vehicles: (int) Number of simulation seconds ran 
     before learning begins
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. 
     Else, it behaves like a MultiagentEnv (/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(self, net_file, route_file, trip_file, summary_file, phases, 
                 out_csv_name=None, use_gui=False, num_seconds=20000, 
                 max_depart_delay=100000, time_to_load_vehicles=0, 
                 delta_time=5, min_green=5, max_green=50, single_agent=False):

        self._net = net_file
        self._route = route_file
        self.trip_file = trip_file 
        self.summary_file = summary_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        # start only to retrieve information
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  

        self.single_agent = single_agent
        self.ts_ids = traci.trafficlight.getIDList()
        lanes = []; ts_num = 0
        for ts in self.ts_ids:
            lanes.append(len(set(traci.trafficlight.getControlledLanes(self.ts_ids[ts_num]))))
            ts_num += 1
        self.lanes_per_ts = max(lanes)
        self.traffic_signals = dict()
        self.phases = phases
         # Number of green phases == number of phases (green+yellow)/2
        self.num_green_phases = len(phases) // 2 
        self.vehicles = dict()
        self.last_measure = dict()
        self.last_reward = {i: 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        # number of simulation seconds ran in reset() before learning starts
        self.time_to_load_vehicles = time_to_load_vehicles  
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay # Max wait to insert vehicle
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = 2

        """
        Default observation space is a vector:
                R^(#greenPhases + 1 + 2 * #lanes)

        s = [current phase one-hot encoded, elapsedTime / maxGreenTime,
             density for each lane, 
             (((queue))) 
                     for each lane]

        You can change this by modifing self.observation_space and the 
        method _compute_observations()

        Action space is which green phase is going to be open for the next
        delta_time seconds
        """
        self.observation_space = spaces.Box(
                low=np.zeros(self.num_green_phases+1+self.lanes_per_ts), 
                high=np.ones(self.num_green_phases + 1 + self.lanes_per_ts))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),
            spaces.Discrete(self.max_green//self.delta_time),
            *(spaces.Discrete(10) for _ in range(self.lanes_per_ts)) 
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        ## collecting p data 
        # self.total_red_time = np.zeros((len(self.ts_ids),1))
        # self.total_green_time = np.zeros((len(self.ts_ids),1))

        self.radix_factors = [s.n for s in self.discrete_observation_space.spaces]
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        traci.close()
        
    def reset(self, run):
        if self.run != 0:
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000', 
                     '--tripinfo-output', str(self.trip_file)+str(run)+str('.xml'),
                     '--summary', self.summary_file,
                     '--random']
        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)

        for ts in self.ts_ids:
            print("ts", ts)
            self.traffic_signals[ts] = TrafficSignal(self, ts, 
                                                     self.delta_time, 
                                                     self.min_green, 
                                                     self.max_green,
                                                     self.phases)
            self.last_measure[ts] = 0.0

        self.vehicles = dict()

        ## reset p data 
        if len(self.ts_ids) == 1:
            self.total_red_time = [0]
            self.total_green_time = [0]
        else:
            self.total_red_time = np.zeros((len(self.ts_ids),1))
            self.total_green_time = np.zeros((len(self.ts_ids),1))
            self.tempgreen = np.zeros(len(self.ts_ids))
            self.tempred = np.zeros(len(self.ts_ids))


        # Load vehicles
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getCurrentTime()/1000 

    def step(self, action):
        # act
        self._apply_actions(action)

        for _ in range(self.yellow_time): 
            self._sumo_step()
        for ts in self.ts_ids:
            self.traffic_signals[ts].update_phase()
        for _ in range(self.delta_time - self.yellow_time):
            self._sumo_step()

        # observe new state and reward
        observation = self._compute_observations()
        reward = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        info = self._compute_step_info()
        self.metrics.append(info)
        self.last_reward = reward

        if self.single_agent:
            return observation[self.ts_ids[0]], reward[self.ts_ids[0]], done['__all__'], {}
        else:
            return observation, reward, done, {}

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and 
                        self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)

    def _compute_observations(self):
        """
        Return the current observation for each traffic signal
        """
        observations = {}
        for ts in self.ts_ids:
            phase_id = [1 
                        if self.traffic_signals[ts].phase//2 == i 
                        else 0 
                        for i in range(self.num_green_phases)] #one-hot encoded
            elapsed = self.traffic_signals[ts].time_on_phase / self.max_green
            density = self.traffic_signals[ts].get_lanes_density()
            #queue = self.traffic_signals[ts].get_lanes_queue()
            observations[ts] = phase_id + [elapsed] + density #+ queue
        return observations

    def _compute_rewards(self):
        return self._total_wait()

    ### MY REWARD FUNCTIONS 
    def _total_wait(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - sum(self.traffic_signals[ts].get_waiting_time())
        return rewards

    def _total_wait_2(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - sum(self.traffic_signals[ts].get_squared_waiting_time())
        return rewards

    def _normalised_mean_wait(self):
        rewards = {}
        for ts in self.ts_ids:
            wait = np.array(self.traffic_signals[ts].get_waiting_times())
            wait_times = wait[wait != 0] 
            num_cars = len(wait_times)

            if len(wait_times) > 0:
                rewards[ts] = - np.power(np.prod(wait_times),1./num_cars)
            else: 
                rewards[ts] = 0
        print(rewards)
        return rewards


    def _normalised_mean_wait2(self):
        rewards = {}
        for ts in self.ts_ids:

            wait = np.array(self.traffic_signals[ts].get_waiting_times())
            wait_times = wait[wait != 0]
            num_cars = len(wait_times)
            wait_times = np.square(wait_times)

            if len(wait_times) > 0:
                rewards[ts] = - np.power(np.prod(wait_times),1./num_cars)
            else: 
                rewards[ts] = 0
        print(rewards)
        return rewards

    def _max_wait_time(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - self.traffic_signals[ts].get_biggest_waiting_time()
        return rewards

    def _max_wait_time2(self):
        rewards = {}
        for ts in self.ts_ids:
            rewards[ts] = - (
                        self.traffic_signals[ts].get_biggest_waiting_time())**2
        return rewards


    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return { #--- this is where you output rl results into the csv file ---
            'step_time': self.sim_step,
            'reward': self.last_reward[self.ts_ids[0]],
            'total_stopped': sum(
                      [sum(self.traffic_signals[ts].get_stopped_vehicles_num())
                       for ts in self.ts_ids]),
            'total_wait_time':sum([self.last_measure[ts]for ts in self.ts_ids])
            #'total_wait_time': sum(
            #    [sum(self.traffic_signals[ts].get_waiting_time()) 
            #     for ts in self.ts_ids])
        }

    def _write_p_data_file(self):
        if len(self.ts_ids)==1:
            f = open('pdata.csv', 'w')
            for i in range(len(self.total_red_time)):
                print(self.total_green_time)
                print(self.total_red_time)
                print(i)
                f.write("{};{}\n".format(self.total_red_time[i], 
                                         self.total_green_time[i]))
            f.close()
        else:    
            for ts in self.ts_ids:
                f = open(str('pdata')+str(ts)+str(".csv"), 'w')
                # red time, green time
                for i in range(len(self.total_red_time)):
                    print(self.total_green_time)
                    print(ts)
                    print(i)
                    f.write("{};{}\n".format(self.total_red_time[ts][i],
                                             self.total_green_time[ts][i]))
                f.close()

    def close(self):
        #self._write_p_data_file()
        traci.close()

    def encode(self, state):
        phase = state[:self.num_green_phases].index(1)
        elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) 
                         for d in state[self.num_green_phases + 1:]]
        return self.radix_encode([phase, elapsed] + density_queue)

    def _discretize_density(self, density):
        if density < 0.1:
            return 0
        elif density < 0.2:
            return 1
        elif density < 0.3:
            return 2
        elif density < 0.4:
            return 3
        elif density < 0.5:
            return 4
        elif density < 0.6:
            return 5
        elif density < 0.7:
            return 6
        elif density < 0.8:
            return 7
        elif density < 0.9:
            return 8
        else:
            return 9

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green
        for i in range(self.max_green//self.delta_time):
            if elapsed <= self.delta_time + i*self.delta_time:
                return i
        return self.max_green//self.delta_time -1

    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name+'_run{}'.format(run)+'.csv', index=False)
