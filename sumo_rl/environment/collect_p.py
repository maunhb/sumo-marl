#### To Do: include this stuff within the environment,
#  the step function updates the moving average, and prints data

import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np 
import traci 

        
class CollectP2():

    def __init__(self, ts_ids, phases, filename):
        self.ts_ids = ts_ids
        self.horiz_phases = len(phases)//2
        self.filename = filename
        self.time = [0]
        if len(self.ts_ids) == 1:
                    self.total_red_time = [0]
                    self.total_green_time = [0]
        else:
            self.total_red_time = np.zeros((1,len(self.ts_ids)))
            self.total_green_time = np.zeros((1,len(self.ts_ids)))
            self.tempgreen = np.zeros(len(self.ts_ids))
            self.tempred = np.zeros(len(self.ts_ids))
            

    def _add_p_data(self, time):
        self.time.append(time)
        if len(self.ts_ids) == 1:
            currentPhase = traci.trafficlight.getPhase(self.ts_ids[0])
            # this requires phases to be set out as the first half vertical, 
            # second half horizontal 
            if currentPhase < self.horiz_phases :
                self.total_red_time.append(self.total_red_time[-1] + 5)
                self.total_green_time.append(self.total_green_time[-1])
            else:
                self.total_red_time.append(self.total_red_time[-1])
                self.total_green_time.append(self.total_green_time[-1] + 5)
        else:
            tsindex = 0
            for ts in self.ts_ids:
                currentPhase = traci.trafficlight.getPhase(ts)
                # this requires phases to be set out as: first half vertical,
                #  second half horizontal 
                if currentPhase < self.horiz_phases :
                    self.tempred[tsindex] = self.total_red_time[-1][tsindex]+5
                    self.tempgreen[tsindex]= self.total_green_time[-1][tsindex]
                else:
                    self.tempred[tsindex] = self.total_red_time[-1][tsindex] 
                    self.tempgreen[tsindex]=self.total_green_time[-1][tsindex]+5
                tsindex += 1

            self.total_red_time = np.vstack((self.total_red_time,
                                             self.tempred))
            self.total_green_time = np.vstack((self.total_green_time,
                                               self.tempgreen))

    def _write_p_data_file(self):
        if len(self.ts_ids)==1:
            f = open(str(self.filename)+'pdata.csv', 'w')
            f.write("time;total_red_time;total_green_time\n")
            for i in range(1,len(self.total_red_time)):
                f.write("{};{};{}\n".format(self.time[i],
                                            self.total_red_time[i], 
                                            self.total_green_time[i]))
            f.close()
        else:   
            tsindex = 0 
            for ts in self.ts_ids: 
                f = open(str(self.filename)+'pdata'+str(ts)+".csv", 'w')
                f.write("time;total_red_time;total_green_time\n")
                # red time, green time
                for i in range(1,len(self.time)):
                    f.write("{};{};{}\n".format(self.time[i],
                                            self.total_red_time[i][tsindex], 
                                            self.total_green_time[i][tsindex]))
                tsindex += 1
                f.close()

    





        





 



