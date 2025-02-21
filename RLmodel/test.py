#constants
IOTCOUNT = 3
# TRANS_SPEED ='lte' # 'wifi' #
TIMESLOT = 30
START_TIME = 5
import random
random.seed(2024)
BATCH = {'vgg':3, 'resnet':3, 'densenet':3}
DURATION = 7200 #for move 
LEVEL = None

def recursive_str(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = recursive_str(value)
        else:
            d[key] = str(value)
    return d

from bisect import bisect_left
#IoT device
class IOTDevice:
    energy_consumption = {'A':{"Forward Flight": 199.33,
                               "Vertical Movement": 498.325, #more than twice as forward flight (*2.5)
                               "Rotational Movement": 66.44, #less than half of forward flight (/3)
                               "Hovering":448.59 , "Camera": 0.98}}#, "transmission":15}}
    battery_capacity = {'A':1270080} #in joule
    layerwise_local_energy = {#Jan6 # this should be accumulative too
        'vgg':{
            '11':{0:0, 3: 0.79, 6: 1.55, 11: 2.89, 27:6.17}, #was 26
            '19':{0:0, 5: 1.94, 10: 3.57, 19: 6.19, 43:11.28} #was 41
        },
        'resnet':{
            '18':{0:0, 4: 1.27, 15: 2.78, 20: 2.52, 49:3.73},
            '50':{0:0, 4: 0.77, 13: 2.05, 20: 3.00, 115:7.46},
        },
        'densenet':{
            '121':{0:0, 4: 0.6, 6: 5.6, 8: 11.65, 14:28.00},
            '161':{0:0, 4: 0.68, 6: 6.56, 8: 15.47, 14:50.99}
        }
    }
    layerwise_trans_energy = {#Jan2
        'lte':{
            'vgg':{
                '11':{3:1.53, 6:0.76, 11:0.38, 27:0.002},
                '19':{5:1.53, 10:0.76, 19:0.38, 43:0.002},
            },
            'resnet':{
                '18':{4:0.38, 15:0.19, 20:0.04, 49:0.002},
                '50':{4:0.38, 13:0.38, 20:0.38, 115:0.002},
            },
            'densenet':{
                '121':{4:0.38, 6:0.19, 8:0.09, 14:0.002},
                '161':{4:0.57, 6:0.28, 8:0.14, 14:0.002}
            }
        },
        'wifi':{
            'vgg':{
                '11':{3:0.61, 6:0.30, 11:0.15, 27:0.001},
                '19':{5:0.61, 10:0.30, 19:0.15, 43:0.001},
            },
            'resnet':{
                '18':{4:0.153, 15:0.076, 20:0.019, 49:0.001},
                '50':{4:0.153, 13:0.153 , 20:0.153, 115:0.001},
            },
            'densenet':{
                '121':{4:0.15, 6:0.076, 8:0.038, 14:0.001},
                '161':{4:0.22, 6:0.11, 8:0.057, 14:0.001}
            }
        }
    }
    layerwise_local_latency = {#Jan2
        'vgg':{
            '11':{3:130.45, 6:276.23, 11:486.44, 27:1044.48},
            '19':{5:295.40, 10:568.88, 19:1015.86, 43:1862.89}
        },
        'resnet':{
            '18':{4:110.98, 15:392.10, 20:483.14, 49:627.59},
            '50':{4:110.41, 13:258.63, 20:341.13, 115:984.62},
        },
        'densenet':{
            '121':{4:73.30, 6:892.62, 8:1895.40, 14:4292.17},
            '161':{4:95.25, 6:1092.37, 8:2487.74, 14:7845.49}
        }
    }
    layerwise_trans_latency = {#Jan2
        'lte':{ #8mbps
            'vgg':{
                '11':{-1:574.4 , 3:3062.7, 6:1531.4, 11:765.8, 27:4.0},
                '19':{-1:574.4 , 5:3062.7, 10:1531.4, 19:765.8, 43:4.0},
            },
            'resnet':{
                '18':{-1:574.4 , 4: 765.8, 15:383.0, 20:95.09, 49:4.0},
                '50':{-1:574.4 , 4: 765.8, 13:765.8, 20:765.8, 115:4.0},
            },
            'densenet':{
                '121':{-1:574.4 , 4:765.8, 6:383.0, 8:191.6, 14:4.0},
                '161':{-1:574.4 , 4:1148.6, 6:574.4, 8:287.3, 14:4.0}
            }
        },
        'wifi':{#20mbps
            'vgg':{
                '11':{-1:229.76 , 3:1225.0, 6:613.0, 11:306.0, 27:2.0},
                '19':{-1:229.76 , 5:1225.0, 10:613.0, 19:306.0, 43:2.0}
            },
            'resnet':{
                '18':{-1:229.76 , 4:306.0, 15:153.0, 20:38.0, 49:2.0,},
                '50':{-1:229.76 , 4:306.0, 13:306.0, 20:306.0, 115:2.0},
            },
            'densenet':{
                '121':{-1:229.76 , 4:306.0, 6:153.0, 8:77.0, 14:2.0},
                '161':{-1:229.76 , 4:459.0, 6:230.0, 8:115.0, 14:2.0}
            }
        }
    }
    server_latency = {
        'vgg':{
            '11': {3:54.607, 6:49.66, 11:36.9496, 27:0.0},
            '19': {5:100.12, 10:85.99, 19:57.3252, 43:0.0}
        },
        'resnet':
        {
            '18':{4:13.51, 15:7.08, 20:5.22, 49:0.0},
            '50':{4:28.90, 13:21.95, 20:19.64, 115:0.0},
        },
        'densenet':{
            '121':{4:62.93, 6:47.34, 8:30.93, 14:0.0},
            '161':{4:117.22, 6:92.45, 8:67.46, 14:0.0}
        }
    }
    accuracy = {
        'vgg': {'11':69.04, '19':72.40},
        'resnet': {'18':69.76, '50':76.15},
        'densenet':{'121':74.43, '161':77.11}
    }
    acc_constraint = {'vgg':70.72, 'resnet':75.77, 'densenet':72.95}
    def __init__(self, ID, model, dnn, battery_level, movedata, transmission_speed):
        self.ID = ID
        self.model = model
        self.dnn = dnn
        self.on = True
        self.battery_level = battery_level #percentage
        self.battery_capacity = self.battery_capacity[self.model]
        self.current_battery = self.battery_capacity #in Joule

        # self.energy_rate = {'move':self.energy_consumption[self.model]['move'], 'hover':self.energy_consumption[self.model]['hover'], \
        #                'camera':self.energy_consumption[self.model]['camera']}#, 'transmission':self.energy_consumption[self.model]['transmission']}
        self.energy_rate = {"Forward Flight": self.energy_consumption[self.model]["Forward Flight"],\
                            "Hovering": self.energy_consumption[self.model]["Hovering"], \
                            "Vertical Movement": self.energy_consumption[self.model]["Vertical Movement"],\
                            "Rotational Movement":  self.energy_consumption[self.model]["Rotational Movement"],
                            "Camera": self.energy_consumption[self.model]['Camera']
                            }
        self.busy= False
        self.job = {'profile':None, 'cut':None, 'expected_latency':None, 'expected_energy':None, 'expected_accuracy':None}
        # self.mode = None #forward/hovering
        self.last_update = START_TIME
        self.movedata = movedata #{'times':[], 'modes':}
        self.move_estimate = None
        self.transmission_speed = transmission_speed
        self.accuracy_constraint = self.set_accuracy_constraint()

    
    def load_movedata(self, e, level):
        all_movedata = {}
        times, modes = [], []
        with open(F"{sys.argv[4]}/TestData/movementdata__{e}_{level}.csv",'r') as f:
            for line in f:
              line = line.replace("\n","").split(",")
              times.append(int(line[0]))
              modes.append(line[1])
        self.movedata = {'times':times, 'modes':modes}
      # return all_movedata

    def update_parameters(self, speed, dnn):
        self.transmission_speed = 'lte' if speed==0 else 'wifi'
        dic = {0:'vgg', 1:'resnet', 2:'densenet'}
        self.dnn = dic[dnn]
        self.set_accuracy_constraint()

    def set_accuracy_constraint(self):
        self.accuracy_constraint = self.acc_constraint[self.dnn]
        return self.accuracy_constraint

    def reset_job(self):
        print("------------------------reset jobb --------------------------------")
        self.job['profile'] = None
        self.job['cut'] = None
        self.job['expected_latency'] = None
        self.job['expected_energy'] = None
        self.job['expected_accuracy'] = None

    def reset_device(self):
        self.on = True
        self.battery_level = 10
        self.current_battery = self.battery_capacity
        self.reset_job()

    def get_battery_level(self):
        return self.battery_level

    def get_accuracy_constraint(self):
        return self.accuracy_constraint

    def set_move_estimate(self, estimate):
        self.move_estimate = estimate

    def compute_move_estimate(self, time):
        movement = {  "Vertical Movement": 0,
                      "Rotational Movement": 0,
                      "Forward Flight": 0,
                      "Hovering":0}

        # print("time", time)
        start_index = bisect_left(self.movedata['times'], time)
        end_index =  bisect_left(self.movedata['times'], time+25)
        # print("time", time, start_index, end_index)
        # print(self.movedata['times'][start_index:end_index])
        # print(self.movedata['modes'][start_index:end_index])

        for index in range(start_index, end_index):
            if index+1 < len(self.movedata['times']):
                movement[self.movedata['modes'][index]] += (self.movedata['times'][index+1] - self.movedata['times'][index])/5
                if self.movedata['modes'][index]=="Vertical Movement" and self.movedata['times'][index+1] - self.movedata['times'][index]<0:
                    raise Exception(self.movedata['times'][index+1] - self.movedata['times'][index])
        for key in movement.keys():
            if movement[key]<0:
                raise Exception("WTF??", key, movement[key])
        self.set_move_estimate(movement)
        # print("DONE computing move estimate< ", self.move_estimate)
        # return self.move_estimate

    def get_move_estimate(self):
        return self.move_estimate

    def is_valid(self, profile_index, cut_index, dnns, time):
        # print("in is_valid function")
        profile = list(dnns[self.dnn].keys())[profile_index] \
                    if profile_index in range(len(dnns[self.dnn])) else None
        cut = dnns[self.dnn][list(dnns[self.dnn].keys())[profile_index]][cut_index]\
                                 if profile and cut_index in range(len(dnns[self.dnn][list(dnns[self.dnn].keys())[profile_index]]))\
                                 else None
        # print("PROFILE AND CUT", profile , cut)
        if profile and cut:
            self.job['profile'], self.job['cut'] = profile, cut
            iot_device_latency = self.execution_latency()+ self.transmission_latency()
            self.job['expected_latency'] = iot_device_latency + self.server_latency_f()
            # print("SETTING expected LATENCY")
            # print("expected latency :",iot_device_latency)
            if iot_device_latency>=30:
                raise Exception("latency more than 30 sec ", self.dnn, iot_device_latency)
            self.job['expected_exe_energy'] = self.execution_energy() + self.transmission_energy()
            self.job['expected_energy'] = self.execution_energy() + self.transmission_energy() + self.moving_energy(time+iot_device_latency)
            if self.job['expected_energy']<0:
                raise Exception("Negative expected energy ", self.execution_energy(),self.transmission_energy(),self.moving_energy(time+iot_device_latency))
            self.job['expected_accuracy'] = self.accuracy_()
            return True
        return False

    def execution_energy(self):
        # print("execution_energy",self.dnn, self.job['profile'], self.job['cut'])
        return round(self.layerwise_local_energy[self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn], 4)

    def transmission_energy(self):
        return round(self.layerwise_trans_energy[self.transmission_speed][self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn], 4)

    def execution_latency(self):
        return round(self.layerwise_local_latency[self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn]/1000, 4)

    def transmission_latency(self):
        # print(self.transmission_speed)
        # raise Exception(round(self.layerwise_trans_latency[self.transmission_speed][self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn]/1000, 4))
        return round(self.layerwise_trans_latency[self.transmission_speed][self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn]/1000, 4)

    def server_latency_f(self):
        return round(self.server_latency[self.dnn][self.job['profile']][self.job['cut']]*BATCH[self.dnn]/1000, 4)

    def accuracy_(self):
        return self.accuracy[self.dnn][self.job['profile']]

    def moving_energy(self, finished_time):
        # print("moving energy ", self.movedata['times'][-1], finished_time)
        from bisect import bisect_left
        # print(self.movedata['times'])
        start_index = bisect_left(self.movedata['times'], self.last_update)
        # print(self.movedata['times'])
        # print(finished_time)
        end_index =  bisect_left(self.movedata['times'], finished_time)
        if finished_time < self.movedata['times'][end_index]:
            end_index -= 1
        # print("index", start_index, end_index)
        # print(self.last_update, self.movedata['times'][start_index])
        # print(finished_time, self.movedata['times'][end_index])

        consumed = (self.movedata['times'][start_index+1] - self.last_update) * self.energy_rate[self.movedata['modes'][start_index]] #first part
        # print("CONSUMED1 ", consumed )
        consumed += (finished_time - self.movedata['times'][end_index]) * self.energy_rate[self.movedata['modes'][end_index]] #last part
        # print("CONSUMED2 ", (finished_time - self.movedata['times'][end_index]))
        for index in range(start_index+1, end_index):
          consumed += (self.movedata['times'][index+1] - self.movedata['times'][index]) * self.energy_rate[self.movedata['modes'][index]]
          # print(F"CONSUMED::: {index} ", consumed)
        # print("CONSUMED ", consumed)
        return consumed + consumed * self.energy_rate['Camera']

    def update_battery_level(self, time):
        # print('update battery level ', time-self.last_update, self.move_estimate,  self.job['expected_latency'])
        if time-self.last_update and self.job['expected_latency'] :
            moving = sum([self.move_estimate[mode]*5*self.energy_rate[mode] for mode in ["Vertical Movement", "Rotational Movement","Forward Flight","Hovering"]])
            camera = TIMESLOT*self.energy_rate['Camera']
            consumed_energy = self.execution_energy() + self.transmission_energy() + moving + camera
            if consumed_energy <0:
                raise Exception("used more than they have ",consumed_energy,self.current_battery)
            self.current_battery -= round(consumed_energy,4)
            self.battery_level = int(10*(self.current_battery/self.battery_capacity))
            if self.battery_level<0:
                raise Exception("WAIT WAIT WAIT",self.current_battery,self.battery_capacity)
            if self.battery_level<=1:
                self.reset_job()
                with open(F'TestResults/{self.transmission_speed}/batterylife_{LEVEL}.csv','a') as f:
                    f.write(F"TURN THE DEVICE OFFFF time{time}  deviceID {self.ID}\n") 
                print(F"TURN THE DEVICE OFFFF time{time}  deviceID {self.ID}") 
                self.on = False
            self.last_update = time

    def job_is_done(self, job, time):
        self.running = False
        self.reset_job()
        # self.job = {'profile':None, 'cut':None, 'expected_latency':None, 'expected_energy':None, 'expected_accuracy':None}
        #self.update_battery(time)

    def __str__(self):
        copy_dict = self.__dict__.copy()
        recursive_dict = recursive_str(copy_dict)
        return str(recursive_dict)

layerwise_local_latency = {
        'VGG':{#done
            '11':{-1:0, 2:0.3058, 5:0.4639, 10:0.6406, 26:0.7913},
            '19':{-1:0, 4:0.5558, 9:0.844, 18:1.1465, 41:1.418}
        },
        'Resnet':{#done
            '18':{-1:0, 3:0.43, 14:0.8167, 19:0.9109, 49:1.0504},
            '50':{-1:0, 3:0.43, 13:0.7963, 20:0.9207, 114:1.3186},
        },
        'Densenet':{#done
            '121':{-1:0, 4:0.1863, 6:0.2245, 8:0.2527, 13:0.2772},
            '161':{-1:0, 4:0.2533, 6:0.3055, 8:0.3415, 13:0.3804}
        }
    }

# #tasks
# from collections import namedtuple
# class Job:
#     def __init__(self, deviceID=None, submission_time=None, data=None, dnn_name=None):
#       self.deviceID = deviceID
#       self.submitted_at = submission_time
#       self.data = data
#       self.dnn_name = dnn_name
#       self.dnn_version = None
#       self.cutpoint = None
#       self.transmission_time = 0
#       self.local_latency = 0
#       self.remote_latency = 0
#       self.finished_at = None

#     def local_delay_estimation(self):
#       return #ToBeCompleted
#     def remote_delay_estimation(self):
#       return #ToBeCompleted

#     def setversion(self, version):
#         self.version = versionf.w

#     def setcutpoint(self, cut):
#         self.cutpoint = cut

#     def __gt__(self, other):
#         if isinstance(other, Job):
#             return self.submitted_at >= other.submitted_at
#         return NotImplemented

#     def __lt__(self, other):
#         if isinstance(other, Job):
#             return self.submitted_at < other.submitted_at
#         return NotImplemented

#     def get_summary_completion(self):
#         return f"req workload: {self.req_cpu_and_time} \ncores at clock instance: {self.cores_at_clock_instance}"

#     def __str__(self):
#         copy_dict = self.__dict__.copy()
#         recursive_dict = recursive_str(copy_dict)
#         return str(recursive_dict)+"\n"

def simulate_movedata(seed, deviceID, duration = DURATION):
    import random
    random.seed(seed)
    simulation_time = duration
    hover_duration = 5 #to capture the photo
    movement_interval = 25

    movement_list = [ # Define the list of possible movements with their probabilities
        ("Vertical Movement", 0.30),
        ("Rotational Movement", 0.10),
        ("Forward Flight", 0.60)
    ]
    current_movement = None
    movement_change_times = []
    current_time = 0
    time_step = 5

    while current_time < simulation_time:
        time_within_pattern = current_time % (movement_interval + hover_duration)

        if time_within_pattern < hover_duration:
            # Drone is hovering
            new_movement = "Hovering"
        else:
            # Choose a random movement based on the specified probabilities
            selected_movement = random.choices(movement_list, [prob for _, prob in movement_list])[0]
            new_movement, _ = selected_movement

        if new_movement != current_movement:
            # Drone's movement mode has changed
            movement_change_times.append([current_time, new_movement])
            current_movement = new_movement

        # Increment the time counter
        current_time += time_step
    # print("before write", movement_change_times)
    with open(f'../DNN Inference Project/movementdata_{deviceID}.csv', 'w') as f:
      #Colab Notebooks/DNNInference/
    #   print("OPEN FILE ", deviceID)
      for row in movement_change_times:
        f.write(",".join([str(i) for i in row])+"\n")
    # print("Simulation complete.")

def load_movedata(deviceIDs, duration = DURATION, simulate = False):
  if simulate:
    print("....... SIMULATING MOVE DATA ..........")
    for dID in deviceIDs:
        simulate_movedata(seed=int(dID), deviceID=dID, duration = duration)
  all_movedata = {}
  for dID in deviceIDs:
    times, modes = [], []
    #with open(F"DNN Inference Project/movementdata_{dID}.csv",'r') as f:
    #with open(F"TestData/movementdata__0_High.csv",'r') as f:
    #    for line in f:
    #      line = line.replace("\n","").split(",")
    #      times.append(int(line[0]))
    #      modes.append(line[1])
    all_movedata[dID] = {'times':times, 'modes':modes}
  return all_movedata

def generate_iots():
    IOTdevices = []
    deviceIDs = ['001', '002', '003']
    dnns = list(get_all_dnns().keys())
    drone_models = ['A', 'A','A']
    movedata = load_movedata(deviceIDs, duration = DURATION)
    for i in range(IOTCOUNT):
        # print(deviceIDs[i])
        # print(drone_models[i])
        # print(dnns[i])
        IOTdevices.append(IOTDevice(ID =deviceIDs[i], model=drone_models[i], dnn=dnns[i], battery_level=10, movedata = movedata[deviceIDs[i]], transmission_speed=None))
    return IOTdevices

def get_all_dnns(): #consider reading from a file
    dnns = {'vgg': {
                '11':[3, 6, 11, 27],
                '19':[5, 10, 19, 43],
                },
            'resnet': {
                '18':[4, 15, 20, 49],
                '50':[4, 13, 20, 115],
            },
            'densenet': {
                '121':[4, 6, 8, 14],
                '161':[4, 6, 8, 14],
            }
        }
    return dnns

def environment_components():
    IOTDevices = generate_iots()
    dnns = get_all_dnns()

    return IOTDevices, dnns

# @title Default title text
"""Reinforcement Learning

Creating the gym environment
"""
import gym
from gym import spaces
import numpy as np
import math
import heapq

class EdgeIoTEnv(gym.Env):
    def __init__(self, number_of_iots=IOTCOUNT): #lte
        #self.jobs, self.mode = simulation_environment()
        # self.transmission_speed = TRANS_SPEED
        self.iots, self.dnns = environment_components()
        self.number_of_iots = number_of_iots
        self.rem_step = 1
        #self.action_dict, self.action_count = self.action_dict_generator()

        self.processing_q = []

        #states are defined as the battery level, moving mode, and if they have a job to be run
        observation_space = [10, 2, 2, 3, 6, 6, 6]*self.number_of_iots
        self.observation_space = spaces.MultiDiscrete(observation_space)
        self.action_space = spaces.MultiDiscrete([2,4]*self.number_of_iots) #spaces.Discrete(number_of_vens+1)
        self.state = [0, 0, 0, 0, 0, 0, 0]*self.number_of_iots
        self.time = None

    # def action_dict_generator(self):
    #     action_dic = {}
    #     counter = 1
    #     for vggprofile in self.dnn['vgg'].keys():
    #         row = [vggprofile]
    #         for vggcut in self.dnn['vgg'][vggcut]:
    #             row.append(vggcut)f
    #             for resnetprofile in self.dnn['resnet'].keys():
    #                 row = [resnetprofile]
    #                 for resnetcut in self.dnn['resnet'][resnetcut]:
    #                     row.append(resnetcut)
    #                     for densenetprofile in self.dnn['densenet'].keys():
    #                         row = [densenetprofile]
    #                         for densenetcut in self.dnn['densenet'][densenetcut]:
    #                             row.append(densenetcut)
    #                             action_dic[counter] = row
    #                             counter += 1
    #     return action_dic, len(action_dic)
    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def reset(self):
        one_state = np.array([0, 0, 0, 0, 0, 0, 0]*self.number_of_iots)
        self.state = np.array([one_state for i in range(self.rem_step)])
        self.state = np.expand_dims(self.state, axis=0)
        return self.state

    def update_state(self, state): #update state based on job queues
        # new_state = np.array([0, 0, 0, 0, 0]*self.number_of_iots)
        # state = np.array([0, 0, 0, 0, 0]*self.number_of_iots)
        for i in range(len(state[0][0])):
            state[0][0][i] = 0
        for i in range(self.number_of_iots):
          if self.iots[i].on:
            state[0][0][7*i] = self.iots[i].get_battery_level()
            state[0][0][7*i+1] = self.iots[i].on #self.iots[i].job
            state[0][0][7*i+2] = int(sys.argv[3])*10#0 #random.choice([0,1]) #transmission speed lte, wifi
            state[0][0][7*i+3] = i #random.choice([0,1,2]) #dnn vgg, resnet, densenet            
            state[0][0][7*i+4:7*i+7] = [i for i in list(self.iots[i].get_move_estimate().values())[:-1]] #omitting hovering info
            self.iots[i].update_parameters(speed =state[0][0][7*i+2], dnn=state[0][0][7*i+3])
        # state[0] = np.roll(state[0], 1, axis = 0)
        # state[0,0,:] = new_state
        return state

    def latency_reward(self, index):
        latency_reward = 0
        dnn, profile, cut = self.iots[index].dnn, self.iots[index].job['profile'], self.iots[index].job['cut']
        transmission_speed = self.iots[index].transmission_speed
        local_only = round(list(IOTDevice.layerwise_local_latency[dnn][profile].items())[-1][1]*BATCH[dnn]/1000, 4)\
                        + round(list(IOTDevice.layerwise_trans_latency[transmission_speed][dnn][profile].items())[-1][1]*BATCH[dnn]/1000, 4)
        latency_reward = 1 - self.iots[index].job['expected_latency']/local_only
        # latency_reward = (latency_reward - 0.996363)/(0.997-0.996365)
        return max(latency_reward,-0.0001)

    def accuracy_reward(self, index):
        a, b = 1/2, 71.12#10, 1/2
        accuracy_reward = 0
        dnn, profile, cut = self.iots[index].dnn, self.iots[index].job['profile'], self.iots[index].job['cut']
        acc = IOTDevice.accuracy[dnn][profile]
        # if self.iots[index].get_accuracy_constraint()-acc >2:
        #     return -1
        accuracy_reward = 1 / (1 + np.exp(-a * (acc - b)))
        # max(0, min(1, 1 / (1 + np.exp(-a * (acc - b)))))
        return accuracy_reward

    def energy_reward(self, index, action):
        energy_reward = 0
        if self.iots[index].job['expected_energy'] > self.iots[index].current_battery:
            self.iots[index].reset_job()
            if action[index*2]==0 and action[index*2+1]==0:
                return -0.5
            else:
                return -1
        dnn, profile, cut = self.iots[index].dnn, self.iots[index].job['profile'], self.iots[index].job['cut']
        local_only = list(IOTDevice.layerwise_local_energy[dnn][profile].items())[-1][1]*BATCH[dnn]\
                            + list(IOTDevice.layerwise_trans_energy[self.iots[index].transmission_speed][dnn][profile].items())[-1][1]*BATCH[dnn]
        energy_reward = 1 - (self.iots[index].job['expected_exe_energy']/local_only)
        # print(self.iots[index].job['expected_exe_energy'], local_only)
        return energy_reward

    def reward_calculation(self, action):
        reward = 0.0
        # w1, w2, w3 = 0.2, 0.4, 0.4 #weights for acc, latency, energy
        w1, w2, w3 = 0.333, 0.333, 0.333 #weights for acc, latency, energy
        for i in range(self.number_of_iots):
            if self.iots[i].on:
                latency = self.latency_reward(i)
                accuracy = self.accuracy_reward(i)
                energy_consumption = self.energy_reward(i, action)
                # reward = math.sqrt(accuracy*latency)/energy_consumption
                with open("reward_values.csv", "a") as f:
                    f.write(F"{accuracy},{latency},{energy_consumption}\n")
                reward += (w1*accuracy + w2*latency + w3*energy_consumption)/self.number_of_iots
                if reward ==1    :
                    raise Exception("High reward", reward, accuracy, latency, energy_consumption)
        return round(reward, 4)

    def step(self, action):
        done = False
        assert self.action_space.contains(action)
        reward = 0.0

        #check action validation
        valid = True
        for i in range(0, len(action), 2):
            if self.iots[i//2].on:
                valid &= self.iots[i//2].is_valid(action[i], action[i+1], self.dnns, self.time)

        if valid:
            # print("It is valid")
            reward = self.reward_calculation(action)
            #start running the job from the iot device
            for i in range(0, len(action), 2):
                if self.iots[i//2].on and self.iots[i//2].job['expected_latency']:
                    # print("heap push", self.time, self.iots[i//2].job['expected_latency'])
                    # print(self.processing_q)
                    heapq.heappush(self.processing_q, (self.time + self.iots[i//2].job['expected_latency'],  i//2, self.iots[i//2].job))
        else:
            print("NOT VALID")
            reward = -1

        print('reward :::', reward)
        if reward < -1:
          raise Exception("WHY THIS VALUE? ", reward)
        return (self.state, reward, done, {})

    # def render(self, mode='human'):
    #     print(self.state)

gym.register(id='EdgeIoTEnv-RS', entry_point='__main__:EdgeIoTEnv')

# env = gym.make('EdgeIoTEnv-RS')

"""**Learning**"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import gym

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#import pylab

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    raise Exception(f'GPUs {gpus}')
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

# Create a custom callback to print activations at each layer during training
class ActivationPrintCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch:", epoch)
        for i, layer in enumerate(self.model.layers):
            activation = layer.get_weights()
        print()

# def OurModel_original(input_shape, action_space, lr):
#     X_input = Input(input_shape)
#     number_of_outputs = 3

#     #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
#     #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
#     #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
#     X = Flatten(input_shape=input_shape)(X_input)

#     X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
#     X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
#     #X = Dense(128, activation="elu", kernel_initializer='he_uniform')(X)
#     #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

#     output_pairs = []
#     # for i in range(numberofoutputs):
#     #     # print(action_space, action_space.nvec[i], type(action_space.nvec[i]))
#     #     output = Dense(action_space.nvec[i], activation="softmax", kernel_initializer='he_uniform', name=f'output_{i+1}')(X)
#     #     output_action.append(output)
#     value = Dense(1, kernel_initializer='he_uniform')(X)
#     for i in range(0, 2 * number_of_outputs, 2):
#         #print("action space size ", action_space.nvec[i], action_space.nvec[i+1])
#         output_layer_1 = Dense(action_space.nvec[i], activation="softmax", kernel_initializer='he_uniform', name=f'output_{i + 1}')(X_input)
#         output_layer_2 = Dense(action_space.nvec[i + 1], activation="softmax", kernel_initializer='he_uniform', name=f'output_{i + 2}')(X_input)
#         output_pairs.append([output_layer_1, output_layer_2])

#     Actor = Model(inputs = X_input, outputs = output_pairs)
#     Actor.compile(loss = ['categorical_crossentropy'] * (2 * number_of_outputs),
#               optimizer = RMSprop(learning_rate=lr),
#               loss_weights = [1.0] * (2 * number_of_outputs))

#     Critic = Model(inputs = X_input, outputs = value)
#     Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
#     return Actor, Critic

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)
    print("X INPUUUT------------------------------", X_input)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)

    output_layers = []
    # Create output layers for each dimension
    for i, num_options in enumerate(action_space.nvec):
        # Shared layers for related dimensions
        if i in [0, 1]:
            shared_layer = Dense(128, activation="relu", name=f'shared_layer_{i+1}')(X_input)
            output_layer = Dense(num_options, activation="softmax", kernel_initializer='he_uniform', name=f'output_{i+1}')(shared_layer)
        elif i in [2, 3]:
            shared_layer = Dense(128, activation="relu", name=f'shared_layer_{i+1}')(X_input)
            output_layer = Dense(num_options, activation="softmax", kernel_initializer='he_uniform', name=f'output_{i+1}')(shared_layer)
        elif i in [4, 5]:
            shared_layer = Dense(128, activation="relu", name=f'shared_layer_{i+1}')(X_input)
            output_layer = Dense(num_options, activation="softmax", kernel_initializer='he_uniform', name=f'output_{i+1}')(shared_layer)
        else:
            # Non-shared layers for independent dimensions
            output_layer = Dense(num_options, activation="softmax", kernel_initializer='he_uniform', name=f'output_{i+1}')(X_input)

        output_layers.append(output_layer)
    # Concatenate the output layers for all dimensions
    concatenated_outputs = Concatenate(name='concatenated_outputs')(output_layers)
    Actor = Model(inputs = X_input, outputs = concatenated_outputs)
    Actor.compile(loss = 'categorical_crossentropy',#optimizer='adam')
              optimizer = RMSprop(learning_rate=lr),
              loss_weights = [1.0] * (2 * len(action_space.nvec)))

    value = Dense(1, kernel_initializer='he_uniform')(X)
    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))
    return Actor, Critic

import traceback

def process_exception(exception):
    print("Exception Handling function which is mine")
    # Extract the exception message
    exception_message = str(exception)

    # Get the traceback information
    traceback_info = traceback.format_exc()

    # Construct links or relevant information
    link1 = "[Link 1](https://example.com/exception1)"
    link2 = "[Link 2](https://example.com/exception2)"

    # Create a formatted message with links
    formatted_message = f"{exception_message}\n\nTraceback:\n{traceback_info}\n\nLinks:\n{link1}\n{link2}"
    print("the error detail ", formatted_message)
    # Raise a new exception with the formatted message
    raise Exception(formatted_message)

class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    exception_event = threading.Event()
    def __init__(self, env_name='EdgeIoTEnv-RS', number_of_iots = IOTCOUNT):
        # Initialization
        self.env_name = env_name
        self.env = gym.make(env_name)
        # self.action_size = self.env.action_space.n
        self.action_size = np.prod(self.env.action_space.nvec)
        self.EPISODES, self.episode, self.max_average = 2000, 0, -1
        self.lock = Lock()
        self.lr = 0.00005 #0.01#0.001#0.000025

        self.ROWS = 7*number_of_iots
        #self.COLS = 80
        self.REM_STEP = 1

        # Instantiate plot memoryf
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = f'Models_{IOTCOUNT}_{self.lr}'
        self.state_size = (self.REM_STEP, self.ROWS)

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}_{}'.format(self.env_name, IOTCOUNT, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        # self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)
        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.env.action_space, lr=self.lr)


    def on_epoch_end(self, epoch, logs=None):
        print("Epoch:", epoch)
        for i, layer in enumerate(self.actor.layers):
            activation = layer.output
            print("Layer", i+1, "Activation:", activation)
        print()

    def act_original(self, state):
        # Use the network to predict the next action to take, using the model
        #temp = self.Actor.predict(state)
        predictions = self.Actor.predict(state)#[0]

        # Separate predictions for each pair
        separated_predictions = []
        for i in range(0, len(predictions), 2):
            predictions_1, predictions_2 = predictions[i], predictions[i + 1]
            separated_predictions.extend([predictions_1, predictions_2])

        # if np.count_nonzero(np.isnan(prediction))!=0:
        #   print("NaN values ",np.count_nonzero(np.isnan(prediction)), len(prediction))
        actions = [np.random.choice(np.arange(len(probabilities)), p=probabilities) for probabilities in separated_predictions]
        # action = np.random.choice(self.action_size, p=prediction)
        #details = self.env.action_dict[action]
        print(actions)
        return actions#details

    def act(self, state):
        predictions = self.Actor.predict(state)[0]
        # raise Exception("Prediction ", predictions)
        actions = [
            np.random.choice(np.arange(options), p = predictions[0, sum(self.env.action_space.nvec[:i]):sum(self.env.action_space.nvec[:i+1])])
            for i, options in enumerate(self.env.action_space.nvec)
        ]
        return actions

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.01 #0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype=np.float64)
        for i in reversed(range(0,len(reward))):
            # if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
            #     running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        print('reward ', reward)
        discounted_r -= np.mean(discounted_r, dtype=np.float64) # normalizing the result

        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards):
        # print("IN Replay")
        # print(self.Actor.summary())
        # print(self.Critic.summary())
        # print(len(states))
        # print(len(actions), actions)
        # print(actions[0].shape)
        # print(len(rewards))
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)

        # Get Critic network predictions
        value = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - value
        # training Actor and Critic networks
        #for i in range(len(states)):
        #  print(states[i],"\t",actions[i],"\t", advantages[i])
        # print(len(actions), len(states))
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)#, callbacks=[ActivationPrintCallback()])
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        self.Critic.save(self.Model_name + '_Critic.h5')

    plt.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            plt.plot(self.episodes, self.scores, 'b')
            plt.plot(self.episodes, self.average, 'r')
            plt.ylabel('Score', fontsize=18)
            plt.xlabel('Steps', fontsize=18)
            try:
                plt.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def reset(self, env):
        state = env.reset()
        return state

    def step(self, action, env, state):
        x = env.step(action)
        next_state, reward, done, info = x
        return next_state, reward, done, info

    def run(self):
        for e in range(self.EPISODES):
            state = self.reset(self.env)
            done, score, SAVING = False, 0, ''
            # Instantiate or reset games memory
            states, actions, rewards = [], [], []
            while not done:
                #self.env.render()
                # Actor picks an action
                action = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.step(action, self.env, state)
                # Memorize (state, action, reward) for training
                states.append(state)
                action_onehot = np.zeros([self.action_size])

                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_averagstepe = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                    print("type of replay ", type(self.replay))
                    self.replay(states, actions, rewards)
         # close environemnt when finish training
        self.env.close()

    def train(self, n_threads):
        #global exception_event
        self.env.close()
        # Instantiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # Create threads
        threads = [ threading.Thread(target=self.train_threading, daemon=False, #True,
                                     args=(self, envs[i], i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
            if A3CAgent.exception_event.is_set():
              # Retrieve the exception
              raised_exception = A3CAgent.exception_event.exception
              # Handle the exception as needed
              process_exception(A3CAgent.exception_event)
            else:
              # No exception occurred within the given timeout period
              print("No exception occurred within the timeout period")

        for t in threads:
            time.sleep(10)
            t.join()

    def train_threading(self, agent, env, thread):

      #global exception_event
      try:
          while self.episode < self.EPISODES:
              # Reset episode
              score, done, SAVING = 0, False, ''
              state = self.reset(env)
              # Instantiate or reset games memory
              states, actions, rewards = [], [], []
              #debuggingactions = []

              env.set_time(START_TIME)
              for i in range(env.number_of_iots):
                env.iots[i].reset_device()
              env.processing_q.clear()

              print(self.episode, "........................... episode .......................")
            #   print("drone ons out:",[drone.on for drone in env.iots])
              while any([env.iots[i].on for i in range(env.number_of_iots)]) and env.time<DURATION*(7/8): #as long as one of the drones is on
                print("loop time ", env.time)
                # print([env.iots[i].current_battery for i in range(env.number_of_iots)])
                while env.processing_q and env.processing_q[0][0] <=env.time:
                    # print(env.processing_q)
                    temp = heapq.heappop(env.processing_q)
                    done_job = temp[1]
                # print("drone ons:",[drone.on for drone in env.iots])
                for i in range(env.number_of_iots):
                    if env.iots[i].on:
                        # print(F"DEVICE {i} IS ON {env.iots[i].on}")
                        env.iots[i].compute_move_estimate(env.get_time())
                        env.iots[i].update_battery_level(env.get_time())

                #I need to update batteries here
                # print( "1")
                state = env.update_state(state)
                print("state ", state)
                # print("2")
                action = agent.act(state)
                print("action ", action)
                # raise Exception("Action ",action)
                # print("3")
                temp = np.copy(state)
                states.append(temp)
                next_state, reward, done, _ = self.step(action, env, state)
                print("new state", next_state)
                # print("4")
                one_hot_encoded_outputs = []
                for i, num_options in enumerate(env.action_space.nvec):
                    dimension_actions = np.array(action[i])
                    one_hot_encoded_dim = tf.one_hot(dimension_actions, depth=num_options)
                    # print("A ", one_hot_encoded_dim.shape)
                    one_hot_encoded_dim = tf.reshape(one_hot_encoded_dim, (-1, num_options))
                    # print("B ", one_hot_encoded_dim.shape)
                     # Ensure consistent shape (None, 1, num_options)
                    one_hot_encoded_outputs.append(one_hot_encoded_dim)

                one_hot_encoded_actions = tf.concat(one_hot_encoded_outputs, axis=-1)#axis=1)
                # print("shape ", one_hot_encoded_actions.shape)
                # Add an extra dimension at axis 1 after concatenation

                # tf.expand_dims(self.state, axis=0)
                actions.append(tf.expand_dims(one_hot_encoded_actions, axis=0))

                # print("5")
                rewards.append(reward)
                score += reward
                state = next_state
                env.set_time(env.time+30)
                # if env.time>100:
                #     break

              self.lock.acquire()
            #   print("5- 1")
              self.replay(states, actions, rewards)
            #   print("5 - 2")
              self.lock.release()

            #   print("6")
              with self.lock:
                  average = self.PlotModel(score, self.episode)
                #   print("7")
                  # saving best models
                  if average >= self.max_average:
                      self.max_average = average
                      self.save()
                      SAVING = "SAVING"
                  else:
                      SAVING = ""
                #   print("8")
                  if(self.episode < self.EPISODES):
                      self.episode += 1

          env.close()
      except Exception as e:
          traceback.print_exception(type(e), e, e.__traceback__)
          A3CAgent.exception_event.set()
          A3CAgent.exception_event.exception = e

    def test(self, Actor_name, Critic_name, level):
        self.load(Actor_name, Critic_name)
        for e in range(50):
            score, done, SAVING = 0, False, ''
            state = self.env.reset()
            states, actions, rewards = [], [], []
            self.env.set_time(START_TIME)
            for i in range(self.env.number_of_iots):
                self.env.iots[i].reset_device()
                self.env.iots[i].load_movedata(e, level)
            self.env.processing_q.clear()

            while any([self.env.iots[i].on for i in range(self.env.number_of_iots)]) and self.env.time<DURATION*(7/8): #as long as one of the drones is on
                while self.env.processing_q and self.env.processing_q[0][0] <=self.env.time:
                    temp = heapq.heappop(self.env.processing_q)
                    done_job = temp[2]
                    with open(F"TestResults/{self.env.iots[i].transmission_speed}/jobs_{e}_{level}.csv", 'a') as f:
                        # raise Exception("done job details ", done_job, type(done_job))
                        f.write(str(done_job)+"\n")

                for i in range(self.env.number_of_iots):
                    if self.env.iots[i].on:
                        self.env.iots[i].compute_move_estimate(self.env.get_time())
                        self.env.iots[i].update_battery_level(self.env.get_time())

                state = self.env.update_state(state)
                print("state ", state)
                action = agent.act(state)
                print("action ", action)
                next_state, reward, done, _ = self.step(action, self.env, state)
                print("reward ", reward)
                state = next_state
                self.env.set_time(self.env.time+30)
            
        self.env.close()

if __name__ == "__main__":
    agent = A3CAgent()
    # agent.train(n_threads=1) # use as A3C
    import sys
    LEVEL = sys.argv[1]
    address = sys.argv[2]+"Models_3_5e-05/"
    agent.test(address+'EdgeIoTEnv-RS_A3C_3_5e-05_Actor.h5',address+'EdgeIoTEnv-RS_A3C_3_5e-05_Critic.h5', LEVEL)
