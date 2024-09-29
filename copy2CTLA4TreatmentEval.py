import numpy as np
import concurrent.futures
import new_data_processing as dp
from differential_equations import radioimmuno_response_model
from BED import get_equivalent_bed_treatment
import pandas as pd
import time
import matplotlib.pyplot as plt
start_time = time.time()
# Define the number of patients
#num_patients = 10
def get_treatment_and_dose(bioEffDose, numRT, param, numPD, numCTLA4):
  RTschedule_list = get_treatment_schedules(numRT, 10)
  if numPD > 0:
    PDschedule_list = get_treatment_schedules(numPD, 10)
  else:
    PDschedule_list = []
  if numCTLA4 > 0:
      CTLA4schedule_list = get_treatment_schedules(numCTLA4, 10)
  else:
      CTLA4schedule_list = []
  schedule = []
  for x in RTschedule_list:
    for y in PDschedule_list:
        for z in CTLA4schedule_list:
            schedule.append([x, y, z])
  DList = []
  D = get_equivalent_bed_treatment(param, bioEffDose, numRT)
  for i in range(len(schedule)):
    DList.append(D)
  return schedule, DList

#recursive function to obtain treatment schedules
def get_treatment_schedules(n, start):
  schedule_list = []
  if n == 1:
    #return [[x] for x in range(start, 16)]
    return [[x] for x in range(start, 31)]
  else:

    return [[y] + rest for y in range(start, 31) for rest in get_treatment_schedules(n-1, y + 1)]

#initialising parameters
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = True
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 50
delta_t = 0.05
# t_treat_c4 = np.zeros(3)
# t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0
PD_fractions = 1
CTLA4_fractions = 1
         #print('errors', errorMerged)
all_res_list = []
IT = (True, True)
RT_fractions = 1
param = pd.read_csv('mean of each parameter for RT set.csv')
param = list(np.transpose(np.array(param))[0])
param[26] = 0.13795567390561228
param[27] = 0.4073542114448485
param[28] = 0.04813514085703568
param[33] = 0.0897670603865841
param.append(2.2458318956090505*10**80)
bed = 80
initial_cell_count = 100000
file_name = 'new RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' CTLA4 ' + str(CTLA4_fractions) + ' ' + str(initial_cell_count) + ' c.csv'
schedule_list, DList = get_treatment_and_dose(bed, RT_fractions, param, PD_fractions, CTLA4_fractions)
#print('Dlist', DList)
#print(schedule_list)
# paramNew = list(param)

# Generate a list of seeds

# Assuming 'params' is your list of parameters
# Load from CSV
params = pd.read_csv('parameters.csv').values.tolist()
sample_size = len(params)
initial_cell_count = 100000
#print(params)
# Now you have a list of parameters for each patient
# You can now evaluate the treatment schedules in parallel
def evaluate_patient(i, t_rad, t_treat_p1, t_treat_c4, D):
  # Create a new random number generator with a unique seed for each patient
  paramNew = params[i]
  paramNew[22] = 0.2/CTLA4_fractions
  paramNew[32] = 0.6/PD_fractions
  paramNew[0] = initial_cell_count
    #print(paramNew)


  # t_rad = np.array(schedule_list[i][0])
  # #t_treat_c4 = np.zeros(3)
  # #t_treat_p1 = np.zeros(3)
  # t_treat_p1 = np.array(schedule_list[i][1])
  # t_treat_c4 = np.array(schedule_list[i][2])
  #t_f2 = max(schedule_list[i][0][0], schedule_list[i][1][0]) + 30
  t_f2 = max(t_rad[-1], t_treat_p1[-1]) +30
  #print('t_f2', t_f2)
  # if not isinstance(t_f2, int):
  #     t_f2 = t_f2[0]

  vol, _, Time, _, C, *_ = radioimmuno_response_model(paramNew, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #print(C)
  # plt.figure()
  # plt.plot(Time, vol[0], color='red')
  # plt.show()
  # plt.close()
  # plt.figure()
  # plt.plot(Time, C[0], color='blue')
  # plt.show()
  # plt.close()
  TCP, min_C_time = get_TCP(C, Time)
  return [TCP, min_C_time]

def get_TCP(C, times):
    for i in range(len(C[0])):
      if float(C[0][i]) == np.min(C):
        return np.exp(-1*np.min(C)), times[i]

def trial_treatment(i, file):
  t_rad = schedule_list[i][0]
  # print('rad', t_rad)
  t_treat_p1 = schedule_list[i][1]
  t_treat_c4 = schedule_list[i][2]
  # print('p1', t_treat_p1)
  t_f2 = max(max(t_rad[-1], t_treat_p1[-1]), t_treat_c4[-1]) + 30
  #print('trial t_f2', t_f2)
  treatment_times = []
  treatment_times_list = []
  D = DList[i]
  args = [(j, t_rad, t_treat_p1, t_treat_c4, D) for j in range(sample_size)]
  #print('args', args)
  res_list = []
  with concurrent.futures.ThreadPoolExecutor() as executor:
          res = list(executor.map(lambda p: evaluate_patient(*p), args))
          res = np.transpose(np.array(res))
          res_list.append(res)
          #print('res', res)
          TCPs = res[0]
          times = res[1]
          #print('tcp', TCPs)
          #print(times)
  treatment_times = [x for x in treatment_times if np.isnan(x) == False]
  treatment_res_list = [t_rad, D, t_treat_p1, 0.6/PD_fractions, t_treat_c4, 0.2/CTLA4_fractions, np.mean(TCPs), np.mean(times), TCPs, times]
  return treatment_res_list

# Define the treatment schedules and doses
#schedules, DList = get_treatment_and_dose(90, RT_fractions, param, PD_fractions, CTLA4_fractions)
#print(schedules)
iterations = len(schedule_list)  # Or any other number of iterations
#param_file = open('parameters.txt', 'w')
# for k in range(min(iterations,50)):
#     print('k', k)
#     print(params[k])
args = [(k, params) for k in range(1600, min(iterations,2400))]
# Use a ThreadPoolExecutor to run the iterations in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    data = list(executor.map(lambda p: trial_treatment(*p), args))

    #print(data)
    # Retrieve results from completed futures

#print(data)
dataFrame = pd.DataFrame(data, columns=["RT Treatment Days", "RT Dose (Gy)", "anti-PD-1 Treatment Days", "anti-PD-1 Dose (mg)", "anti-CTLA-4 Treatment Days", "anti-CTLA-4 Dose (mg)", "Mean TCP", "Mean Time to reach Minimum Tumour Cell Count", "List of TCPs", "List of Times to Reach Minimum Tumour Cell Count"])
print(dataFrame)
dataFrame.to_csv(file_name, index=False)
end_time = time.time()
f = open('time taken RT ' + str(RT_fractions) + ' treatment eval c constant seed.txt', 'w')
f.write("TIME TAKEN " + str(end_time - start_time))
print("TIME TAKEN " + str(end_time - start_time))
f.close()
