import pandas as pd
import numpy as np
import new_data_processing as dp
errorControl = pd.read_csv("errors for control set.csv")
errorControl = list(np.transpose(np.array(errorControl))[0])
errorControl.append(0)
errorRT = pd.read_csv("new errors for RT set.csv")
errorRT = list(np.transpose(np.array(errorRT))[0])
errorRT.append(0)
errorPD = pd.read_csv("PD1 errors.csv")
errorPD = list(np.transpose(np.array(errorPD))[0])
errorPD.append(0)
errorCTLA4 = pd.read_csv("PD1 CTLA4 errors.csv")
errorCTLA4 = list(np.transpose(np.array(errorCTLA4))[0])
param = pd.read_csv("mean of each parameter for RT set.csv")
param = list(np.transpose(np.array(param))[0])
param[26] = 0.13795567390561228
param[27] = 0.4073542114448485
param[28] = 0.04813514085703568
param[33] = 0.0897670603865841
param.append(2.2458318956090505*10**80)
#print(get_equivalent_bed_treatment(param, 50, 1))
errors = [errorControl, errorRT]
errorMerged = dp.merge_lists(errors)
errorMerged = dp.merge_lists([errorMerged, errorPD])
errorMerged = dp.merge_lists([errorMerged, errorCTLA4])
errorMerged[22] = 0
errorMerged[32] = 0
num_patients = 500
params = [list(param) for _ in range(num_patients)]
#print(errorMerged)

seeds = range(num_patients)
# Modify the parameters for each patient
for i in range(num_patients):
    rng = np.random.default_rng(seeds[i])
    for j in range(len(param)):
        
        if errorMerged[j] != 0:
            logNormalParams = dp.log_normal_parameters(param[j], errorMerged[j])
            params[i][j] = min(max(rng.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 2*param[j])
        else:
            params[i][j] = param[j]
        if j == 26:
            params[i][j] = min(max(rng.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 1.2*param[j])

# Assuming 'params' is your list of parameters
df = pd.DataFrame(params)
print(params)
# Save to CSV
df.to_csv('parameters.csv', index=False)
