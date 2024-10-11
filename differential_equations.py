import matplotlib.pyplot as plt
import numpy as np


def growth(lambda_1, C_tot, lambda_2):
    # Logistic proliferation
    return lambda_1 * (1 - lambda_2 * C_tot)


def natural_release(rho, C):
    return rho * C


def RT_release(psi, C):
    return max(0, psi * C)


def A_natural_out(sigma, A):
    return -1 * sigma * A


def immune_death_T(iota, C, T):
    return -1 * iota * T * C


def T_natural_out(eta, T):
    return - eta * T


def tumor_volume(C, T, vol_C, vol_T):
    return C * vol_C + T * vol_T


def tum_kinetic(phi, tau_1, tau_2, t):
    if t <= tau_1:
        a = 0
    elif t > tau_2:
        a = 1
    else:
        a = (t - tau_1) / (tau_2 - tau_1)
    return -1 * a * phi


def immune_death_dePillis(C, T, p, q, s, p1, p_1, mi, vol_flag, time_flag, t, t_treat, delta_t, j=None):
    # if j!= None and j>650:
    #         print(j)
    #         print("Ta immune1", Ta_lym[:,j])
    m = 0
    # if j!= None and j>650:
    #         print("Ta immune1", Ta_lym[:,j])
    if vol_flag == 0 or time_flag == 0:
        pass
    else:
        if abs(t - t_treat) < delta_t / 2:
            p_1 = p_1 - mi * p_1 * delta_t + p1
            m = 1
        else:
            p_1 = p_1 - mi * p_1 * delta_t
    # if j!= None and j>650:
    #        print("Ta immune2", Ta_lym[:,j])
    if C == 0:
        f = 0
    else:
        # print((s + (T / C) ** q))
        f = p * (1 + p_1) * (T / C) ** q / (s + (T / C) ** q)
        if np.isnan(f):
            f = 0  # p * (1 + p_1)
        # if j <= 550:
        #     print('inside function', f)
    return f, m, p_1


def markov_TCP_analysis(im_death, prol, C, delta_t):
    #print('C', C)
    cell_num = int(np.rint(C))
    # print("cell coubt", cell_num)
    f = prol * delta_t  # Birth probability
    g = im_death * delta_t  # Dead probability
    #print('f', f)
    #print('g', g)
    e = f + g

    # normalises the probabilities if the sum is more than 1
    if e > 1:
        f = f/e
        g = g/e
    # generates an array choosing whether the cells multiplie, die or stay constant
    # nested min max for probability of staying constant makes sure probability stays between 0 and 1
    nothingProbability = min(max(0, 1 - f - g), 1)

    # print("birth", type(f))
    # print("death", type(g))
    #print("nothing", nothingProbability)
    if isinstance(f, np.ndarray):
        f = f[0]
    if isinstance(g, np.ndarray):
        g = g[0]
    if isinstance(nothingProbability, np.ndarray):
        nothingProbability = nothingProbability[0]
    probabilities = np.array([f, g, nothingProbability], dtype=float).flatten()
    #print("probability", probabilities)
    # print(probabilities.ndim)
    cell_array = np.random.choice(np.array([2, 0, 1]).flatten(), size=(
        1, cell_num), replace=True, p=probabilities)
    #print("cell aray", cell_array)
# Create a list to store the randomly selected values
#     cell_array = []

    C = np.sum(np.array(cell_array))
    return C


def A_activate_T(a, b, K, h, c4, c_4, ni, t_treat, t, delta_t, T, A, vol_flag, time_flag, Ta, Tb, j=None, multiplier = 1):
    # working as expected
    m = 0
    newTa = Ta
    newTb = Tb
    if vol_flag == 0 or time_flag == 0:
        pass
    else:
        if abs(t - t_treat) < delta_t / 2:
            # c4 is anti-CTLA4 concentration for each injection
            # c_4 is anti CTLA4 concentration as function of time
            # increment c_4 by c4 if treatment occurs at timestep
            # print('treatment')
            #print('time', t)
            c_4 = c_4 - ni * c_4 * delta_t + c4
            m = 1
        else:
            c_4 = c_4 - ni * c_4 * delta_t
        # if c_4 > 2.0:
            #print('current c4', c_4, 'at time', t)

    if c4 == 0:
        c_4 = 0
    T_ac = a * T * A            # active
    T_in = b / (1 + multiplier*c_4) * T * A  # inactive
# check if T or A become negative
    # T(t+1) < 0, K is initial count of T
    T0_flag = T + delta_t * (- 1*T_ac - T_in + h) < 0
    A0_flag = A + delta_t * (- 1*T_ac - T_in) < 0  # A(t+1) < 0

    if T0_flag or A0_flag:
        # if any of them are negative
        if T0_flag:
            #print('T0 flag')
            delta_t_1 = -1*T / (-1 * T_ac - T_in + h)  # T = 0
            T_1 = 0
            A_1 = max(0, A + delta_t_1 * (-1 * T_ac - T_in))

            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
        elif A0_flag:
            #print('A neg')
            delta_t_1 = -1*A / (- 1*T_ac - 1 * T_in)  # A = 0
            A_1 = 0
            #print('A no treatment', A + delta_t * (-1* T_ac - b*T*A))
            #print('A treated', A + delta_t * (-1* T_ac - b/(1+c_4)*T*A))
            T_1 = max(0, T + delta_t_1 * (-1 * T_ac - T_in + h))
            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
        else:
            #print('else')
            delta_t_2 = -1*A / (- 1*T_ac - T_in)  # A = 0
            delta_t_3 = -1*T / (-1 * T_ac - T_in + h)  # T = 0
            delta_t_1 = min(delta_t_2, delta_t_3)
            A_1 = 0
            T_1 = 0
            newTa = Ta + delta_t_1 * T_ac
            newTb = Tb + delta_t_1 * T_in
            delta_t_1 = delta_t_3

        delta_t_2 = delta_t - delta_t_1
        T = min(K, T_1 + delta_t_2 * h)
        A = A_1
    else:
        #print('fine')
        T = min(K, T + delta_t * (-1 * T_ac - T_in + h))
        #print('K', K)
        #print(T + delta_t * (-1 * T_ac - T_in + h))
        A = max(0, A + delta_t * (-1 * T_ac - T_in))

        newTa = Ta + delta_t * T_ac
        newTb = Tb + delta_t * T_in
        # print('Tb treat', newTb)
        # print('Tb no treat', Tb + delta_t*b*A*T)
        # print('Tb diff', newTb - (Tb + delta_t*b*A*T))
    return T, A, newTa, newTb, m, c_4


def cropArray(array, j):
    return array[:, 0:j]


def radioimmuno_response_model(param, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov):
    # Extract all the parameters
    C_0 = param[0]
    lambda_1 = param[1]
    alpha_C = param[2]
    beta_C = param[3]
    phi = param[4]
    tau_dead_1 = param[5]
    tau_dead_2 = param[6]
    vol_C = param[7]
    A_0 = param[8]
    rho = param[9]
    psi = param[10]
    sigma = param[11]
    tau_1 = param[12]
    Ta_tum_0 = param[13]
    alpha_T = param[14]
    beta_T = param[15]
    tau_2 = param[16]
    eta = param[17]
    T_lym_0 = param[18]
    h = param[19]
    iota = param[20]
    vol_T = param[21]
    c4 = param[22]
    r = param[23]
    ni = param[24]
    a = param[25]
    b = (r - 1) * a
    p = param[26]
    q = param[27]
    s = param[28]
    recovery = param[29]
    lambda_2 = param[30]
    beta_2 = param[31]
    p1 = param[32]
    mi = param[33]
    c4_list = []
    if len(param) == 35:
        multiplier = param[34]
    else:
        multiplier = 1
    #print('c4', c4)
    # Create discrete time array
    time = np.arange(0, t_f1 + t_f2 + 1 + delta_t, delta_t)
    m = len(time)

    # Select LQL or modified LQ
    if LQL == 1 and D[0] > 0:
        beta_C = min(beta_C, 2 * beta_C * (beta_2 *
                     D[0] - 1 + np.exp(-1*beta_2 * D[0])) / beta_2**2)
    #else:
        
        #beta_C = beta_C * (1 + beta_2 * np.sqrt(D[0]))

    # Activate vascular death if activate_vd is 1 and first dose > 15Gy
    if activate_vd == 1 and D[0] > 15:
        vascular_death = 0
    else:
        vascular_death = 1

    # Initialise variables
    C = np.zeros((1, m))       # Tumor cells (tumor))
    #C_no_treat = np.zeros((1,m))
    A = np.zeros((1, m))       # Antigens (activation zone))
    #A_no_treat = np.zeros((1,m))
    Ta_tum = np.zeros((1, m))  # Activated T-cells (tumor))
    #Ta_tum_no_treat = np.zeros((1, m))
    # T-cell available to be activated (activation zone))
    T_lym = np.zeros((1, m))
    #T_lym_no_treat = np.zeros((1,m))
    Ta_lym = np.zeros((1, m))  # Activated T-cells (activation zone))
    #Ta_lym_no_treat = np.zeros((1,m))
    Tb_lym = np.zeros((1, m))  # Inactivated T-cells (activation zone))
    #Tb_lym_no_treat = np.zeros((1,m))
    vol = np.zeros((1, m))     # Tumor volume

    # Delay index
    del_1 = max(0, round(tau_1/delta_t) - 1)
    del_2 = max(0, round(tau_2/delta_t) - 1)

    d = len(D)
    C_dead = np.zeros((1, d))  # Damaged tumor cells at each RT time
    # Alive damaged tumor cells evolution for each RT dose
    M = np.zeros((d, m))
    # Total alive damaged tumor cells at each time step
    C_dam = np.zeros((1, m))
    C_tot = C.copy()             # Total alive tumor cells
    #C_tot_no_treat = C
    # Surviving fraction with LQ model parameters
    SF_C = np.zeros((1, d))    # Tumor cells surviving fraction
    SF_T = np.zeros((1, d))    # Lymphocytes surviving fraction

    # Variables initial value
    C[0] = C_0
    # C_no_treat[0] = C_0
    A[0] = A_0
    # A_no_treat[0] = A_0
    Ta_tum[0] = Ta_tum_0
    # Ta_tum_no_treat[0] = Ta_tum_0
    T_lym[0] = T_lym_0
    # T_lym_no_treat[0] = T_lym_0
    C_tot[0] = C_0
    # C_tot_no_treat[0] = C_0
    # Free behavior in time or volume
    free_flag = free[0]   # 1 for free behavior, 0 otherwise
    free_op = free[1]     # 1 for time, 0 for volume

    t_eq = -1
    vol_flag = 1          # 1 if initial volume was achieved, 0 otherwise
    time_flag = 1         # 1 if initial time was achieved, 0 otherwise

    # if free_flag == 1:
    # if free_op == 0:
    #vol_in = free[2]
    #vol_flag = 0
    # else:
    #t_in = free[2]
    #time_flag = 0
    # else:
    #m = t_f2/delta_t + 1

    p_1 = 0
    c_4 = 0
    tf_id = max(1, round(t_f2 / delta_t))
    k = 0                 # Radiation vector index
    ind_c4 = 0            # c4 treatment vector index
    ind_p1 = 0            # p1 treatment vector index
    # print(m)
    # print(del_1)
    # print(del_2)
# initialise all the arrays to have the initial conditions
    for i in range(max(del_1, del_2) + 1):
        C[:, i] = C_0
        A[:, i] = A_0
        Ta_tum[:, i] = Ta_tum_0
        T_lym[:, i] = T_lym_0
        #Ta_lym[:,i] = 0
        Tb_lym[:, i] = 0
        C_tot[:, i] = C_0
        # C_no_treat[:, i] = C_0
        # A_no_treat[:, i] = A_0
        # Ta_tum_no_treat[:, i] = Ta_tum_0
        # T_lym_no_treat[:, i] = T_lym_0
        # Tb_lym_no_treat[:, i] = 0
        # C_tot_no_treat[:, i] = C_0
        vol[:, i] = C_0*vol_C
        c4_list.append(0)

    # Algorithm
    j = i
    im_death = Ta_lym
    im_death_no_treat = Ta_lym
    #print("max possible j", m-1)
    # print(C.shape)
    while j <= m-1:
        # if j>=1820:
        #     print(j)
        #     print("C as per start of main function", C[:,j])
        #     print("C total", C_tot[:,j])
        #     print("Tb_lym", Tb_lym[:,j])
        # growth rate of C due to natural tumor growth
        prol = growth(lambda_1, C_tot[:, j], lambda_2)[0]
        #print(prol)
        p_11 = p_1
        ind_p11 = ind_p1
        storeTalym = (Ta_lym[:, j][0],)
        #storeTalym_no_treat = (Ta_lym_no_treat[:, j][0],)
        # if j>650:
        #print("C", C[:,j])
        #print("Store Ta3", storeTalym)
        #print("Tb_lym", Tb_lym[:,j])
        # p1_flag = 0
        # if vol_flag == 0 or time_flag == 0:
        #     pass
        # else:
        #     if abs(time[j+1] - t_treat_p1[ind_p1]) < delta_t / 2:
        #         p_1 = p_1 - mi * p_1 * delta_t + p1
        #         p1_flag = 1
        #     else:
        #         p_1 = p_1 - mi * p_1 * delta_t
        # im_death[:,j] = p * (1 + p_1) * (Ta_tum[:,j] / C_tot[:,j]) ** q / (s + (Ta_tum[:,j] / C_tot[:,j]) ** q)
        # if np.isnan(im_death[:,j]):
        #   #stop division by 0 - if C tot is 0, C is 0 and so there is no change because of immune cell death
        #     im_death[:,j] = 0
        # print(t_treat_p1)
        # print(C_tot.shape)
        # print(Ta_tum.shape)
        # print(im_death.shape)
        # print(time)
        # print(t_treat_p1)
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('start', C[:,j])
        [im_death[:, j], p1_flag, p_1] = immune_death_dePillis(C_tot[:, j], Ta_tum[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)  # This line modifies Ta_lym[:,j] somehow
        immune = (im_death[:, j][0],)
        # if j <=550:
        #     print('next call')
        #[im_death_no_treat[:,j], x,y] = immune_death_dePillis(C_tot_no_treat[:, j], Ta_tum_no_treat[:, j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j)
        
        #immune_no_treat = (im_death_no_treat[:, j][0],)
        # if j <= 550:
        #     print('immune cell death', immune)
        #     print('immune cell death no treat', immune_no_treat)
        #     print('diff', immune[0] - immune_no_treat[0])
        #     print('treat', p * (1 + p_1) * (Ta_tum[:,j] / C_tot[:,j]) ** q / (s + (Ta_tum[:,j] / C_tot[:,j]) ** q))
        #     print('no treat', p *  (1 + p_1) * (Ta_tum_no_treat[:,j] / C_tot_no_treat[:,j]) ** q / (s + (Ta_tum_no_treat[:,j] / C_tot_no_treat[:,j]) ** q))
        Ta_lym[:, j] = storeTalym[0]
        #Ta_lym_no_treat[:,j] = storeTalym_no_treat[0]
        # if j>650:
        #print("Tb_lym", Tb_lym[:,j])
        #print("store Ta", storeTalym)
        if p1_flag == 1:
            # if treatment was administered, increase the t treat p1 index by 1 (up to max of len(t_treat_p1) - 1 so no errors occur)
            ind_p1 = min(ind_p1 + 1, len(t_treat_p1) - 1)
        # Markov
        if C[:, j] <= 15000 and use_Markov:
            #print('prol', prol)
            #print('immune', immune[0])
            newC = (markov_TCP_analysis(immune[0], prol, C[:, j][0], delta_t),)
            
        elif C[:, j] == 0:
            newC = (0,)
            newC_no_treat = (0,)
        else:
            newC = (max(0, (C[:, j] + delta_t * (prol - immune[0]) * C[:, j])[0]),)
            #print('before', newC[0])
            #newC_no_treat = (max(0, C_no_treat[:, j] + delta_t * (prol - immune[0]) * C_no_treat[:, j]),)
        T_lym[:, j+1], A[:, j+1], Ta_lym[:, j+1], Tb_lym[:, j+1], c4_flag, c_4 = A_activate_T(
            a, b, T_lym_0, h, c4, c_4, ni, t_treat_c4[ind_c4 - 1], time[j+1], delta_t, T_lym[:, j], A[:, j], vol_flag, time_flag, Ta_lym[:, j], Tb_lym[:, j], j, multiplier)

        c4_list.append(c_4)
        #T_lym_no_treat[:, j+1], A_no_treat[:, j+1], Ta_lym_no_treat[:, j+1], Tb_lym_no_treat[:, j+1], *_ = A_activate_T(
            #a, b, T_lym_0, h, 0, 0, ni, t_treat_c4[ind_c4 - 1], time[j+1], delta_t, T_lym_no_treat[:, j], A_no_treat[:, j], vol_flag, time_flag, Ta_lym_no_treat[:, j], Tb_lym_no_treat[:, j], j)

        if c4_flag == 1:
            ind_c4 = min(ind_c4 + 1, len(t_treat_c4) - 1)
        # get the rate at which antigen is released by tumor cells, delayed due to delay between antigen release and t cell activation
        nat_rel = natural_release(rho, C_tot[:, (j+1) - del_1])
        #nat_rel_no_treat = natural_release(rho, C_tot_no_treat[:, (j+1) - del_1])
        # calc how many cells died in the timestep, delayed due to delay between antigen release and t cell activation
        dead_step = M[:, j-del_1] - M[:, j+1-del_1]
        dead_step[dead_step < 0] = 0  # clear negative differences to be 0

        # sum up for total of all cells that died in timestep due to all RT doses
        dead_step = np.sum(dead_step)

        RT_rel = RT_release(psi, dead_step)
        # exponential decay of antigen
        A_nat_out = A_natural_out(sigma, A[:, j])
        #A_nat_out_no_treat = A_natural_out(sigma, A_no_treat[:, j])
        # getting next value of A by using small change formula
        A[:, j+1] = A[:, j+1] + delta_t * (nat_rel + A_nat_out + RT_rel*phi)
        #A_no_treat[:, j+1] = A_no_treat[:, j+1] + delta_t * (nat_rel_no_treat + A_nat_out_no_treat) + RT_rel
        #print('A treatment', A[:, j+1])
        # if  A[:, j] + delta_t * (- 1*a*A[:, j] - b/(1+c_4)*T_lym[:, j-1]*A[:, j]) < 0:
        #     print('negative treat')
        # if A[:, j] + delta_t * (- 1*a*A[:, j] - b*T_lym[:, j-1]*A[:, j]) < 0:
        #     print('negative no treat')
        #     print('A no treat', delta_t*(nat_rel + A_natural_out(sigma, A[:, j])) + RT_rel)
        # else:
        #     print('A no treatment', A[:, j] + delta_t * (- 1*a*A[:, j] - b * T_lym[:, j]*A[:, j] + nat_rel + A_natural_out(sigma, A[:, j])) + RT_rel)
        # T cell
        # interaction between tumor cell and Ta cells
        T_out = immune_death_T(iota, C[:, j] + C_dam[:, j], Ta_tum[:, j])
        #T_out_no_treat = immune_death_T(iota, C_no_treat[:, j] + C_dam[:, j], Ta_tum_no_treat[:, j])
        
        #print('T lym diff', T_lym[:,j+1] - T_lym_no_treat[:,j+1])
        #print('A diff', A[:,j+1] - A_no_treat[:,j+1]) #as expected, no ctla4 decreases A
        # if j >= 1981 and j <=1983:
        # print(j)
        #print("T out", T_out)
        # exponential natural elimination of Ta
        T_nat_out = T_natural_out(eta, Ta_tum[:, j])
        #T_nat_out_no_treat = T_natural_out(eta, Ta_tum_no_treat[:, j])
        
        #Ta_tum[:,j+1] = Ta_tum[:,j] + vascular_death * Ta_lym[:,(j+1) - del_2] + delta_t * (T_out + T_nat_out )
        Ta_tum[:, j+1] = Ta_tum[:, j] + vascular_death * delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2] + delta_t * (T_out + T_nat_out)
        #Ta_tum_no_treat[:, j+1] = Ta_tum_no_treat[:, j] + vascular_death * delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2] + delta_t * (T_out_no_treat + T_nat_out_no_treat)
        #print('Ta tum diff', Ta_tum[:,j+1] - Ta_tum_no_treat[:,j+1])
        # if j <= 550:
        #     print(time[j+1])
        #     #print(del_2*delta_t)
        #     print('t out diff', T_out - T_out_no_treat)
        #     print('t nat out diff', T_nat_out - T_nat_out_no_treat)
        #     print('Ta tum increase', delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2])
        #     print('Ta tum decrease', delta_t * (T_out + T_nat_out))
        #     print('Ta tum step', vascular_death * delta_t * a * A[:, j - del_2]*T_lym[:, j - del_2] + delta_t * (T_out + T_nat_out))
        #     print('Ta tum increase no treat', delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2])
        #     print('Ta tum decrease no treat', delta_t * (T_out_no_treat + T_nat_out_no_treat))
        #     print('Ta tum step no treat', vascular_death * delta_t * a * A_no_treat[:, j - del_2]*T_lym_no_treat[:, j - del_2] + delta_t * (T_out_no_treat + T_nat_out_no_treat))
        #     print('A diff', A[:,j+1] - A_no_treat[:,j+1])
        #     print('Ta tum diff', Ta_tum[:,j+1] - Ta_tum_no_treat[:,j+1])
        #     print()
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
        #print("C", C[:,j])
        #print("Ta as per middle of main function", Ta_lym[:,j])
        #print("Ta as per middle of main function", Ta_lym[:,j+1])
        if (time[j+1] > t_eq and activate_vd == 1 and D[0] >= 15):
            vascular_death = min(1, recovery * (time[j+1 - t_eq]))

        # if vol_flag == 1 and time_flag == 1 and D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
            # print("RT")
            # print(j)
            # print(time[j])
            # calculaate survival fractions of cancer cells and T cells
            SF_C[:, k] = np.exp(-1 * alpha_C * D[k] - beta_C * D[k] ** 2)
            
            #print('SF', SF_C[:,k])
            #print('poisson', np.exp(-1*newC[0]*SF_C[:,k]))
            #print('C', C[:,j])
            SF_T[:, k] = np.exp(-1 * alpha_T * D[k] - beta_T * D[k] ** 2)
            # updates cancer cell count by killing off (1-SFC)*C of the cancer cells
           # print('before', newC[0])
            C_remain = newC[0] * SF_C[:,k][0]
            #C_remain = np.random.poisson(newC[0]*SF_C[:,k][0])
            C_dead[:, k] = newC[0] - C_remain
           #print(C[:,j])
            # print(SF_C[:,k])
            # print("before RT kill", C[:, j+1])
            # C[:,j+1] = C[:,j+1] - C_dead[:,k]
            #print('before treat', newC[0])
            newC = (C_remain,)
            #print('after treat', newC[0])
            
            #print('treat', newC[0])
            # print(C[:, j])
            # print(C[0][500:-1])
            # print(Ta_tum[:,j])
            # print("before RT kill", Ta_tum[:, j+1])
            Ta_tum[:, j+1] = Ta_tum[:, j+1] - (1 - SF_T[:, k]) * Ta_tum[:, j+1]
            # print(Ta_tum[:, j+1])
            # print(Ta_tum[0][500:-1])
            for ii in range(d):
              # C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin = tum_kinetic(
                    phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                im_death_d = immune_death_dePillis(
                    C_tot[:, j], Ta_tum[:, j], p, q, s, p1, p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                M[ii, j+1] = max(0, M[ii, j] + delta_t *
                                 (C_kin - im_death_d) * M[ii, j])

            M[k, j+1] = M[k, j+1] + C_dead[:, k]

        # The sum of the columns of M is the total damaged tumor cells that
        # are going to die in each time step
            C_dam_new = (np.sum(M[:, j+1]),)
            #print(C_dam_new)
            k = min(k + 1, len(t_rad) - 1)

        # elif vol_flag == 1 and time_flag == 1 and D[0] != 0:
        elif D[0] != 0:
            for ii in range(d):
              # C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin = tum_kinetic(
                    phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                im_death_d = immune_death_dePillis(
                    C_tot[:, j], Ta_tum[:, j], p, q, s, p1, p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                M[ii, j+1] = max(0, M[ii, j] + delta_t *
                                 (C_kin - im_death_d) * M[ii, j])
            # The sum of the columns of M is the total damaged tumor cells that
        # are going to die in each time step
            C_dam_new = (np.sum(M[:, j+1]),)
            #print(C_dam_new)
        # print(j)
        # print(Ta_tum[:,j+1])
        # get rid of negative values
        
        
        # if C_no_treat[:, j+1] < 0:
        #     C_no_treat[:, j+1] = 0
        # if A_no_treat[:, j+1] < 0:
        #     A_no_treat[:, j+1] = 0
        # if Ta_tum_no_treat[:, j+1] < 0:
        #     Ta_tum_no_treat[:, j+1] = 0
        # update total count of cancer cells (damaged and healthy)

        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
        #print("Ta as per later middle of main function", Ta_lym[:,j])
        #print("Ta as per later middle of main function", Ta_lym[:,j+1])
        if vol_flag != 1 and vol[j+1] >= vol_in:
            t_eq = time[j+1]
            t_rad = t_rad + t_eq
            t_treat_p1 = t_treat_p1 + t_eq
            t_treat_c4 = t_treat_c4 + t_eq
            vol_flag = 1
        elif time_flag != 1 and time[j+1] >= t_in:
            m = j + 1 + tf_id
            t_rad = np.array(t_rad) + t_in
            t_treat_p1 = np.array(t_treat_p1) + t_in
            t_treat_c4 = np.array(t_treat_c4) + t_in
            time_flag = 1
        
        # C_no_treat[:, j+1] = newC_no_treat[0]
        if newC[0] < 0.5:
            newC = (0,)
        if C_dam_new[0] < 0.5:
            C_dam_new = (0,)
        if A[:, j+1] < 0:
            A[:, j+1] = 0
        if Ta_tum[:, j+1] < 0:
            Ta_tum[:, j+1] = 0
        
        #print('before', newC[0])
        C_dam[:,j+1] = C_dam_new[0]
        
        #C_tot_new = (newC[0] + C_dam_new[0],)
        
        #print(C_tot_new[0])
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('C var', C[:,j+1])
       
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('C var', C[:,j+1])
            # C_tot_no_treat[:, j+1] = newC_no_treat[0] + C_dam[:, j+1]
            
            # calculate tumour volume at the time step by V = C*VC + Ta*VT
        
        
        #print('c4', c_4)
        #print('A treatment', A[:, j+1])
        #print('A no treatment', A_no_treat[:, j+1])
        
        
        
        
        
        
        C[:, j+1] = newC[0]
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('after assign', newC[0])
        #     print('C after assign', C[:,j+1])
        #print('before', C[:,j+1])
       
        C_tot[:,j+1] = newC[0] + C_dam_new[0]
        #C[:, j+1] = newC[0]
        # if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        #     print('after C tot', newC[0])
        #     print('C after C tot', C[:,j+1])
        #print('after', C[:,j+1])
        vol[:, j+1] = tumor_volume(C_tot[:,j+1], Ta_tum[:, j+1], vol_C, vol_T)
        j = j + 1
        #print('A treatment', A[:, j])
        #print('A no treatment', A_no_treat[:, j])
        #print()
        if time[j-1] > t_f1 and vol_flag == 0:
            time = time[0:j]

            vol = cropArray(vol, j)
            C_tot = cropArray(C_tot, j)
            C = cropArray(C, j)
            C_dam = cropArray(C_dam, j)
            A = cropArray(A, j)
            Ta_tum = cropArray(Ta_tum, j)
            T_lym = cropArray(T_lym, j)
            Ta_lym = cropArray(Ta_lym, j)
            Tb_lym = cropArray(Tb_lym, j)

            #print('last time', time[-1])
            # return vol, t_eq, time, C_tot, C, C_dam, A, Ta_tum, T_lym, Ta_lym, Tb_lym
        if time[j-1] > t_eq + t_f2 and vol_flag == 1:
            time = time[0:j]
            vol = cropArray(vol, j)
            C_tot = cropArray(C_tot, j)
            C = cropArray(C, j)
            C_dam = cropArray(C_dam, j)
            A = cropArray(A, j)
            #A_no_treat = cropArray(A_no_treat, j)
            Ta_tum = cropArray(Ta_tum, j)
            T_lym = cropArray(T_lym, j)
            Ta_lym = cropArray(Ta_lym, j)
            Tb_lym = cropArray(Tb_lym, j)
            c4_list = c4_list[0:j]
            #print('A treatment', A[:,j-1])
            #print('b', b)
            #print('b treat', b/(1+c_4))
            # if A[:,j-2] + delta_t * (- 1*a*A[:,j-2] - b*T_lym[:,j-2]*A[:,j-2]) < 0:
            #print('A no treat', delta_t*( nat_rel + A_natural_out(sigma, A[:,j-2])) + RT_rel)
            # else:
            #print('A no treatment', A[:,j-2]+ delta_t * (- 1*a*A[:,j-2] - b*T_lym[:,j-2]*A[:,j-2] + nat_rel + A_natural_out(sigma, A[:,j-2])) + RT_rel)
            #print('last time', time[-1])
            return vol, t_eq, time, C_tot, C, C_dam, A, Ta_tum, T_lym, Ta_lym, Tb_lym, c4_list
