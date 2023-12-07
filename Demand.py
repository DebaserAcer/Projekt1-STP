import numpy as np
import pandas as pd




def DestinationChoice(Oi, t_ij, c_ij, Dj):
    VoT = 116 / 60
    beta_c = -0.005 * VoT 
    beta_t = -0.005
    beta_a = 1

    v_ij = beta_c *c_ij +  beta_t * t_ij + beta_a * np.log(Dj)

    exp_v = np.exp(v_ij)

    P = exp_v / exp_v.sum(axis=1, keepdims=True)
    Tij = Oi[:,np.newaxis] * P
  
    return Tij

def DestinationLogsum(t_ij, c_ij, Dj):
    VoT = 116 / 60
    beta_c = -0.005 * VoT 
    beta_t = -0.005
    beta_a = 1

    v_ij = beta_c *c_ij +  beta_t * t_ij + beta_a * np.log(Dj)

    exp_v = np.exp(v_ij)
    logsum = np.log(exp_v.sum(axis=1, keepdims=True))
  
    return logsum.flatten()

def ModeChoice(dest_logsum):
    v = 0.05 * dest_logsum
    P = np.exp(v)/(1 + np.exp(v))

    return P

def ModeLogsum(dest_logsum):
    v = 0.05 * dest_logsum
    logsum = np.log(1 + np.exp(v))

    return logsum.flatten()

def TripChoice(mode_logsum):
    v = 0.8 * mode_logsum
    P = np.exp(v)/(1 + np.exp(v))
    return P

def Demand(t_ij: np.array, c_ij: np.array, landuse: pd.DataFrame, workerscolumn, jobscolumn):
    Ni = landuse[workerscolumn].to_numpy()
    Ej = landuse[jobscolumn].to_numpy()


    # Trip production
    lsm_car = DestinationLogsum(t_ij, c_ij, Ej)
    lsm_mode = ModeLogsum(lsm_car)
    P_trip = TripChoice(lsm_mode)
    P_car = ModeChoice(lsm_car)


    Oi = Ni * P_trip.T * P_car.T


    # Quick balancing of totals
    #Oi = np.round(Oi * sum(Dj)/sum(Oi),1)

    T_ij = DestinationChoice(Oi, t_ij, c_ij, Ej)
    return T_ij

