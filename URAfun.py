import numpy as np
def efun(x,mic_pos_local,distvector_local):
    #x[0]: alpha x[1]: beta
    Pline = np.array([[np.cos(x[0]),np.sin(x[0]),0]])
    #print('Pline',np.shape(Pline))
    projected_mic_distance = np.matmul(Pline,mic_pos_local)
    #print('projected_mic_distance',np.shape(projected_mic_distance))
    #print('distvector_local',np.shape(distvector_local))
    expected_time_difference = np.cos(x[1])*projected_mic_distance
    error = np.inner((expected_time_difference-distvector_local),(expected_time_difference-distvector_local))
    return error