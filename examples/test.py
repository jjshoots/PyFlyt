import numpy as np
import scipy as sp
from scipy.optimize import linear_sum_assignment

start_locations = np.expand_dims(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), axis=0)
drone_locations = np.expand_dims(np.array([9, 8, 7, 6, 5, 4, 3, 2, 1]), axis=1)

cost = (start_locations - drone_locations) ** 2

_, reassignment = linear_sum_assignment(cost)

drone_locations = drone_locations[reassignment]
print(drone_locations)