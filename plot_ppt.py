
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from utiles import *
from scipy.stats import gaussian_kde
from utiles import *
from data_utiles import *
from default_args import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def illustration():
    args = toy_args()
    args['data_type'] = 1
    args['data_size'] = 100
    args['hidden_dim'] = 32
    args = modify_args(args)
    data = Toy_Dataset(args)
    data.device = DEVICE
    model_name_list=['simple']
    for model_name in model_name_list:
        train_all(data, args, model_name)
    plot_mapping_illustration(data, args, 'simple')
illustration()
print(1/0)




# run_flow()
# print(1/0)




# Define the vector field v(x, t)
def v(x, t):
    # xt = np.sin(x+np.pi/4) * 4 + t
    xt = np.sin(x) * 4
    return xt

# Parameters
num_timesteps = 1000
num_evaluation = 1000
num_simulations = 1000
dt = 1.0 / num_timesteps
ub = 5
lb = -5
x_range = np.linspace(lb, ub, num_evaluation)
t_range = np.linspace(0, 1, num_timesteps)
z_range = np.linspace(lb, ub, 8)

# Initialize the PDF
pdf = np.zeros((num_timesteps, len(x_range)))
traj = np.zeros((num_timesteps, len(z_range)))
simu = np.zeros((num_timesteps, num_simulations))
pdf[0, :] = (1/np.sqrt(2*np.pi)) * np.exp(-x_range**2/2)
traj[0, :] = z_range
simu[0,:] = np.random.randn(num_simulations)

# Simulate the random variable evolution
for t_idx in range(1, num_timesteps):
    # dW = np.random.normal(0, np.sqrt(dt), n_simulations)
    simu[t_idx,:] = simu[t_idx-1,:] + v(simu[t_idx-1,:], t_range[t_idx-1]) * dt #+ sigma * dW
# Compute the PDF for each time step on a regular grid
for t in range(1, num_timesteps):
    kde = gaussian_kde(simu[t,:], )
    pdf[t,:] = kde(x_range)
# Solve the ODE using the forward Euler method
for t_idx in range(1, num_timesteps):
    cur_z = traj[t_idx-1,:]
    traj[t_idx,:] = cur_z + dt * v(cur_z, t_range[t_idx-1])

plot_flow_illustration(pdf, traj, x_range, z_range, t_range, lb, ub)

















