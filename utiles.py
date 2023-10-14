import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
from net_utiles import *




def train_all(data, args, model_type):
    # Unpack arguments
    input_dim = data.x_train.shape[-1]
    output_dim = data.y_train.shape[1]
    data_dim = data.x_train.shape[0]
    # Set training parameters
    instance = args['instance'] 
    num_epochs = args['num_iteration']
    batch_dim = args['batch_dim']
    hidden_dim = args['hidden_dim']
    num_layers = args['num_layer']
    output_act = args['output_act']
    network = args['network']
    pred_type = 'edge' if args['data_set'] == 'tsp' else 'node'
    # Additional parameters for different models
    num_cluster = args['num_cluster']
    latent_dim = args['latent_dim']
    time_step = args['time_step']
    output_norm = args['output_norm']
    noise_type = 'gaussian'#args['noise_type']
    if model_type == 'simple':
        model = Simple_NN(network, input_dim, output_dim, hidden_dim, num_layers, output_act, pred_type).to(data.device)
    elif model_type in ['cluster','hindsight']:
        model = GMM(network, input_dim, output_dim, hidden_dim, num_layers, num_cluster, output_act, pred_type).to(data.device)
    elif model_type == 'vae':
        model = VAE(network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'gan':
        model = GAN(network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'wgan':
        model = WGAN(network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type).to(data.device)
    elif model_type == 'diffusion':
        model = DM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type in ['gaussian', 'rectified', 'interpolation', 'conditional']:
        model = FM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type == 'potential':
        model = AM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type == 'consistency_training':
        model = CM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
        vector_model = torch.load(f'models/{instance}/rectified_{network}.pth')
        vector_model.eval()
    elif model_type == 'consistency_distillation':
        model = CD(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
        vector_model = torch.load(f'models/{instance}/rectified_{network}.pth')
        vector_model.eval()
    else:
        raise NotImplementedError
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['learning_rate_decay'][0], gamma=args['learning_rate_decay'][1])
    # Train the model
    loss_record = []
    for epoch in range(num_epochs):
        batch_indices = np.random.choice(data_dim, batch_dim)
        x_batch = data.x_train[batch_indices].to(data.device)
        y_batch = data.y_train[batch_indices].to(data.device)
        t_batch = torch.rand([batch_dim, 1]).to(data.device)
        if noise_type == 'gaussian':
            z_batch = torch.randn_like(y_batch).to(data.device)
        elif noise_type == 'uniform_fixed':
            z_batch = torch.rand_like(y_batch).to(data.device) * 2 - 1
        else:
            NotImplementedError
        optimizer.zero_grad()
        if model_type == 'simple':
            y_pred = model(x_batch)
            loss = model.loss(y_pred, y_batch)
        elif model_type == 'cluster':
            y_pred = model(x_batch)
            loss = model.loss(x_batch, y_pred, y_batch)
        elif model_type == 'hindsight':
            y_pred = model(x_batch)
            loss = model.hindsight_loss(x_batch, y_pred, y_batch)
        elif model_type == 'vae':
            y_pred, mean, logvar = model.encoder_decode(x_batch)
            loss = model.loss(y_pred, y_batch, mean, logvar)
        elif model_type in ['gan', 'wgan']:
            y_pred =  model(x_batch, z_batch)
            loss = model.loss_d(x_batch, y_batch, y_pred)
            if epoch % args['update_generator_freq'] == 0:
                loss += model.loss_g(x_batch, y_pred)
        elif model_type == 'diffusion':
            noise_pred = model.predict_noise(x_batch, y_batch, t_batch, z_batch)
            loss = model.loss(z_batch, noise_pred)
        elif model_type in ['gaussian', 'rectified', 'interpolation', 'conditional']:
            z_batch = torch.randn_like(y_batch).to(data.device)
            yt, vec_target = model.flow_forward(y_batch, t_batch, z_batch, model_type)
            vec_pred = model.predict_vec(x_batch, yt, t_batch)
            loss = model.loss(y_batch, z_batch, vec_pred, vec_target, model_type)
        elif model_type == 'potential':
            loss = model.loss(x_batch, y_batch, z_batch, t_batch)
        elif model_type in ['consistency_training']:
            N = math.ceil(math.sqrt((epoch * (1000**2 - 4) / num_epochs) + 4) - 1) + 1
            boundaries = model.kerras_boundaries(1.0, 0.002, N, 1).to(device=data.device)
            t = torch.randint(0, N - 1, (x_batch.shape[0], 1), device=data.device)
            t_1 = boundaries[t+1]
            t_2 = boundaries[t]
            loss = model.loss(x_batch, y_batch, z_batch, t_1, t_2, data, vector_model)
        elif model_type == "consistency_distillation":
            forward_step = 10
            N = math.ceil(1000*(epoch/num_epochs) + 4) + forward_step
            boundaries = torch.linspace(0,1-1e-3,N).to(device=data.device)
            # model.kerras_boundaries(0.5, 0.001, N, 1).to(device=data.device)
            t = torch.randint(0, N - forward_step, (x_batch.shape[0], 1), device=data.device)
            t_1 = boundaries[t]
            # t_2 = boundaries[t+1]
            # step_size = 0.001#max(1/(epoch+10), 0.001)
            # t_1 = t_batch
            # t_2 = torch.clamp(t_batch + step_size, 0,1) 
            loss = model.loss(x_batch, y_batch, z_batch, t_1, 1/N, forward_step, data, vector_model)
        else:
            raise NotImplementedError
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_record.append(loss.item())
        # if model_type in ['consistency_training', 'consistency_distillation']:
        #     with torch.no_grad():
        #         mu = 0.1#math.exp(2*math.log(0.5) / N)
        #         for p, ema_p in zip(model.model.parameters(), model.target_model.parameters()):
        #             ema_p.mul_(mu).add_(p, alpha=1 - mu)
        if epoch % args['test_freq'] == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
    instance = args['instance']
    torch.save(model, f'models/{instance}/{model_type}_{network}.pth')
    # plt.plot(np.arange(num_epochs), loss_record)
    # plt.show()


def test_single(x_test, args, model_name, sample_num=1):
    instance = args['instance'] 
    inf_step = args['inf_step']
    network = args['network']
    test_dim = x_test.shape[0]
    model = torch.load(f'models/{instance}/{model_name}_{network}.pth')
    model.eval()
    z_test = 0
    with torch.no_grad():
        if model_name == 'simple':
            y_pred = model(x_test)
        elif model_name in ['cluster', 'hindsight']:
            y_pred = model(x_test).view(x_test.shape[0], -1, args['num_cluster'])
            # if sample_num == 1:
            #     index = torch.LongTensor(np.random.choice(np.arange(args['num_cluster']), [y_pred.shape[0],y_pred.shape[1],1])).to(x_test.device)
            #     y_pred = torch.gather(y_pred, 2, index).view(x_test.shape[0],-1)
        elif model_name in ['gan', 'wgan', 'vae']:
            x_test = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test.shape[0], args['latent_dim']]).to(x_test.device)
            y_pred = model(x_test, z_test)
        elif model_name in ['rectified', 'gaussian', 'conditional', 'interpolation']:
            x_test = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test.shape[0], args['output_dim']]).to(x_test.device)
            y_pred = model.flow_backward(x_test, z_test, 1/inf_step, method = args['ode_solver'])
        elif model_name == 'diffusion':
            x_test = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            z_test = torch.randn(size=[x_test.shape[0], args['output_dim']]).to(x_test.device)
            y_pred = model.diffusion_backward(x_test, z_test, inf_step, eta=args['eta'])
        elif model_name in ['potential']:
            with torch.enable_grad():
                x_test = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
                y_pred = model.flow_backward(x_test, 1/inf_step, method = args['ode_solver'])
        elif model_name in ['consistency_training', 'consistency_distillation']:
            x_test = torch.repeat_interleave(x_test, repeats=sample_num, dim=0)
            y_pred = model.sampling(x_test, inf_step=1)
        else:
            NotImplementedError
    if sample_num>1 and model_name not in ['cluster', 'hindsight']:
        y_pred = y_pred.view(test_dim, y_pred.shape[-1], -1) ##  batch * output_dim * sample
    return y_pred


def plot_single(data, args, model_name):
    instance = args['instance'] 
    plt.figure(figsize=[7,6])
    plt.xlabel('Input parameter', fontsize=16)
    plt.ylabel('Output solution', fontsize=16)
    if model_name == 'data':
        plt.scatter(data.x, data.y, marker='.', c='forestgreen', alpha=0.5, s=20)
        plt.title(f'Data distribution', fontsize=20)
    else:
        # plt.scatter(data.xt, data.yt, marker='*', c='turquoise', alpha=0.5)
        for i in range(np.shape(data.yt)[1]):
            yt = data.yt[:,i]
            if i==0:
                plt.plot(data.xt, yt, c='C0', alpha=0.7, linewidth=2.5)
            else:
                plt.plot(data.xt, yt, c='C0', alpha=0.7, linewidth=2.5, label='_nolegend_')

        x_test = torch.linspace(-1,1, 150).to(data.device).view(-1,1)#torch.rand(size=[200, 1]).to(data.device) * 2 - 1
        y_pred = test_single(x_test, args, model_name, sample_num=6).cpu().numpy()
        x_test = x_test.cpu().numpy()
        if model_name in ['simple', 'hindsight', 'cluster']:
            for k in range(y_pred.shape[2]):
                plt.plot(x_test, y_pred[:,:,k], c='C2', alpha=0.7, linestyle='--', linewidth = 2.5, label = 'NN approximated mapping')
        elif model_name in ['gan', 'difussion', 'rectified']:
            for k in range(y_pred.shape[2]):
                plt.scatter(x_test, y_pred[:,:,k], c='C1', s=15, alpha=0.7, label = 'NN approximated distribution')
        plt.legend(['Target mappings', 'Model output'], fontsize=14, loc='best')
        plt.title(f'{model_name}', fontsize=20)
        # np.save(f'results/{instance}/{model_name}_test_data.npy', [x_test, y_pred])
    plt.tight_layout()  
    plt.savefig(f'results/{instance}/{model_name}.png', dpi=300)
    # plt.show()
    plt.close()

import matplotlib.colors as mcolors
def plot_all(data, args, model_name_list):
    instance = args['instance'] 
    model_title = ['NN', 'Hindsight', 'Cluster', 'GAN', 'Diffusion', 'RectFlow']
    fig = plt.figure(figsize=[4*len(model_name_list)//2, 3.3*2])
    for k, model_name in enumerate(model_name_list):
        plt.subplot(2, len(model_name_list)//2, k+1)        
        if k in [0,3]:
            plt.ylabel('Output solutions', fontsize=16)
        else:
            plt.yticks([])
        if k<3:
            plt.xticks([])
        else:
            plt.xlabel('Input parameter', fontsize=16)
        # if k!=1:
        #     plt.xticks([-1,0,1])

        ## plot target mappings
        for i in range(np.shape(data.yt)[1]):
            yt = data.yt[:,i]
            if i==0:
                plt.plot(data.xt, yt, c='C0', alpha=0.9, linewidth=1.5, label='Target mappings')
            else:
                plt.plot(data.xt, yt, c='C0', alpha=0.9, linewidth=1.5)
        # x_test, y_pred = np.load(f'results/{instance}/{model_name}_test_data.npy', allow_pickle=True)

        x_test = torch.linspace(-1,1, 100).to(data.device).view(-1,1)#torch.rand(size=[200, 1]).to(data.device) * 2 - 1
        y_pred, z_sample = test_single(x_test, args, model_name, sample_num=10)
        x_test = x_test.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        if model_name in ['simple', 'hindsight', 'cluster']:
            # plt.plot(x_test, y_pred[:,:,0], c='C2', alpha=0.9, linestyle='--', linewidth = 2.5, label = 'NN mapping')
            for i in range(y_pred.shape[2]):
                if i==0:
                    plt.plot(x_test, y_pred[:,:,i], c='C2', alpha=0.9, linestyle='--', linewidth = 2.5, label = 'NN approximated mappings')
                else:
                    plt.plot(x_test, y_pred[:,:,i], c='C2', alpha=0.9, linestyle='--', linewidth = 2.5)
        elif model_name in ['gan', 'diffusion', 'rectified']:
            z_sample = z_sample.cpu().numpy()
            x_test = np.concatenate([x_test]*y_pred.shape[2], axis=1)
            x_test = np.reshape(x_test, [-1, 1])
            y_pred = np.reshape(y_pred, [-1, 1])

            plt.scatter(x_test, y_pred, marker='o', s=15, alpha=0.5,  
                        c=np.exp(-np.linalg.norm(z_sample, axis=1)), 
                        norm=mcolors.Normalize(vmin=0, vmax=1),
                         label='NN generated solutions', zorder=2.5)

        # plt.legend(['Target mappings', 'Model output'], fontsize=14, loc= 'best')
        # plt.scatter(x_test, y_pred, marker='o', s=15, alpha=0.99, 
        #             c=np.random.uniform(0.7,0.9, x_test.shape[0]), 
        #             cmap='GnBu', norm=mcolors.Normalize(vmin=0, vmax=1),
        #             label='Model predictions', zorder=2.5)
        plt.title(f'{model_title[k]}', fontsize=18)
    lines, lables = fig.axes[0].get_legend_handles_labels()
    lines1, labels1 = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines+[lines1[-1]], lables+[labels1[-1]], fontsize=14, ncol=3, bbox_to_anchor=[0.94,0.59])
    plt.tight_layout()  
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'results/{instance}/model.png', dpi=350)
    plt.show()
    plt.close()


def plot_mapping_illustration(data, args, model_name):
    instance = args['instance']
    data_dim = data.x.shape[0]//2
    plt.figure(figsize=[5.6,4])
    plt.xlabel('Input parameter', fontsize=14)
    plt.ylabel('Output solutions', fontsize=14)
    for i in range(np.shape(data.yt)[1]):
        yt = data.yt[:,i]
        if i==0:
            plt.plot(data.xt, yt, alpha=0.9, c='C0', linewidth=2.0, label='Target mappings')
        else:
            plt.plot(data.xt, yt, alpha=0.9, c='C0', linewidth=2.0, label='_nolegend_')
    x_test = torch.rand(size=[10000, 1]).to(data.device) * 2 - 1
    x_test,_ = torch.sort(x_test, 0)
    y_pred = test_single(x_test, args, model_name).cpu().numpy()
    x_test = x_test.cpu().numpy()
    plt.plot(x_test, y_pred, alpha=0.9, c='C2', linewidth=2.5, linestyle='--',label='NN\'s mapping')
    # plt.scatter(data.x, data.y,  marker='o', alpha=0.7, s=25 , label='Training dataset', zorder=2.5)
    plt.scatter(data.x, data.y, marker='o', s=15, alpha=0.9, 
                c=np.random.uniform(0.2,0.3, data.x.shape[0]), 
                cmap='Oranges', norm=mcolors.Normalize(vmin=0, vmax=1),
                label='Training dataset', zorder=2.5)
    plt.legend(fontsize=12)
    plt.tight_layout()  
    plt.savefig(f'results/{instance}/{model_name}_illustration.png', dpi=500)
    plt.show()
    plt.close()
    # np.save(f'results/{instance}/{model_name}_test_data.npy', [x_test, y_pred])


def plot_flow_illustration(pdf, traj, x_range, z_range, t_range, lb, ub, condition='toy'):
    # Create a GridSpec object with 1 row and 3 columns
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[1, 3, 1], wspace=0.1)
    max_pdf = np.max(pdf)
    cmap = plt.get_cmap('viridis')
    # norm = Normalize(vmin=0, vmax=max_pdf)

    # Create a new figure
    fig = plt.figure(figsize=(6, 3))

    # Add subplots to the figure using cells from the GridSpec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot the data in each subplot
    # ax1.scatter(pdf[0], x_range, c=pdf[0]/max_pdf, marker='o', s=7, cmap=cmap)
    points = np.array([pdf[0], x_range]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,  linewidth=2)
    lc.set_array(pdf[0])
    ax1.add_collection(lc)
    ax1.set_title('Initial PDF', fontsize=14)
    ax1.set_xlabel('Density', fontsize=13)
    ax1.set_ylabel('Random Variable (x)', fontsize=13)
    ax1.set_ylim([lb,ub])
    ax1.set_xlim([-0.01,max(pdf[0])+0.01])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')

    # ax3.scatter(pdf[-1], x_range, c=pdf[-1]/max_pdf, marker='o', s=7, cmap=cmap)
    points = np.array([pdf[-1], x_range]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,  linewidth=2)
    lc.set_array(pdf[-1])
    ax3.add_collection(lc)
    ax3.set_title('Terminal PDF', fontsize=14)
    ax3.set_xlabel('Density', fontsize=13)
    ax3.set_ylim([lb,ub])
    ax3.set_xlim([-0.01 ,max(pdf[-1])+0.01])
    ax3.set_xticks([])
    ax3.set_yticks([])  
    ax3.axis('off')

    for i in range(len(z_range)):
        ax2.plot(t_range, traj[:, i], color='darkseagreen', linewidth=1, linestyle='-', alpha=0.8)
    ax2.set_xlabel('Time (t)', fontsize=13)
    ax2.set_xlim([0,1])
    ax2.set_ylim([lb,ub])
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.axis('off')

    # Adjust the space between subplots
    # fig.subplots_adjust(wspace=0)
    # plt.show()
    plt.savefig(f'results/vec_{condition}.png', dpi=400, bbox_inches='tight')


def plot_flow(pdf, traj, x_range, z_range, t_range, lb, ub, condition='toy'):
    # Create a GridSpec object with 1 row and 3 columns
    gs = GridSpec(nrows=1, ncols=3, width_ratios=[1, 3, 1], wspace=0.1)
    max_pdf = np.max(pdf)
    cmap = plt.get_cmap('viridis')
    # norm = Normalize(vmin=0, vmax=max_pdf)

    # Create a new figure
    fig = plt.figure(figsize=(6, 3))

    # Add subplots to the figure using cells from the GridSpec
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Plot the data in each subplot
    # ax1.scatter(pdf[0], x_range, c=pdf[0]/max_pdf, marker='o', s=7, cmap=cmap)
    points = np.array([pdf[0], x_range]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,  linewidth=4)
    lc.set_array(pdf[0])
    ax1.add_collection(lc)
    ax1.set_title('Initial PDF', fontsize=14)
    ax1.set_xlabel('Density', fontsize=13)
    ax1.set_ylabel('Random Variable (x)', fontsize=13)
    ax1.set_ylim([lb,ub])
    ax1.set_xlim([-0.01,max(pdf[0])+0.01])
    ax1.set_xticks([0])
    ax1.set_yticks([lb,0,ub])

    # ax3.scatter(pdf[-1], x_range, c=pdf[-1]/max_pdf, marker='o', s=7, cmap=cmap)
    points = np.array([pdf[-1], x_range]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap,  linewidth=4)
    lc.set_array(pdf[-1])
    ax3.add_collection(lc)
    ax3.set_title('Terminal PDF', fontsize=14)
    ax3.set_xlabel('Density', fontsize=13)
    ax3.set_ylim([lb,ub])
    ax3.set_yticks([])
    ax3.set_xlim([-0.01 ,max(pdf[-1])+0.01])
    ax3.set_xticks([0])


    ax2.set_title('Evolution of PDF', fontsize=14)
    index = (x_range<=ub) & (x_range>=lb)
    # X, T = np.meshgrid(x_range[index], t_range)
    # ax2.contourf(T, X, pdf[:,index], cmap=cmap, levels=200, norm=norm)
    for i in range(len(t_range)):
        ax2.scatter(t_range[i]*np.ones(x_range[index].shape), x_range[index], c=pdf[i,index], marker='o', s=5, cmap=cmap)
    for i in range(len(z_range)):
        j = np.random.randint(0, 300)
        ax2.plot(t_range[j:], traj[j:,i], color='w', linewidth=1, linestyle='--', alpha=0.8,
                dashes=(70,35))
    ax2.set_xlabel('Time (t)', fontsize=13)
    ax2.set_xlim([0,1])
    ax2.set_ylim([lb,ub])
    ax2.set_yticks([])
    ax2.set_xticks([0,0.5,1])


    # Adjust the space between subplots
    # fig.subplots_adjust(wspace=0)
    # plt.show()
    plt.savefig(f'results/vec_{condition}.png', dpi=400, bbox_inches='tight')


def run_flow_1():
    # Define the vector field v(x, t)
    def v(x, t):
        xt = np.sin(x+np.pi/4) * 4 + t
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
    z_range = np.linspace(lb, ub, 30)

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

    plot_flow(pdf, traj, x_range, z_range, t_range, lb, ub)


# Integrate the Gaussian PDF using the trapezoidal rule
def run_flow():
    # Define the vector field v(x, t)
    def vec(x, z, t, model):
        model.eval()
        x_tensor = torch.ones(size=[z.shape[0], 1]) * x
        z_tensor = torch.tensor(z).view(-1,1)
        t_tensor = torch.ones(size=[z.shape[0], 1]) * t
        vec =  model.predict_vec(x_tensor, z_tensor, t_tensor)
        return vec.view(-1).detach().cpu().numpy()
    model = torch.load('models/2_toy/rectified_mlp.pth', map_location='cpu')
    condition = 0
    # Parameters
    num_timesteps = 1000
    num_evaluation = 1000
    num_simulations = 1000
    dt = 1.0 / num_timesteps
    ub =3
    lb = -3
    x_range = np.linspace(lb, ub, num_evaluation)
    t_range = np.linspace(0, 1, num_timesteps)
    z_range = np.linspace(lb, ub, 30)

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
        cur_x = simu[t_idx-1,:]
        cur_vec = vec(condition, cur_x, t_range[t_idx-1], model)
        simu[t_idx,:] = cur_x + cur_vec * dt #+ sigma * dW
    # Compute the PDF for each time step on a regular grid
    bw_ub = (ub-lb)/num_simulations*500
    bw_lb = (ub-lb)/num_simulations*10
    for t in range(1, num_timesteps):
        kde = gaussian_kde(simu[t,:], bw_method=max(bw_ub*0.995**t, bw_lb))
        pdf[t,:] = kde(x_range)
    # Solve the ODE using the forward Euler method
    for t_idx in range(1, num_timesteps):
        cur_z = traj[t_idx-1,:]
        cur_vec = vec(condition, cur_z,  t_range[t_idx-1], model)
        traj[t_idx,:] = cur_z + dt * cur_vec

    plot_flow(pdf, traj, x_range, z_range, t_range, lb, ub, condition)

