import os

def toy_args():
    args = {'data_set': 'toy',
            'data_type': 2,
            'data_size': 10000,
            'network': 'mlp',
            'output_dim': 1,
            'latent_dim': 1,
            'num_iteration': 10000,
            'test_freq': 3000,
            'batch_dim': 1024,
            'hidden_dim': 128,
            'num_layer': 3,
            'output_act': None,
            'learning_rate': 1e-3,
            'learning_rate_decay': [1000,0.9],
            'weight_decay': 1e-6,
            'num_cluster': 3,
            'update_generator_freq': 5,
            'output_norm': False,
            'test_dim': 512,
            'time_step': 1000,
            'inf_step': 100,
            'eta':0.5,
            'ode_solver': 'Euler'}
    return args


def opt_args():
    args = toy_args()
    args['data_set'] = ['max_cut', 'min_cover'][0]
    args['graph_dim'] = 10
    args['graph_sparsity'] = [0.3,0.7]
    args['output_dim'] = args['graph_dim']
    args['data_type'] = str(args['graph_dim'])
    args['test_freq'] = 3000
    args['learning_rate'] = 1e-3
    args['output_act'] = None
    args['network'] = 'att'
    args['data_dim'] = 10000
    args['batch_dim'] = 256
    args['hidden_dim'] = 256
    args['num_layer'] = 1
    args['num_cluster'] = 4
    args['latent_dim'] = args['graph_dim']
    args['test_dim'] = 1024
    args['time_step'] = 1000
    args['inf_step'] = 5
    args['inf_sample'] = 100
    args['eta'] = 1
    args['ode_solver'] = 'Euler'
    return args


def opf_args():
    args = toy_args()
    args['data_set'] = 'acpf'
    args['graph_dim'] = 5
    args['network'] = 'mlp'
    args['data_type'] = str(args['graph_dim'])
    args['num_iteration'] = 10000
    args['test_freq'] = 1000
    args['learning_rate'] = 1e-3
    args['learning_rate_decay'] = [1000, 0.9]
    args['output_act'] = None
    args['data_dim'] = -1
    args['batch_dim'] = 512
    args['hidden_dim'] = min(args['graph_dim']*4, 2048)
    args['num_layer'] = 3
    args['latent_dim'] = args['graph_dim']
    args['test_dim'] = 1024
    args['time_step'] = 1000
    args['inf_step'] = 100
    args['inf_sample'] = 10
    args['cor_step'] = 10
    return args  


def modify_args(args, data_set=None, data_type=None):
    if data_set is None and data_type is None:
        instance =str(args['data_type']) + '_' + str(args['data_set'])
    else:
        instance = str(data_type) + '_' + str(data_set)
        args['data_set'] = data_set
        args['data_type'] = data_type
    args['instance'] = instance
    if not os.path.exists(f'models/{instance}'):
        os.makedirs(f'models/{instance}')
    if not os.path.exists(f'results/{instance}'):
        os.makedirs(f'results/{instance}')
    return args



