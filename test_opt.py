from utiles import *
from data_utiles import *
from default_args import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                      
def _main_():
    for prob in ['max_clique', 'max_is', 'max_cut']:
        for dim in [50, 100]:
            args = opt_args()
            args['data_set'] = prob
            args['graph_dim'] = dim
            args['latent_dim'] = args['graph_dim']
            args['output_dim'] = args['graph_dim']
            args['data_type'] = str(args['graph_dim'])
            args = modify_args(args)
            data = Opt_Dataset(args)
            data.device = DEVICE
            for attr in dir(data):
                var = getattr(data, attr)
                if torch.is_tensor(var):
                    try:
                        setattr(data, attr, var.to(DEVICE))
                    except AttributeError:
                        pass
        # ['simple', 'hindsight', 'cluster', 'gan', 'diffusion', 'rectified']
        model_name_list = ['simple', 'hindsight', 'cluster', 'gan', 'diffusion', 'rectified']
        for approach in model_name_list:
            train_all(data, args, approach)

        obj_target = data.objective(data.graph_test, data.y_test.cpu().numpy())
        obj_target_1 = data.decoding(data.graph_test, data.y_test.cpu().numpy())
        print((np.abs(obj_target - obj_target_1)).sum())
        # print(1/0)
        st = time.time()
        _ = data.opt_solve(data.graph_test[:10])
        et = time.time()
        opt_time = (et-st)/10
        sample_num = args['inf_sample']
        for approach in model_name_list:
            obj_pred_list = []
            vio_pred_list = []
            if approach in ['simple', 'hindsight', 'cluster']:
                y_pred = test_single(data.x_test, args, approach, 2).cpu().numpy()
                for i in range(y_pred.shape[2]):
                    obj_pred = data.decoding(data.graph_test, y_pred[:,:,i])
                    obj_pred_list.append(obj_pred)  
                    # vio_pred = data.violation(data.graph_test, y_pred[:,:,i])
                    # vio_pred_list.append(vio_pred)
            else:
                for _ in range(sample_num):
                    y_pred = test_single(data.x_test, args, approach).cpu().numpy()
                    obj_pred = data.decoding(data.graph_test, y_pred)
                    obj_pred_list.append(obj_pred)
                    # vio_pred = data.violation(data.graph_test, y_pred)
                    # vio_pred_list.append(vio_pred)
            ave_inf_time = test_inf_time(data.x_test[0:1], args, approach, sample_num)
            obj_pred_max = np.array(obj_pred_list).T.max(1)
            # vio_pred_min = np.array(vio_pred_list).T.min(1)
            print(f'opt_gap of {approach}: {np.mean(np.abs(obj_pred_max-obj_target)/np.abs(obj_target)+1e-5): .4f},\
                   speedup {opt_time/ave_inf_time:.2f}')#  violation {np.sum(vio_pred_min)/len(vio_pred_min)} \

def test_inf_time(x_test, args, approach, sample_num):
    st = time.time()
    _ = test_single(x_test, args, approach, sample_num)
    et = time.time()
    return et-st

_main_()