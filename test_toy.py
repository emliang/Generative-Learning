from utiles import *
from data_utiles import *
from default_args import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _main_():
    args = toy_args()
    args = modify_args(args)
    data = Toy_Dataset(args)
    data.device = DEVICE
    # plot_single(data, args, model_name='data')
    model_name_list = ['simple', 'hindsight', 'cluster', 'gan', 'diffusion', 'rectified']
    # for model_name in model_name_list:
    #     train_all(data, args, model_name)
    plot_all(data, args, model_name_list)
_main_()