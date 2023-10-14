import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Variable

torch.set_default_dtype(torch.float32)


"""
Naive model Class
"""
class Simple_NN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, output_act, pred_type):
        super(Simple_NN, self).__init__()
        if network == 'mlp':
            self.net = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act)
        elif network == 'att':
            self.net = ATT(input_dim, 1, hidden_dim, num_layers, output_act, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, y_pred, y_target):
        return self.criterion(y_pred, y_target)


class GMM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, num_cluster, output_act, pred_type):
        super(GMM, self).__init__()
        self.num_cluster = num_cluster
        self.output_dim = output_dim
        if network == 'mlp':
            self.Predictor = MLP(input_dim, output_dim * num_cluster, hidden_dim, num_layers, output_act)
            self.Classifier = MLP(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', output_dim)
        elif network == 'att':
            self.Predictor = ATT(input_dim, num_cluster, hidden_dim, num_layers, output_act, pred_type=pred_type)
            self.Classifier = ATT(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', 1, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y_pred = self.Predictor(x).view(x.shape[0], -1, self.num_cluster)
        return y_pred

    def loss(self, x, y_pred, y_target):
        c_pred = self.Classifier(x, y_target).view(x.shape[0], -1, self.num_cluster)
        y_pred = (c_pred * y_pred).sum(-1)
        loss = self.criterion(y_pred, y_target)
        return loss
    
    def hindsight_loss(self, x, y_pred, y_target):
        loss = (y_pred - y_target.unsqueeze(-1))**2
        loss = loss.mean(dim=1)
        loss = loss.min(dim=1)[0]
        return loss.mean()



"""
GNN model Class
"""
class GNN(nn.Module):
    """
    TSP:
        Input: node: B*N*2, edge: 0
        Output: edge: B*N*N
    MCP/MIS/MCUT:
        Input: node: 0, edge: B*N*N
        Output: node: B*N
    """

    def __init__(self, hidden_dim, num_layer):
        self.num_layer = num_layer
        self.node_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_emb = ScalarEmbeddingSine(hidden_dim, normalize=False)
        self.gnn_emb = nn.ModuleList(
            [GNNLayer(hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True) for _
             in range(num_layer)])
        self.out_emb = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, node, edge):
        node_emb = self.node_emb(node)
        edge_emb = self.edge_emb(edge)
        for layer in zip(self.gnn_emb):
            node_emb, edge_emb = layer(node_emb, edge_emb)
        node_pred = self.out_emb(node_emb)
        return node_pred



"""
Generative Adversarial model Class
"""
class VAE(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(VAE, self).__init__() 
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Encoder = MLP(input_dim, latent_dim*2, hidden_dim, num_layers, None)
            self.Decoder = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
        elif network == 'att':
            NotImplementedError
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()          

    def forward(self, x):
        z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        y_pred = self.Decoder(x, z)
        return y_pred

    def reparameterize(self, mean, logvar):
        z = torch.randn_like(mean).to(mean.device)
        return mean + z * torch.exp(0.5 * logvar)

    def encoder_decode(self, x):
        para = self.Encoder(x)
        mean, logvar = torch.chunk(para, 2, dim=-1)
        z = self.reparameterize(mean, logvar)
        y_recon = self.Decoder(x, z)
        return y_recon, mean, logvar

    def loss(self, y_recon, y_target, mean, logvar):
        kl_div = -0.5 * torch.sum(1+logvar-mean.pow(2)-logvar.exp())
        return self.criterion(y_recon, y_target) + kl_div.mean()


class GAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = MLP(input_dim, 1, hidden_dim, num_layers, 'sigmoid', output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, num_layers, 'sigmoid', latent_dim=1, agg=True)
        else:
            NotImplementedError
        self.criterion = nn.BCELoss()

    def forward(self, x, z):
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        return self.criterion(self.Discriminator(x, y_pred), valid)

    def loss_d(self, x, y_target, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        fake = torch.zeros([x.shape[0], 1]).to(x.device)
        d_loss = (self.criterion(self.Discriminator(x, y_target), valid) +
                  self.criterion(self.Discriminator(x, y_pred.detach()), fake)) / 2
        return d_loss


class WGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, None, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x):
        z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        return -torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return w_dis_dual 

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp


class DWGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, 'abs', output_dim)
            # self.Discriminator = MLP(input_dim, 1,  hidden_dim, num_layers, output_act, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x):
        z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        v_pred = self.Generator(x, z)
        return z + v_pred

    def loss_g(self, x, y_pred):
        return torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return -w_dis_dual + 50 * (torch.mean(self.Discriminator(x, y_target))) ** 2

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp



"""
Generative Diffusion model Class
"""
class DM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(DM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.con_dim = input_dim
        self.time_step = time_step
        self.output_dim = output_dim
        beta_max = 0.02
        beta_min = 1e-4
        self.betas = sigmoid_beta_schedule(self.time_step, beta_min, beta_max)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def predict_noise(self, x, y, t, noise):
        y_t = self.diffusion_forward(y, t, noise)
        noise_pred = self.model(x, y_t, t)
        return noise_pred

    def diffusion_forward(self, y, t, noise):
        if self.normalize:
            y = y * 2 - 1
        t_index = (t * self.time_step).to(dtype=torch.long)
        alphas_1 = self.sqrt_alphas_cumprod[t_index].to(y.device)
        alphas_2 = self.sqrt_one_minus_alphas_cumprod[t_index].to(y.device)
        return (alphas_1 * y + alphas_2 * noise)

    def diffusion_backward(self, x, z, inf_step=100, eta=0.5):
        if inf_step==self.time_step:
            """DDPM"""
            for t in reversed(range(0, self.time_step)):
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                z = self.sqrt_recip_alphas[t]*(z-self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) \
                    + torch.sqrt(self.posterior_variance[t]) * noise
        else: 
            """DDIM"""
            sample_time_step = torch.linspace(self.time_step-1, 0, (inf_step + 1)).to(x.device).to(torch.long)
            for i in range(1, inf_step + 1):
                t = sample_time_step[i - 1] 
                prev_t = sample_time_step[i] 
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                y_0 = (z - self.sqrt_one_minus_alphas_cumprod[t] * pred_noise) / self.sqrt_alphas_cumprod[t]
                var = eta * self.posterior_variance[t]
                z = self.sqrt_alphas_cumprod[prev_t] * y_0 + torch.sqrt(torch.clamp(1 - self.alphas_cumprod[prev_t] - var, 0, 1)) * pred_noise + torch.sqrt(var) * noise
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def loss(self, noise_pred, noise):
        return self.criterion(noise_pred, noise)

class FM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(FM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def flow_forward(self, y, t, z, vec_type='gaussian'):
        if self.normalize:
            y = 2 * y - 1  # [0,1] normalize to [-1,1]
        if vec_type == 'gaussian':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for mu and sigma
            """
            mu = y * t
            sigma = (self.min_sd) * t + 1 * (1 - t)
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'conditional':
            mu = t * y + (1 - t) * z
            sigma = ((self.min_sd * t) ** 2 + 2 * self.min_sd * t * (1 - t)) ** 0.5
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'rectified':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for z and y
            """
            yt = t * y + (1 - t) * z
            vec = y-z
        elif vec_type == 'interpolation':
            """
            t = 0:  x
            t = 1:  N(0,1)
            Linear interpolation for z and y
            """
            # yt = (1 - t) * y + t * z
            yt = t * y + (1 - t) * z
            vec = None
            # return torch.cos(torch.pi/2*t) * y + torch.sin(torch.pi/2*t) * z
            # return (torch.cos(torch.pi*t) + 1)/2 * y + (torch.cos(-torch.pi*t) +1)/2  * z
        return yt, vec

    def flow_backward(self, x, z, step=0.01, method='Euler', direction='forward'):
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0):
            z += ode_step(self.model, x, z, t, step, method)
            t += step
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def predict_vec(self, x, yt, t):
        vec_pred = self.model(x, yt, t)
        return vec_pred

    def loss(self, y, z, vec_pred, vec, vec_type='gaussian'):
        if vec_type in ['gaussian', 'rectified', 'conditional']:
            return self.criterion(vec_pred, vec)
        elif vec_type in ['interpolation']:
            loss = 1 / 2 * torch.sum(vec_pred ** 2, dim=1, keepdim=True) \
                   - torch.sum((y - z) * vec_pred, dim=1, keepdim=True)
            return loss.mean()
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi/2*torch.sin(torch.pi/2*t) * y +  torch.pi/2*torch.cos(torch.pi/2*t) * z) * vec, dim=-1, keepdim=True)
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi*torch.sin(torch.pi*t)*y +    torch.pi*torch.sin(-torch.pi*t) * z) * vec, dim=-1, keepdim=True)
        else:
            NotImplementedError

class AM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(AM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, 1, hidden_dim, num_layers, None, output_dim, 'silu')
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
    
    def loss(self, x, y, z, t):
        s0 = self.model(x, y, torch.zeros_like(t))
        s1 = self.model(x, y, torch.ones_like(t))
        yt = t*y + (1-t)*z 
        yt.requires_grad = True
        st = self.model(x, yt, t)
        vec = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        loss = self.criterion(vec, y-z)
        # loss =  1 / 2 * torch.sum(vec ** 2, dim=1, keepdim=True) \
        #            - torch.sum((y - z) * vec, dim=1, keepdim=True)
        # t.requires_grad = True
        # st = self.model(x, yt, t)
        # dsdt = torch.autograd.grad(st, t, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # dsdy = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # loss = s0 - s1 + 0.5 * torch.sum(dsdy**2, dim=1, keepdim=True) + dsdt
        return loss.mean()

    def flow_backward(self, x, step=0.01, method='Euler', direction='forward'):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device)
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0):
            z.requires_grad = True
            st = self.model(x, z, torch.ones(size=[z.shape[0],1]).to(x.device)*t)
            dsdz = torch.autograd.grad(st, z, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
            z.requires_grad = False
            z += dsdz.detach() * step
            t += step
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

class CM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(CM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
            # self.target_model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
            # self.target_model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
        self.std = 80
        self.eps = 0.002
    
    def flow_forward(self, y, z, t):
        yt = y + z * t * self.std
        # yt = (1-t) * y + z * t 
        return yt

    def predict(self, x, yt, t, model):
        return self.c_skip_t(t) * yt + self.c_out_t(t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) * self.std
        t = torch.ones(size=[x.shape[0],1]).to(x.device)
        y0 =  self.predict(x, z, t, self.target_model)
        return y0

    def loss(self, x, y, z, t1, t2, data, vec):
        # y = data.scaling_v(y)
        yt1 = self.flow_forward(y, z, t1)
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            yt2 = self.flow_forward(y, z, t2)
            y02 = self.predict(x, yt2, t2, self.target_model)
            # y02_scale = data.scaling_v(y02)
        return (self.criterion(y01, y02)).mean() #+ 0.0001*pel.mean()

    def c_skip_t(self, t):
        t = t * self.std
        return 0.25 / (t.pow(2) + 0.25)
    
    def c_out_t(self, t):
        t = t * self.std
        return 0.25 * t / ((t + self.eps).pow(2)).pow(0.5)

    def kerras_boundaries(self, sigma, eps, N, T):
        # This will be used to generate the boundaries for the time discretization
        return torch.tensor(
            [
                (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
                ** sigma
                for i in range(N)
            ]
        )

class CD(CM):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super().__init__(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type)
        self.criterion = nn.L1Loss()

    def loss(self, x, y, z, t1, step, forward_step, data, vec_model):
        # print(self.c_out_t(torch.zeros(1)), self.c_skip_t(torch.zeros(1)))
        yt1 = (1-t1) * z + t1 * y
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            v_pred_1 = vec_model.predict_vec(x, yt1, t1) * step
            yt2 = yt1 + v_pred_1
            t2 = t1 + step
            for _ in range(forward_step):
                v_pred_2 = vec_model.predict_vec(x, yt2, t2) * step
                yt2 = yt2 + v_pred_2
                t2 = t2 + step
            y02 = self.predict(x, yt2, t2, self.model)

            # v_pred_1 = vec_model.predict_vec(x, yt2, t2) * step
            # yt2 = yt1 + (v_pred_0 + v_pred_1)/2 
            # v_pred_0 = vec_model.predict_vec(x, yt1, t1) * step
            # v_pred_1 = vec_model.predict_vec(x, yt1 + v_pred_0 * 0.5, t1 + step * 0.5) * step
            # v_pred_2 = vec_model.predict_vec(x, yt1 + v_pred_1 * 0.5, t1 + step * 0.5) * step
            # v_pred_3 = vec_model.predict_vec(x, yt1 + v_pred_2, t1 + step) * step
            # yt2 = yt1 + (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
        return (self.criterion(y01, y02)).mean() #+ 0.001*pel.mean()
    
    def continnuous_loss(self, x, y, z, t, stpe, forward_step, data, vec_model):
        yt = (1-t) * z + t * y
        with torch.no_grad():
            vec = vec_model.predict_vec(x, yt, t)
        yt.requires_grad_(True)
        t.requires_grad_(True)
        y0 = self.predict(x, yt, t, self.model)
        dy = torch.autograd.grad(y0, yt, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        dt = torch.autograd.grad(y0, t, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        return torch.square(dy * vec + dt).mean()



    def predict(self, x, yt, t, model):
        return yt + (1-t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # t = torch.zeros(size=[x.shape[0],1]).to(x.device)
        # y0 =  self.predict(x, z, t, self.target_model)
        y0 = 0
        for dt in torch.linspace(0,1,1):
            z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
            t = torch.ones(size=[x.shape[0],1]).to(x.device) * dt
            yt = (1-t) * z + t * y0
            y0 =  self.predict(x, yt, t, self.model)
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # yt1 = (y0+z)/2
        # y0 =  self.predict(x, yt1, t+0.5, self.target_model)
        return y0

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def ode_step(model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, t: float, step: float, method: str = 'Euler'):
    model.eval()
    t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t

    def model_eval(z_eval, t_eval):
        return model(x, z_eval, t_eval)

    with torch.no_grad():
        if method == 'Euler':
            v_pred = model_eval(z, t_tensor) * step
        else:
            v_pred_0 = model_eval(z, t_tensor) * step
            if method == 'Heun':
                v_pred_1 = model_eval(z + v_pred_0, t_tensor + step) * step
                v_pred = (v_pred_0 + v_pred_1) / 2
            elif method == 'Mid':
                v_pred = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
            elif method == 'RK4':
                v_pred_1 = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
                v_pred_2 = model_eval(z + v_pred_1 * 0.5, t_tensor + step * 0.5) * step
                v_pred_3 = model_eval(z + v_pred_2, t_tensor + step) * step
                v_pred = (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
    return v_pred





class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        gumbel_noise = self.sample_gumbel(x.size())
        y = x + gumbel_noise.to(x.device)
        soft_sample = F.softmax(y / self.temperature, dim=-1)

        if self.hard:
            hard_sample = torch.zeros_like(soft_sample).scatter(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample - soft_sample.detach() + soft_sample
        else:
            sample = soft_sample

        return sample

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Time_emb(nn.Module):
    def __init__(self, emb_dim, time_steps, max_period):
        super(Time_emb, self).__init__()
        self.emb_dim = emb_dim
        self.time_steps = time_steps
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        t = t.view(-1) * self.time_steps
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)


class Lip_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0):
        super(Lip_MLP, self).__init__()
        if latent_dim > 0:
            w = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            b = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
            t = [Time_emb(hidden_dim, time_steps=1000, max_period=1000)]
            self.w = nn.Sequential(*w)
            self.b = nn.Sequential(*b)
            self.t = nn.Sequential(*t)
            net = []
        else:
            latent_dim = input_dim

        emb = [LinearNormalized(latent_dim, hidden_dim), nn.ReLU()]
        self.emb = nn.Sequential(*emb)
        net = []
        for _ in range(num_layer):
            net.extend([LinearNormalized(hidden_dim, hidden_dim), nn.ReLU()])
        net.append(LinearNormalized(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.t(t)
        y = self.net(emb)
        return self.act(y)

    def project_weights(self):
        self.net.project_weights()


class ATT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation=None, latent_dim=0, agg=False,
                 pred_type='node', act='relu'):
        super(ATT, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)

        net = []
        # for _ in range(num_layer):
            # net.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            # net.extend([MHA(hidden_dim, 64, hidden_dim // 64),
            #             ResBlock(hidden_dim, hidden_dim//4)])
        # net.append(nn.Linear(hidden_dim, output_dim))
        # self.net = nn.Sequential(*net)
        self.net = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(hidden_dim // 64, 1))
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        # self.mha  = MHA(hidden_dim, 64, hidden_dim // 64)

        self.agg = agg
        self.pred_type = pred_type

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        ### x: B * N * F
        ### z: B * N
        ### t: B * 1
        batch_size = x.shape[0]
        node_size = x.shape[1]
        if z is None:
            # print(x.shape, self.emb)
            emb = self.emb(x)
        else:
            z = z.view(batch_size, -1, 1)
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.temb(t).view(batch_size, 1, -1)
        emb = emb.permute(1, 0, 2)
        emb = self.trans(emb)
        emb = emb.permute(1, 0, 2)
        y = self.net(emb)  # B * N * 1
        
        if self.agg:
            y = y.mean(1)
        else:
            if self.pred_type == 'node':
                y = y.view(x.shape[0], -1)  # B * N
            else:
                y = torch.matmul(y.view(batch_size, node_size, 1), y.view(batch_size, 1, node_size))  # B * N * N
                col, row = torch.triu_indices(node_size, node_size, 1)
                y = y[:, col, row]
        return self.act(y)


class MHA(nn.Module):
    def __init__(self, n_in, n_emb, n_head):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.key = nn.Linear(n_in, n_in)
        self.query = nn.Linear(n_in, n_in)
        self.value = nn.Linear(n_in, n_in)
        self.proj = nn.Linear(n_in, n_in)

    def forward(self, x):
        # x: B * node * n_in
        batch = x.shape[0]
        node = x.shape[1]
        ### softmax
        #### key: B H node emb
        #### que: B H emb node
        key = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        query = self.query(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,2,1)
        value = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        score = torch.matmul(key, query)/(self.n_emb**0.5) # x: B * H * node * node
        prob = torch.softmax(score, dim=-1) # B * H * node * node (prob)
        out = torch.matmul(prob, value) # B * H * Node * 64
        out = out.permute(0,2,3,1).contiguous() # B * N * F * H
        out = out.view(batch, -1, self.n_emb*self.n_head)
        return x + self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid), 
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)


class SDPBasedLipschitzLinearLayer(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1) + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class PartialLinearNormalized(nn.Module):
    def __init__(self, input_dim, output_dim, con_dim):
        super(PartialLinearNormalized, self).__init__()
        self.con_dim = con_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_1 = spectral_norm(nn.Linear(input_dim - con_dim, output_dim))

    def forward(self, x):
        with torch.no_grad():
            weight_copy = self.linear_1.weight.data.clone()
            self.linear.weight.data[:, self.con_dim:] = weight_copy
        return self.linear(x)


class distance_estimator(nn.Module):
    def __init__(self, n_feats, n_hid_params, hidden_layers, n_projs=2, beta=0.5):
        super().__init__()
        self.hidden_layers = hidden_layers  # number of hidden layers in the network
        self.n_projs = n_projs  # number of projections to use for weights onto Steiffel manifold
        self.beta = beta  # scalar in (0,1) for stabilizing feed forward operations

        # Intialize initial, middle, and final layers
        self.fc_one = torch.nn.Linear(n_feats, n_hid_params, bias=True)
        self.fc_mid = nn.ModuleList(
            [torch.nn.Linear(n_hid_params, n_hid_params, bias=True) for i in range(self.hidden_layers)])
        self.fc_fin = torch.nn.Linear(n_hid_params, 1, bias=True)

        # Normalize weights (helps ensure stability with learning rate)
        self.fc_one.weight = nn.Parameter(self.fc_one.weight / torch.norm(self.fc_one.weight))
        for i in range(self.hidden_layers):
            self.fc_mid.weight = nn.Parameter(self.fc_mid[i].weight / torch.norm(self.fc_mid[i].weight))
        self.fc_fin.weight = nn.Parameter(self.fc_fin.weight / torch.norm(self.fc_fin.weight))

    def forward(self, u):
        u = self.fc_one(u).sort(1)[0]  # Apply first layer affine mapping
        for i in range(self.hidden_layers):  # Loop for each hidden layer
            u = u + self.beta * (self.fc_mid[i](u).sort(1)[0] - u)  # Convex combo of u and sort(W*u+b)
        u = self.fc_fin(u)  # Final layer is scalar (no need to sort)
        J = torch.abs(u)
        return J

    def project_weights(self):
        self.fc_one.weight.data = self.proj_Stiefel(self.fc_one.weight.data, self.n_projs)
        for i in range(self.hidden_layers):
            self.fc_mid[i].weight.data = self.proj_Stiefel(self.fc_mid[i].weight.data, self.n_projs)
        self.fc_fin.weight.data = self.proj_Stiefel(self.fc_fin.weight.data, self.n_projs)

    def proj_Stiefel(self, Ak, proj_iters):  # Project to closest orthonormal matrix
        n = Ak.shape[1]
        I = torch.eye(n)
        for k in range(proj_iters):
            Qk = I - Ak.permute(1, 0).matmul(Ak)
            Ak = Ak.matmul(I + 0.5 * Qk)
        return Ak


