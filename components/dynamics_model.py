import torch
import torch.nn as nn
from torch.functional import F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7, ensemble_select_size=5, weight_decay=0.0, load_model=False):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.ensemble_select_size = ensemble_select_size

        self.register_parameter('weight', nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

        self.register_parameter('saved_weight', nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', nn.Parameter(self.bias.detach().clone()))

        if not load_model:
            self.register_parameter('select', nn.Parameter(torch.tensor(list(range(0, self.ensemble_size))), requires_grad=False))
        else:
            self.register_parameter('select', nn.Parameter(torch.tensor(list(range(0, self.ensemble_select_size))), requires_grad=False))
        # self.select = list(range(0, self.ensemble_size))

        self.weight_decay = weight_decay

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        # self.select = indexes
        self.register_parameter('select', nn.Parameter(torch.tensor(indexes), requires_grad=False))
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def reset_select(self):
        self.register_parameter('select', nn.Parameter(torch.tensor(list(range(0, self.ensemble_size))), requires_grad=False))
    
    def get_decays(self):
        decays = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decays


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_features,
        ensemble_size=7,
        ensemble_select_size=5,
        load_model=False,
        with_reward=True,
        device="cpu"
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size
        self.ensemble_select_size = ensemble_select_size
        self.load_model = load_model
        self.device = torch.device(device)

        self.activation = Swish()

        module_list = []
        module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size, ensemble_select_size, 0.000025, load_model))
        module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size, ensemble_select_size, 0.00005, load_model))
        module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size, ensemble_select_size, 0.000075, load_model))
        module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size, ensemble_select_size, 0.000075, load_model))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size, ensemble_select_size, 0.0001, load_model)

        self.register_parameter('max_logvar', nn.Parameter(torch.ones(obs_dim + self.with_reward) * 0.5, requires_grad=True))
        self.register_parameter('min_logvar', nn.Parameter(torch.ones(obs_dim + self.with_reward) * -10, requires_grad=True))

        self.to(self.device)

    def forward(self, obs_action):
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def train(self):
        for layer in self.backbones:
            layer.reset_select()
        self.output_layer.reset_select()
    
    def get_decays(self):
        decays = 0
        for layer in self.backbones:
            decays += layer.get_decays()
        decays += self.output_layer.get_decays()
        return decays