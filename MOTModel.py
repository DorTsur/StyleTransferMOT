import torch
import torch.nn as nn
import torch.nn.functional as F



class MOTModel(object):
    def __init__(self, d,k,eps, lr, barycenter_weights):
        self.models = []
        self.d = d
        self.k = k
        self.eps = eps
        if barycenter_weights == 'uniform':
            self.cost_weights = [1/(k-1)]*(k-1)
        elif barycenter_weights == 'only_0':
            self.cost_weights = [1.] + [0.] * (k - 2)
        elif barycenter_weights == 'only_1':
            self.cost_weights = [0.] * (k - 2) + [1.]
        else:
            lst = [float(char)/(k-1) for char in barycenter_weights]
            self.cost_weights = [l/sum(lst) for l in lst]

        # create k models:
        self.models = [Net_EOT(dim=d, K=max(d, 256), deeper=True).cuda() for _ in range(self.k)]

        self.param_groups = [
            {'params': list(self.models[i].parameters()), 'lr': lr} for i in range(k)
        ]

        self.all_params = []
        for model in self.models:
            self.all_params += list(model.parameters())

        self.opt_all = torch.optim.Adam(self.param_groups, lr=lr)
        self.opt_all = torch.optim.Adam(self.all_params, lr=lr)


    def zero_grad_models(self):
        if isinstance(self.opt, list):
            # list of opts
            for opt in self.opt:
                opt.zero_grad()
        else:
            # one opt to rule them all
            self.opt.zero_grad()

    def calc_exp_term(self, phi, X):
        """
        steps:
        1. calculate the cost object from the data
        2. mix it with phi terms
        3. reduce to loss
        Assumption - i=0 is the barycenter, i=1 is the content, i=2...k-1 are the styles
        """
        # Calc cost:

        n = phi[0].shape[0]
        c = 1/n*torch.ones(size=(n, 1)).cuda()
        vectors_0 = X[:, 0, :].unsqueeze(0)  # (1, n, d)
        for i in range(1, self.k):
            vectors_i = X[:, i, :].unsqueeze(1)  # (n, 1, d)
            vector_diffs = vectors_0 - vectors_i  # (n, n, d)
            C_i = torch.norm(vector_diffs, dim=2)/torch.tensor(X.shape[-1], dtype=torch.float32)  # (n, n), squared L2 norms
            exponent = (torch.sum(phi[i] - self.cost_weights[i-1]*C_i, dim=1)) / self.eps
            L_i = 1 / n * torch.exp(exponent)
            c *= L_i.unsqueeze(1)
        return c.sum()












class Net_EOT(nn.Module):
    def __init__(self,dim,K,deeper=False):
        super(Net_EOT, self).__init__()
        self.deeper=deeper
        if deeper:
            self.fc1 = nn.Linear(dim, 10 * K)
            self.fc2 = nn.Linear(10 * K, 10 * K)
            self.fc3 = nn.Linear(10 * K, 1)
        else:
            self.fc1 = nn.Linear(dim, K)
            self.fc2 = nn.Linear(K, 1)

    def forward(self, x):
        if self.deeper:
            x1 = F.relu(self.fc1(x))
            x11 = F.relu(self.fc2(x1))
            x2 = self.fc3(x11)
            #
            x2 = torch.tanh(x2)
            #
        else:
            x1 = F.relu(self.fc1(x))
            x2 = self.fc2(x1)
            #
            x2 = torch.tanh(x2)
            #
        return x2






#################################
#################################
#################################
#        Barycenter Models:
#################################
#################################
#################################
class LayerNormBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LayerNormBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # self.ln = nn.LayerNorm(in_features)  # LayerNorm for stability
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear(x)
        # x = self.ln(x)
        x = self.activation(x)
        return x

class BaryTransformModel_(nn.Module):
    def __init__(self, in_dim=64):
        super(BaryTransformModel_, self).__init__()

        hidden_dims = (in_dim, in_dim*2, in_dim)
        # Initial layer: Latent dim to first hidden dimension
        self.initial_linear = nn.Linear(in_dim, hidden_dims[0])
        self.ln_initial = nn.LayerNorm(hidden_dims[0])
        self.activation = nn.ReLU()

        # Hidden layers
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(LayerNormBlock(hidden_dims[i], hidden_dims[i+1]))
        self.hidden_layers = nn.Sequential(*layers)

        # Final layer: last hidden dimension to output dimension
        self.final_linear = nn.Linear(hidden_dims[-1], in_dim)

    def forward(self, z):
        x = self.initial_linear(z)
        x = self.ln_initial(x)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.final_linear(x)
        return x










#################################
#################################
#################################
#        old Models:
#################################
#################################
#################################


class BaryTransformModel(nn.Module):
    def __init__(self, args, in_dim):  # dimension of the output vector
        super(BaryTransformModel, self).__init__()
        input_dim = in_dim
        hidden_dim = max(512, 2*in_dim)
        # Linear + BN + Activation block
        def gen_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            gen_block(input_dim, hidden_dim),
            gen_block(hidden_dim, hidden_dim),
            gen_block(hidden_dim, hidden_dim),
            gen_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, z):
        return self.net(z)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # DCGAN-like initialization
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.constant_(m.bias, 0)
class BarycenterModel(nn.Module):
    def __init__(self, content, noise):
        super(BarycenterModel, self).__init__()
        # content should be a leaf tensor, so detach if needed
        # content.detach().clone()

        # temp = content.detach().cpu().numpy().copy()
        # param = torch.from_numpy(temp)

        if noise:
            content += 0.1*torch.randn(content.shape)

        self.param = nn.Parameter(content)


    def forward(self, indices):
        # Return the subset of barycenter selected by indices
        # indices should be a LongTensor (or equivalent) of indices
        out = self.param[:, :, indices[0], indices[1]]
        return out
