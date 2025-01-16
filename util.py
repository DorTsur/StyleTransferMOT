from __future__ import division
import torch
from tqdm import tqdm
from modelsNIPS_ import decoder1,decoder2,decoder3,decoder4,decoder5
from modelsNIPS_ import encoder1,encoder2,encoder3,encoder4,encoder5
import torch.nn as nn
import torchfile
from torch.utils.data import DataLoader
from MOTModel import MOTModel, BarycenterModel, BaryTransformModel, BaryTransformModel_
import wandb
from Loader import GaussianDataset
import torchvision.utils as vutils
import os
from torch.distributions import MultivariateNormal





class pytorch_lua_wrapper:
    def __init__(self, lua_path):
        self.lua_model = torchfile.load(lua_path)

    def get(self, idx):
        return self.lua_model._obj.modules[idx]._obj



class trainerMOT(nn.Module):
    def __init__(self,args,k):
        super(trainerMOT, self).__init__()
        self.args = args
        self.load_models()
        self.k = k
        self.eps = args.eps


    def load_models(self):
        encoder = [encoder1,encoder2,encoder3,encoder4,encoder5]
        decoder = [decoder1,decoder2,decoder3,decoder4,decoder5]

        vgg_enc = []
        vgg_dec = []
        self.enc = []
        self.dec = []
        for i in range(5):
            path_enc = f'models/vgg{i+1}.t7'
            vgg_enc.append(pytorch_lua_wrapper(path_enc))
            self.enc.append(encoder[i](vgg_enc[i]))

            path_dec = f'models/decoder{i+1}.t7'
            vgg_dec.append(pytorch_lua_wrapper(path_dec))
            self.dec.append(decoder[i](vgg_dec[i]))

    def solveMOTBarycenterGenerative(self, content, styles, level):
        """
        Solving an MOT Barycenter optimization problem using PyTorch.
        Alternating between an MOT problem (neural potentials) and a Barycenter transform
        """
        # initialize MOT problem
        shape = content.shape
        n1 = shape[2]
        n2 = shape[3]
        d = shape[1]
        k = len(styles) + 2

        # initialize MOT model
        self.MOT = MOTModel(d, k, self.eps, self.args.nemot_lr, self.args.barycenter_weights)

        # two gaussian input tensor options, randn or same moments
        same_moments = True
        if same_moments:
            mu, sig = compute_gaussian_statistics(content.reshape(-1,d))
            gauss_in = torch.distributions.MultivariateNormal(mu, sig).sample([n1,n2])
            gauss_in = gauss_in.permute(2,0,1).unsqueeze(0)
        else:
            gauss_in = torch.randn(size=(1, d, n1, n2))

        X = [gauss_in] + [content] + styles

        X = torch.concatenate(X, 0)
        X = self.gen_dataloader(X, batch_size=self.args.batch_size)

        barycenter = BaryTransformModel_(in_dim=d).cuda()
        self.optimizer_barycenter = torch.optim.Adam(barycenter.parameters(), lr=self.args.barycenter_lr)

        # MOT + Barycenter routine
        for epoch in range(self.args.epochs):
            # gen dataloader:
            l = []
            grad_norm_b = 0
            grad_norm = 0
            # give 10 epoch warmup to NEMOT
            opt_flag = epoch > 9 and epoch % 2==0

            for batch, indices in tqdm(X, desc="Training Epoch"):
                if opt_flag:
                    self.optimizer_barycenter.zero_grad()
                else:
                    self.MOT.opt_all.zero_grad()



                batch = batch.cuda()

                g_in = batch[:,0,:]
                b = barycenter(g_in).unsqueeze(1)

                batch = torch.concatenate([b, batch[:,1:,:]], axis=1)
                phi = [self.MOT.models[i](batch[:, i, :]) for i in range(self.k)]
                e_term = self.MOT.calc_exp_term(phi, batch)

                loss = (sum(phi).mean() - self.eps * e_term)
                loss = loss if opt_flag else -loss

                loss.backward()
                l.append(loss.item())

                if opt_flag:
                    grad_norm_b = torch.nn.utils.clip_grad_norm_(barycenter.parameters(), max_norm=self.args.max_grad_norm_barycenter)
                    self.optimizer_barycenter.step()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.MOT.all_params, max_norm=self.args.max_grad_norm)
                    self.MOT.opt_all.step()

            if epoch % 50==0:
                # observe the transform applied to the entire guassian batch.
                with torch.no_grad():
                    gauss_in = gauss_in.cuda()
                    gauss_in = gauss_in.reshape(-1, d)
                    out = barycenter(gauss_in).detach().clone()
                    out[content.reshape(-1, d) == styles[0].reshape(-1, d)] = 0
                    out = out.reshape(1, d, n1, n2).cpu()
                    im = self.dec[level](out)
                    vutils.save_image(im, os.path.join(self.args.folderPath, f'level_{level}_epoch_{epoch}.png'))




            print(f'Finished Epoch {epoch + 1}, average loss is {sum(l) / len(l)}')
            if epoch>3:
                print(f'grad norm MOT: {grad_norm}')
                print(f'grad norm barycenter: {grad_norm_b}')
            if self.args.using_wandb:
                wandb.log({'epoch_loss': loss.item()})

        with torch.no_grad():
            gauss_in = gauss_in.cuda()
            gauss_in = gauss_in.reshape(-1, d)
            out = barycenter(gauss_in).detach().clone()
            out = out.reshape(1, d, n1, n2)
        return out.cpu()

    def gen_dataloader(self, X, batch_size):
        # pass
        dataset = ImagesDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def solveMOTBarycenter(self, content, styles):
        """
        Solving an MOT Barycenter optimization problem using PyTorch.
        Alternating between an MOT problem (neural potentials) and a Barycenter transform
        """

        # initialize barycenter:
        shape = content.shape

        # initialize MOT problem
        d = shape[1]
        k = len(styles) + 2
        self.MOT = MOTModel(d, k, self.eps, self.args.nemot_lr, self.args.barycenter_weights)

        enc_images = [content] + styles
        X = torch.concatenate(enc_images, 0)

        # creating the barycenter.
        content_clone = content.clone()

        # creating as a model.
        barycenter_ = BarycenterModel(content_clone, self.args.add_bary_noise).cuda()
        self.optimizer_barycenter = torch.optim.Adam(barycenter_.parameters(), lr=self.args.barycenter_lr)

        # MOT + Barycenter routine
        for epoch in range(self.args.epochs):
            # gen dataloader:
            x_b = self.gen_dataloader(X, batch_size=self.args.batch_size)
            l = []
            for batch, indices in tqdm(x_b, desc="Training Epoch"):
                opt_flag = epoch < 0 and epoch % 3

                if opt_flag:
                    self.optimizer_barycenter.zero_grad()
                else:
                    self.MOT.opt_all.zero_grad()

                batch = batch.cuda()
                b_ = barycenter_(indices)
                b_ = b_.permute(2, 0, 1)
                batch = torch.concatenate([b_, batch], axis=1)
                phi = [self.MOT.models[i](batch[:, i, :]) for i in range(self.k)]

                e_term = self.MOT.calc_exp_term(phi, batch)

                #checking with MSE loss:




                if opt_flag:
                    loss = -torch.log(e_term)
                else:
                    loss = -(sum(phi).mean() - self.eps * e_term)

                # prev:
                # loss = (sum(phi).mean() - self.eps * e_term)
                # loss = loss if opt_flag else -loss

                loss.backward()
                l.append(loss.item())

                # possible spot for grad clipping.

                if opt_flag:
                    grad_norm_b = torch.nn.utils.clip_grad_norm_(barycenter_.parameters(), max_norm=self.args.max_grad_norm_barycenter)
                    self.optimizer_barycenter.step()
                else:
                    all_params = []
                    for model in self.MOT.models:
                        all_params += list(model.parameters())

                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.args.max_grad_norm)
                    self.MOT.opt_all.step()

            print(f'Finished Epoch {epoch + 1}, average loss is {sum(l) / len(l)}')
            if epoch>3:
                print(f'grad norm MOT: {grad_norm}')
                print(f'grad norm barycenter: {grad_norm_b}')
            if self.args.using_wandb:
                wandb.log({'epoch_loss': loss.item()})

        with torch.no_grad():
            new_tensor = barycenter_.param.detach().clone()
        return new_tensor.cpu()




class trainerGauss(nn.Module):
    def __init__(self,args):
        super(trainerGauss, self).__init__()
        self.args = args
        self.load_models()

    def load_models(self):
        encoder = [encoder1,encoder2,encoder3,encoder4,encoder5]
        decoder = [decoder1,decoder2,decoder3,decoder4,decoder5]

        vgg_enc = []
        vgg_dec = []
        self.enc = []
        self.dec = []
        for i in range(5):
            path_enc = f'models/vgg{i+1}.t7'
            vgg_enc.append(pytorch_lua_wrapper(path_enc))
            self.enc.append(encoder[i](vgg_enc[i]))

            path_dec = f'models/decoder{i+1}.t7'
            vgg_dec.append(pytorch_lua_wrapper(path_dec))
            self.dec.append(decoder[i](vgg_dec[i]))

    def GaussianOTBarycenter(self, content, styles, level):
        """
        Stages:
        1. Create Gaussians of content and style
        2. Learn Barycenter map using Gaussians
        3. Apply barycenter map to the content
        """
        # assume one style as a start:
        d = content.shape[1]
        n1 = content.shape[2]
        n2 = content.shape[3]

        content = content.reshape(-1, d)
        style = styles[0].reshape(-1, d)

        mean_c, cov_c = compute_gaussian_statistics(content)
        mean_s, cov_s = compute_gaussian_statistics(style)

        mean_list = [mean_c, mean_s]
        cov_list = [cov_c, cov_s]

        # Synthetic Gaussians
        # mean_c_, cov_c_ = compute_gaussian_statistics(content)
        # mean_c = torch.zeros_like(mean_c_)
        # mean_s = torch.ones_like(mean_c_)
        # cov_c = torch.eye(d)
        # cov_s = 10*torch.eye(d)
        # mean_list = [mean_c, mean_s]
        # cov_list = [cov_c, cov_s]

        lambda_list = [0.5,0.5]
        print(f'lambda_list: {lambda_list}')

        mean_bary, cov_bary = wasserstein_barycenter_multiple(mean_list, cov_list, lambda_list)

        out = apply_barycenter_transform(content, mean_c, cov_c, mean_bary, cov_bary)

        print(f'expectation deviation in level {level}: {(mean_c - mean_bary).norm()}, cov deviation {(cov_c-cov_bary).norm()} whole vec deviation is {torch.norm(out-content)}')

        out = out.reshape(1, d, n1, n2)
        return out


def compute_gaussian_statistics(tensor):
    """
    Compute the mean and covariance of the feature distribution.
    tensor shape is (n,d)
    """
    mean = torch.mean(tensor, dim=0)
    centered_features = tensor - mean
    covariance = torch.matmul(centered_features.T, centered_features) / (tensor.size(0) - 1)
    return mean, covariance


def wasserstein_barycenter_multiple(mean_list, cov_list, lambda_list, max_iter=50):
    """
    Compute the Wasserstein barycenter of multiple Gaussian distributions using the fixed-point method.
    Fixed point operations rely on truncated SVD method
    """
    # Compute barycenter mean
    mean_bary = sum(lmb * mean for lmb, mean in zip(lambda_list, mean_list))

    # Initialize barycenter covariance
    j0 = torch.argmax(torch.tensor(lambda_list))
    cov_bary = cov_list[j0].clone()

    # Sigma_old = 0.5 * (cov_bary + cov_bary.T)  # intialization at higher lambda covariance
    print('d')
    Sigma_old = cov_bary
    # We'll record iteration differences for debugging
    history = []

    for it in range(max_iter):
        # 2a) Compute Sigma_{l-1}^{1/2} and Sigma_{l-1}^{-1/2}
        Sigma_old_sqrt, Sigma_old_inv_sqrt = matrix_sqrt_inv_sqrt(Sigma_old)

        # 2b) Build the sum over j
        Sigma_new = torch.zeros_like(Sigma_old)
        for j, lam_j in enumerate(lambda_list):
            # Inside: (Sigma_{l-1}^{1/2} Cov_j Sigma_{l-1}^{1/2})^(1/2)
            middle = Sigma_old_sqrt @ cov_list[j] @ Sigma_old_sqrt
            middle_sqrt, _ = matrix_sqrt_inv_sqrt(middle)
            # Calculate
            term = Sigma_old_inv_sqrt @ middle_sqrt @ Sigma_old_inv_sqrt
            Sigma_new += lam_j * term

        # 3) Check for convergence
        diff = torch.norm(Sigma_new - Sigma_old, p='fro').item()
        diff_init = torch.norm(Sigma_new - cov_bary, p='fro').item()
        history.append((it, diff))

        print (f'finished iteration {it}, diff  from t-1 {diff}, diff from content cov {diff_init}')

        Sigma_old = 0.5 * (Sigma_new + Sigma_new.T)  # force symmetry
        # Sigma_old = 0.5 * (Sigma_new + Sigma_new.T) + torch.eye(Sigma_old.shape[0], device=Sigma_old.device) * 1e-6
        # Sigma_old = torch.triu(Sigma_new) + torch.triu(Sigma_new, diagonal=1).T
        # Sigma_old = Sigma_new  # force symmetry

    return mean_bary, Sigma_old

def matrix_sqrt_inv_sqrt(Sigma, eps=1e-6):
    r"""
    Computes \Sigma^{1/2} and \Sigma^{-1/2} via truncated SVD.

    Parameters
    ----------
    Sigma : torch.Tensor
        A (d x d) symmetric positive semi-definite matrix.
    eps : float
        Threshold for truncating small singular values.
    max_rank : int, optional
        Maximum rank for truncated SVD.

    Returns
    -------
    Sigma_sqrt : torch.Tensor (d x d)
        Approximate square root of Sigma.
    Sigma_inv_sqrt : torch.Tensor (d x d)
        Approximate inverse square root of Sigma.
    """
    # 1) Perform truncated SVD
    U, S, V = truncated_svd(Sigma, eps=eps)
    # U,S,V = torch.linalg.svd(Sigma)
    # V = V.T

    # 2) Compute sqrt(S) and 1/sqrt(S)
    S_sqrt = torch.sqrt(S)  # shape: (k,)
    S_inv_sqrt = 1.0 / S_sqrt  # shape: (k,)

    # 3) Reconstruct Sigma^(1/2) = U * diag(S_sqrt) * V^T
    Sigma_sqrt = (U * S_sqrt) @ V.T

    # 4) Reconstruct Sigma^(-1/2) = U * diag(1/sqrt(S)) * V^T
    Sigma_inv_sqrt = (U * S_inv_sqrt) @ V.T

    return Sigma_sqrt, Sigma_inv_sqrt

def truncated_svd(matrix,max_rank=None,eps = 1e-6):

    # 1) Compute SVD (use 'full_matrices=False' for the "economy" SVD)
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # 2) Clamp small singular values
    S = torch.clamp(S, min=eps)

    # 3) If a max_rank is given, keep only top 'max_rank' singular values
    if max_rank is not None:
        r = min(max_rank, S.size(0))  # S.size(0) is the total number of singular values
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]
    else:
        # keep them all, but they are already clamped by eps
        pass

    return U, S, Vh.T

def apply_barycenter_transform(x, mu_c, cov_c, mu_b, cov_b):
    Sigma_c_sqrt, Sigma_c_inv_sqrt = matrix_sqrt_inv_sqrt(cov_c)
    middle = Sigma_c_sqrt @ cov_b @ Sigma_c_sqrt
    middle_sqrt, _ = matrix_sqrt_inv_sqrt(middle)
    M = Sigma_c_inv_sqrt @ middle_sqrt @ Sigma_c_inv_sqrt

    x_centered = x - mu_c
    T_x = x_centered @ M.T  # Use matrix multiplication directly
    T_x = T_x + mu_b

    return T_x







def truncated_svd_(matrix, eps=1e-6,):
    U, S, Vh = torch.linalg.svd(matrix)

    # Filter by singular value magnitude
    keep = S > eps
    keep_indices = keep.nonzero(as_tuple=True)[0]

    # Now pick the top 'k' singular values/vectors
    U_trunc = U[:, keep_indices]  # shape: (d x k)
    S_trunc = S[keep_indices]  # shape: (k,)
    V_trunc = Vh[keep_indices, :].T  # shape: (d x k) because Vh is (d x d)

    return U_trunc, S_trunc, V_trunc

#############
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # data is of shape (k, d, n1, n2)
        self.data = data
        self.k, self.d, self.n1, self.n2 = data.shape

        # Flatten the last two dimensions to easily index samples
        self.n = self.n1 * self.n2

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Convert the flattened index back to (i1, i2)
        i1 = idx // self.n2
        i2 = idx % self.n2

        # Get the sample corresponding to indices (i1, i2)
        sample = self.data[:, :, i1, i2]  # Shape: (k, d)

        # Return both the sample and its corresponding indices (i1, i2)
        return sample, (i1, i2)

class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = pytorch_lua_wrapper('models/vgg1.t7')
        decoder1_torch = pytorch_lua_wrapper('models/decoder1.t7')
        vgg2 = pytorch_lua_wrapper('models/vgg2.t7')
        decoder2_torch = pytorch_lua_wrapper('models/decoder2.t7')
        vgg3 = pytorch_lua_wrapper('models/vgg3.t7')
        decoder3_torch = pytorch_lua_wrapper('models/decoder3.t7')
        vgg4 = pytorch_lua_wrapper('models/vgg4.t7')
        decoder4_torch = pytorch_lua_wrapper('models/decoder4.t7')
        vgg5 = pytorch_lua_wrapper('models/vgg5.t7')
        decoder5_torch = pytorch_lua_wrapper('models/decoder5.t7')


        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self,cF,sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF,1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        c_u,c_e,c_v = torch.svd(contentConv,some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
        s_u,s_e,s_v = torch.svd(styleConv,some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
        step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
        whiten_cF = torch.mm(step2,cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        # sF = cF.clone()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        # csF.data.resize_(ccsF.size()).copy_(ccsF)
        csF = ccsF.clone()
        return csF