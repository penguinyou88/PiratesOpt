import torch
import gpytorch
import botorch
import matplotlib
from matplotlib import pyplot as plt

from time import sleep

#set dtype
torch.set_default_dtype(torch.float64)

#
font = {'weight' : 'bold',
        'size'   : 22}
plt.rc('font', **font)

from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, LinearKernel, PeriodicKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.sampling.normal import SobolEngine
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound

# BO torch models
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP
from botorch.optim import optimize_acqf

#Get Periodic, regular, and Heteroskedastic GP models

def get_trained_GP(X,Y,Yvar = None, kernel_type='RBF'):

    X_torch = X
    f_X_torch = Y.flatten()

    nu = 2.5 # Parameter for the Matern kernel

    input_dim = X_torch.shape[1]
    output_dim = Y.shape[1]

    # If single dimensional problem increase the dimension
    if input_dim == 1:
        X_torch = X_torch.unsqueeze(-1)

    if output_dim == 1:
        f_X_torch = f_X_torch.unsqueeze(-1)

    # Standard scaling output
    standardize = Standardize(m=output_dim)
    outcome_transform = standardize

    if kernel_type == "Heteroskedastic":
      model_ = HeteroskedasticSingleTaskGP(X_torch, X_torch, Yvar)
      mll = ExactMarginalLogLikelihood(model.likelihood, model)
      fit_gpytorch_model(mll)
    else:
      # covariance module: Select option
      if kernel_type == 'RBF':
          covar_module = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
      elif kernel_type == 'Linear':
          covar_module = ScaleKernel(LinearKernel(ard_num_dims=input_dim))
      elif kernel_type == 'Periodic':
          covar_module = ScaleKernel(PeriodicKernel(ard_num_dims=input_dim, period_length_constraint=Interval(2*torch.pi-1E-10, 2*torch.pi+1E-10),  ))
      elif kernel_type == 'Matern':
          covar_module = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=input_dim))

      # Set the likelihood
      likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=0, upper_bound=0.1e-5))

      # Define the model
      model = SingleTaskGP(train_X = X_torch, train_Y = f_X_torch,
                           covar_module=covar_module,
                           likelihood=likelihood,
                           input_transform = Normalize(2),
                           outcome_transform=outcome_transform)

      # call the training procedure
      model.outcome_transform.eval()
      mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
      fit_gpytorch_model(mll)

    return model



def opt_aqf_fun(model, bounds, x_train, y_train, max_bound):

  best_value = y_train.max()
  aqf = ExpectedImprovement(model=model, best_f=best_value)
  # # opt acq fun (continuous)
  # x_new, _ = optimize_acqf(acq_function=aqf,
  #                               bounds=bounds,q=1,
  #                               num_restarts=100,
  #                               raw_samples=10000,)
  # optimize over grid
  N = max_bound
  xg = yg = torch.linspace(1, N, N)
  X, Y = torch.meshgrid(xg, yg)
  x_grid = torch.cat([X.reshape(N**2, 1), Y.reshape(N**2, 1)], -1)
  with torch.no_grad():
    aqf_val = aqf(x_grid.unsqueeze(1))
  i_new = torch.argmax(aqf_val)
  x_new = x_grid[i_new,:].unsqueeze(0)
  # x_new = torch.round(x_new)
  # if x_new in x_train:
  #   x_new = torch.round(torch.rand(1, 2)*max_bound)

  return x_new

def generate_xtrue1(N):
  return torch.ceil(torch.rand(1, 2)*N)

def distance1(x, x_true):
  return torch.round( 10 * torch.sqrt( torch.sum((x - x_true)**2, dim=1) ) )

def return_scaling1(x_true,N):
    # get min, max distance
    xg = yg = torch.linspace(1, N, N)
    X, Y = torch.meshgrid(xg, yg)
    x_grid = torch.cat([X.reshape(N**2, 1), Y.reshape(N**2, 1)], -1)
    with torch.no_grad():
        dist_grid = distance1(x_grid, x_true)
    max_dist = torch.max(dist_grid)
    min_dist = torch.min(dist_grid)

    return min_dist, max_dist

# function evaluation
def func1(x, x_true, N, min_dist, max_dist):
    x = torch.round(x)
    f = (0 - 99)/(max_dist - min_dist) * (distance1(x, x_true) - min_dist) + 99
    
    return torch.round(f)

def generate_xtrue2(N):
  Ndata = 30
  X = torch.ceil(N*torch.rand((Ndata,2)))
  covar = ScaleKernel(RBFKernel(ard_num_dims=2))
  covar.base_kernel.lengthscale = 3
  covar.outputscale = 10
  K = covar(X, X, diag=True).detach().unsqueeze(-1)
  Y = torch.sqrt(K) * torch.randn((Ndata,1))

  return [X, Y]

def distance2(x, x_true):
  X, Y = x_true

  covar = ScaleKernel(RBFKernel(ard_num_dims=2))
  covar.base_kernel.lengthscale = 3
  covar.outputscale = 10
  likelihood = GaussianLikelihood(noise_constraint=Interval(lower_bound=1e-5, upper_bound=1e-3))

  standardize = Standardize(m=1)
  outcome_transform = standardize

  model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=covar, likelihood=likelihood, outcome_transform=outcome_transform)
  model.eval()

  with torch.no_grad():
    posterior = model(x)

  return torch.round( posterior.mean.detach() )

def return_scaling2(x_true,N):
    # get min, max distance
    xg = yg = torch.linspace(1, N, N)
    X, Y = torch.meshgrid(xg, yg)
    x_grid = torch.cat([X.reshape(N**2, 1), Y.reshape(N**2, 1)], -1)
    with torch.no_grad():
        dist_grid = distance2(x_grid, x_true)
    max_dist = torch.max(dist_grid)
    min_dist = torch.min(dist_grid)

    return min_dist, max_dist

# function evaluation
def func2(x, x_true, N, min_dist, max_dist):
    x = torch.round(x)
    f = (0 - 99)/(max_dist - min_dist) * (distance2(x, x_true) - min_dist) + 99
    
    return torch.round(f)

def BO_step(x_train, y_train, x_true,bounds,min_dist,max_dist,max_bound=10, func=func1):
  model = get_trained_GP(x_train.double(), y_train.double())
  x_new = opt_aqf_fun( model , bounds.double(), x_train, y_train, max_bound)
  y_new = func(x_new, x_true, max_bound, min_dist, max_dist).double()
  return x_new, y_new, x_train, y_train
