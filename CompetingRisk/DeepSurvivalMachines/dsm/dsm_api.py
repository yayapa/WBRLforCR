# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
This module is a wrapper around torch implementations and
provides a convenient API to train Deep Survival Machines.
"""

from ray import java_function
from dsm.dsm_torch import DeepSurvivalMachinesTorch
from dsm.dsm_torch import DeepRecurrentSurvivalMachinesTorch
from dsm.dsm_torch import DeepConvolutionalSurvivalMachinesTorch
from dsm.dsm_torch import DeepCNNRNNSurvivalMachinesTorch
from dsm.dsm_torch import DeepSurvivalMachinesMultiEncoderTorch

import dsm.losses as losses

from dsm.utilities import train_dsm
from dsm.utilities import _get_padded_features, _get_padded_targets
from dsm.utilities import _reshape_tensor_with_nans

import torch
import numpy as np
import pandas as pd
#from DeepSurvivalMachines.dsm.dsm_api import DeepSurvivalMachines
#from DeepSurvivalMachines.dsm.dsm_api import DeepSurvivalMachinesMultiEncoder

__pdoc__ = {}
__pdoc__["DeepSurvivalMachines.fit"] = True
#__pdoc__["DeepSurvivalMachinesMultiEncoder.fit"] = True
__pdoc__["DeepSurvivalMachines._eval_nll"] = True
__pdoc__["DeepConvolutionalSurvivalMachines._eval_nll"] = True
__pdoc__["DSMBase"] = False


class DSMBase():
  """Base Class for all DSM models"""

  def __init__(self, k=3, layers=None, distribution="Weibull",
               temp=1000., discount=1.0, cuda=False, fusion_outdim=None):
    self.k = k
    self.layers = layers
    self.dist = distribution
    self.temp = temp
    self.discount = discount
    self.fitted = False
    self.cuda = cuda # Two levels: 1 full GPU, 2 batch GPU (prefer 1 if fit on memory)
    self.fusion_outdim = fusion_outdim

  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepSurvivalMachinesTorch(inputdim,
                                     k=self.k,
                                     layers=self.layers,
                                     dist=self.dist,
                                     temp=self.temp,
                                     discount=self.discount,
                                     optimizer=optimizer,
                                     risks=risks)

  def fit(self, x, t, e, vsize=0.15, val_data=None,
          iters=1, learning_rate=1e-3, batch_size=100,
          elbo=True, optimizer="Adam", random_state=100):

    r"""This method is used to train an instance of the DSM model.

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = 1 \) means the event took place.
    vsize: float
        Amount of data to set aside as the validation set.
    val_data: tuple
        A tuple of the validation dataset. If passed vsize is ignored.
    iters: int
        The maximum number of training iterations on the training dataset.
    learning_rate: float
        The learning rate for the `Adam` optimizer.
    batch_size: int
        learning is performed on mini-batches of input data. this parameter
        specifies the size of each mini-batch.
    elbo: bool
        Whether to use the Evidence Lower Bound for optimization.
        Default is True.
    optimizer: str
        The choice of the gradient based optimization method. One of
        'Adam', 'RMSProp' or 'SGD'.
    random_state: float
        random seed that determines how the validation set is chosen.

    """

    processed_data = self._preprocess_training_data(x, t, e,
                                                    vsize, val_data,
                                                    random_state)
    x_train, t_train, e_train, x_val, t_val, e_val = processed_data

    #Todo: Change this somehow. The base design shouldn't depend on child
    if type(self).__name__ in ["DeepConvolutionalSurvivalMachines",
                               "DeepCNNRNNSurvivalMachines"]:
      inputdim = tuple(x_train.shape)[-2:]
    elif isinstance(x_train, dict):
      inputdim = {k: x_train[k].size(1) for k in x_train}
    else:
      inputdim = x_train.shape[-1]

    maxrisk = int(np.nanmax(e_train.cpu().numpy()))
    model = self._gen_torch_model(inputdim, optimizer, risks=maxrisk)

    if self.cuda:
      model = model.cuda() 
 
    model, _ = train_dsm(model,
                         x_train, t_train, e_train,
                         x_val, t_val, e_val,
                         n_iter=iters,
                         lr=learning_rate,
                         elbo=elbo,
                         bs=batch_size, cuda=self.cuda==2)

    self.torch_model = model.eval()
    self.fitted = True

    return self 


  def compute_nll(self, x, t, e):
    r"""This function computes the negative log likelihood of the given data.
    In case of competing risks, the negative log likelihoods are summed over
    the different events' type.

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: np.ndarray
        A numpy array of the event/censoring times, \( t \).
    e: np.ndarray
        A numpy array of the event/censoring indicators, \( \delta \).
        \( \delta = r \) means the event r took place.

    Returns:
      float: Negative log likelihood.
    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, 0, None, 0)
    _, _, _, x_val, t_val, e_val = processed_data
    x_val, t_val, e_val = x_val,\
        _reshape_tensor_with_nans(t_val),\
        _reshape_tensor_with_nans(e_val)

    with torch.no_grad():
        if self.cuda == 2:
          # Data need to be on GPU if loss computed
          x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

        loss = 0
        for r in range(self.torch_model.risks):
          loss += float(losses.conditional_loss(self.torch_model,
                        x_val, t_val, e_val, elbo=False,
                        risk=str(r+1)).item())
    return loss

  def _preprocess_test_data(self, x):
    if isinstance(x, dict):
      if isinstance(x[next(iter(x))], pd.DataFrame):
        data = {mod: torch.from_numpy(x[mod].to_numpy()) for mod in x}
      else:
        data = {mod: torch.from_numpy(x[mod]) for mod in x}
    else:
      if isinstance(x, pd.DataFrame):
        data = torch.from_numpy(x.to_numpy())
      else:
        data = torch.from_numpy(x)
    if self.cuda:
      if isinstance(data, dict):
        data = {mod: arr.cuda() for mod, arr in data.items()}
      else:
        data = data.cuda()
    return data

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_state):
        # Determine number of samples.
    if isinstance(x, dict):
        # Assume all modalities have the same number of rows.
        n_samples = next(iter(x.values())).shape[0]
    else:
        n_samples = x.shape[0]

    idx = list(range(n_samples))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    
    #x_train, t_train, e_train = x[idx], t[idx], e[idx]
    # Split x using the shuffled indices.
    if isinstance(x, dict):
        # For each modality, use .iloc to split rows.
        x_train = {mod: df[idx] for mod, df in x.items()}
    else:
        x_train = x[idx]
        
        
    t_train = t[idx]
    e_train = e[idx]

    #x_train = torch.from_numpy(x_train).double()
    #t_train = torch.from_numpy(t_train).double()
    #e_train = torch.from_numpy(e_train).double()
    
    # Convert x_train to numpy arrays.
    if isinstance(x_train, dict):
      if isinstance(x_train[next(iter(x_train))], pd.DataFrame):
        for mod in x_train:
            # Convert each modality DataFrame to a numpy array.
            x_train[mod] = x_train[mod].to_numpy()
    else:
      if isinstance(x_train, pd.DataFrame):
        x_train = x_train.to_numpy()
    
    #print("Shape in preprocess:", 
    #      {mod: arr.shape for mod, arr in x_train.items()} if isinstance(x_train, dict) else x_train.shape)

    # Convert to torch tensors.
    if isinstance(x_train, dict):
        for mod in x_train:
            x_train[mod] = torch.from_numpy(x_train[mod]).double()
    else:
        x_train = torch.from_numpy(x_train).double()
    
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

    if val_data is None:

      #vsize = int(vsize*x_train.shape[0])
      vsize = int(vsize*n_samples)
      if isinstance(x_train, dict):
            x_val = {mod: arr[-vsize:] for mod, arr in x_train.items()}
            x_train = {mod: arr[:-vsize] for mod, arr in x_train.items()}
      else:
            x_val = x_train[-vsize:]
            x_train = x_train[:-vsize]
            
      t_val = t_train[-vsize:]
      e_val = e_train[-vsize:]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]
      
      #x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

      #x_train = x_train[:-vsize]
      #t_train = t_train[:-vsize]
      #e_train = e_train[:-vsize]

    else:

      x_val, t_val, e_val = val_data
      
      #print("x_val", x_val)
      
      if isinstance(x_val, dict):
            for mod in x_val:
              if isinstance(x_val[mod], pd.DataFrame):
                x_val[mod] = x_val[mod].to_numpy()
      else:
        if not isinstance(x_val, torch.Tensor):
          if isinstance(x_val, pd.DataFrame):
            x_val = x_val.to_numpy()
      #x_val = x_val.to_numpy()

      #x_val = torch.from_numpy(x_val).double()
      
      if isinstance(x_val, dict):
            for mod in x_val:
              if not isinstance(x_val[mod], torch.Tensor):
                    x_val[mod] = torch.from_numpy(x_val[mod]).double()
      else:
            #x_val = x_val.to_numpy()
            x_val = torch.from_numpy(x_val).double()
      t_val = torch.from_numpy(t_val).double()
      e_val = torch.from_numpy(e_val).double()

    if self.cuda == 1:
        if isinstance(x_train, dict):
            x_train = {mod: arr.cuda() for mod, arr in x_train.items()}
            x_val = {mod: arr.cuda() for mod, arr in x_val.items()}
        else:
            x_train = x_train.cuda()
            x_val = x_val.cuda()
        t_train, e_train = t_train.cuda(), e_train.cuda()
        t_val, e_val = t_val.cuda(), e_val.cuda()
      #x_train, t_train, e_train = x_train.cuda(), t_train.cuda(), e_train.cuda()
      #x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    return (x_train, t_train, e_train,
            x_val, t_val, e_val)


  def predict_mean(self, x, risk=1):
    r"""Returns the mean Time-to-Event \( t \)

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    Returns:
      np.array: numpy array of the mean time to event.

    """
    
    if self.fitted:
      x = self._preprocess_test_data(x)
      scores = losses.predict_mean(self.torch_model, x, risk=str(risk))
      return scores
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_mean`.")
  def predict_risk(self, x, t, risk=1):
    r"""Returns the estimated risk of an event occuring before time \( t \)
      \( \widehat{\mathbb{P}}(T\leq t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the risks at each time in t.

    """

    if self.fitted:
      return 1-self.predict_survival(x, t, risk=str(risk))
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_risk`.")


  def predict_survival(self, x, t, risk=1):
    r"""Returns the estimated survival probability at time \( t \),
      \( \widehat{\mathbb{P}}(T > t|X) \) for some input data \( x \).

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which survival probability is
        to be computed
    Returns:
      np.array: numpy array of the survival probabilites at each time in t.

    """
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = losses.predict_cdf(self.torch_model, x, t, risk=str(risk))
      return np.exp(np.array(scores)).T
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_pdf(self, x, t, risk=1):
    r"""Returns the estimated pdf at time \( t \),
      \( \widehat{\mathbb{P}}(T = t|X) \) for some input data \( x \). 

    Parameters
    ----------
    x: np.ndarray
        A numpy array of the input features, \( x \).
    t: list or float
        a list or float of the times at which pdf is
        to be computed
    Returns:
      np.array: numpy array of the estimated pdf at each time in t.

    """
    x = self._preprocess_test_data(x)
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = losses.predict_pdf(self.torch_model, x, t, risk=str(risk))
      return np.exp(np.array(scores)).T
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")


class DeepSurvivalMachines(DSMBase):
  """A Deep Survival Machines model.

  This is the main interface to a Deep Survival Machines model.
  A model is instantiated with approporiate set of hyperparameters and
  fit on numpy arrays consisting of the features, event/censoring times
  and the event/censoring indicators.

  For full details on Deep Survival Machines, refer to our paper [1].

  References
  ----------
  [1] <a href="https://arxiv.org/abs/2003.01176">Deep Survival Machines:
  Fully Parametric Survival Regression and
  Representation Learning for Censored Data with Competing Risks."
  arXiv preprint arXiv:2003.01176 (2020)</a>

  Parameters
  ----------
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  distribution: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  Example
  -------
  >>> from dsm import DeepSurvivalMachines
  >>> model = DeepSurvivalMachines()
  >>> model.fit(x, t, e)

  """

  def __call__(self):
    if self.fitted:
      print("A fitted instance of the Deep Survival Machines model")
    else:
      print("An unfitted instance of the Deep Survival Machines model")

    print("Number of underlying distributions (k):", self.k)
    print("Hidden Layers:", self.layers)
    print("Distribution Choice:", self.dist)


class DeepSurvivalMachinesMultiEncoder(DeepSurvivalMachines):
  def _gen_torch_model(self, inputdim, optimizer, risks):
     model = DeepSurvivalMachinesMultiEncoderTorch(inputdim,
                                              k=self.k,
                                              layers=self.layers,
                                              fusion_outdim=self.fusion_outdim,
                                              dist=self.dist,
                                              temp=self.temp,
                                              discount=self.discount,
                                              optimizer=optimizer,
                                              risks=risks)
     if self.cuda > 0:
       model = model.cuda()
     return model
   

class DeepRecurrentSurvivalMachines(DSMBase):

  """The Deep Recurrent Survival Machines model to handle data with
  time-dependent covariates.

  For full details on Deep Recurrent Survival Machines, refer to our paper [1].
  
  References
  ----------
  [1] <a href="http://proceedings.mlr.press/v146/nagpal21a.html">
  Deep Parametric Time-to-Event Regression with Time-Varying Covariates 
  AAAI Spring Symposium on Survival Prediction</a>

  """

  def __init__(self, k=3, layers=None, hidden=None,
               distribution="Weibull", temp=1000., discount=1.0, typ="LSTM"):
    super(DeepRecurrentSurvivalMachines, self).__init__(k=k,
                                                        layers=layers,
                                                        distribution=distribution,
                                                        temp=temp,
                                                        discount=discount)
    self.hidden = hidden
    self.typ = typ
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepRecurrentSurvivalMachinesTorch(inputdim,
                                              k=self.k,
                                              layers=self.layers,
                                              hidden=self.hidden,
                                              dist=self.dist,
                                              temp=self.temp,
                                              discount=self.discount,
                                              optimizer=optimizer,
                                              typ=self.typ,
                                              risks=risks)

  def _preprocess_test_data(self, x):
    data = torch.from_numpy(_get_padded_features(x))
    if self.cuda:
      data = data.cuda()
    return data

  def _preprocess_training_data(self, x, t, e, vsize, val_data, random_state):
    """RNNs require different preprocessing for variable length sequences"""

    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)

    x = _get_padded_features(x)
    t = _get_padded_targets(t)
    e = _get_padded_targets(e)

    x_train, t_train, e_train = x[idx], t[idx], e[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])

      x_val, t_val, e_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:]

      x_train = x_train[:-vsize]
      t_train = t_train[:-vsize]
      e_train = e_train[:-vsize]

    else:

      x_val, t_val, e_val = val_data

      x_val = _get_padded_features(x_val)
      t_val = _get_padded_features(t_val)
      e_val = _get_padded_features(e_val)

      x_val = torch.from_numpy(x_val).double()
      t_val = torch.from_numpy(t_val).double()
      e_val = torch.from_numpy(e_val).double()

    if self.cuda == 1:
      x_train, t_train, e_train = x_train.cuda(), t_train.cuda(), e_train.cuda()
      x_val, t_val, e_val = x_val.cuda(), t_val.cuda(), e_val.cuda()

    return (x_train, t_train, e_train,
            x_val, t_val, e_val)


class DeepConvolutionalSurvivalMachines(DSMBase):
  """The Deep Convolutional Survival Machines model to handle data with
  image-based covariates.

  """

  def __init__(self, k=3, layers=None, hidden=None, 
               distribution="Weibull", temp=1000., discount=1.0, typ="ConvNet"):
    super(DeepConvolutionalSurvivalMachines, self).__init__(k=k,
                                                            distribution=distribution,
                                                            temp=temp,
                                                            discount=discount)
    self.hidden = hidden
    self.typ = typ
  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepConvolutionalSurvivalMachinesTorch(inputdim,
                                                  k=self.k,
                                                  hidden=self.hidden,
                                                  dist=self.dist,
                                                  temp=self.temp,
                                                  discount=self.discount,
                                                  optimizer=optimizer,
                                                  typ=self.typ,
                                                  risks=risks)


class DeepCNNRNNSurvivalMachines(DeepRecurrentSurvivalMachines):

  """The Deep CNN-RNN Survival Machines model to handle data with
  moving image streams.

  """

  def __init__(self, k=3, layers=None, hidden=None,
               distribution="Weibull", temp=1000., discount=1.0, typ="LSTM"):
    super(DeepCNNRNNSurvivalMachines, self).__init__(k=k,
                                                     layers=layers,
                                                     distribution=distribution,
                                                     temp=temp,
                                                     discount=discount)
    self.hidden = hidden
    self.typ = typ

  def _gen_torch_model(self, inputdim, optimizer, risks):
    """Helper function to return a torch model."""
    return DeepCNNRNNSurvivalMachinesTorch(inputdim,
                                           k=self.k,
                                           layers=self.layers,
                                           hidden=self.hidden,
                                           dist=self.dist,
                                           temp=self.temp,
                                           discount=self.discount,
                                           optimizer=optimizer,
                                           typ=self.typ,
                                           risks=risks)


