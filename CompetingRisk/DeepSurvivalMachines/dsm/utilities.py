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


"""Utility functions to train the Deep Survival Machines models"""

from dsm.dsm_torch import DeepSurvivalMachinesTorch
from dsm.losses import unconditional_loss, conditional_loss

from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np

import gc
import logging


def get_optimizer(model, lr):

  if model.optimizer == 'Adam':
    return torch.optim.Adam(model.parameters(), lr=lr)
  elif model.optimizer == 'SGD':
    return torch.optim.SGD(model.parameters(), lr=lr)
  elif model.optimizer == 'RMSProp':
    return torch.optim.RMSprop(model.parameters(), lr=lr)
  else:
    raise NotImplementedError('Optimizer '+model.optimizer+
                              ' is not implemented')

def pretrain_dsm(model, t_train, e_train, t_valid, e_valid,
                 n_iter=10000, lr=1e-2, thres=1e-4, cuda = False):

  premodel = DeepSurvivalMachinesTorch(1, 1,
                                       dist=model.dist,
                                       risks=model.risks,
                                       optimizer=model.optimizer).double()

  if cuda:
    premodel.cuda()
    t_train, e_train = t_train.cuda(), e_train.cuda()
    t_valid, e_valid = t_valid.cuda(), e_valid.cuda()

  optimizer = get_optimizer(premodel, lr)

  oldcost = float('inf')
  patience = 0
  costs = []
  for _ in tqdm(range(n_iter)):

    optimizer.zero_grad()
    loss = 0
    for r in range(model.risks):
      loss += unconditional_loss(premodel, t_train, e_train, str(r+1))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      valid_loss = 0
      for r in range(model.risks):
        valid_loss += unconditional_loss(premodel, t_valid, e_valid, str(r+1))
      valid_loss = valid_loss.detach().cpu().numpy()
      costs.append(valid_loss)
      #print(valid_loss)
      if np.abs(costs[-1] - oldcost) < thres:
        patience += 1
        if patience == 3:
          break
      oldcost = costs[-1]

  return premodel

def _reshape_tensor_with_nans(data):
  """Helper function to unroll padded RNN inputs."""
  data = data.reshape(-1)
  return data[~torch.isnan(data)]

def _get_padded_features(x):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(x_) for x_ in x])
  padx = []
  for i in range(len(x)):
    pads = np.nan*np.ones((d - len(x[i]),) + x[i].shape[1:])
    padx.append(np.concatenate([x[i], pads]))
  return np.array(padx)

def _get_padded_targets(t):
  """Helper function to pad variable length RNN inputs with nans."""
  d = max([len(t_) for t_ in t])
  padt = []
  for i in range(len(t)):
    pads = np.nan*np.ones(d - len(t[i]))
    padt.append(np.concatenate([t[i], pads]))
  return np.array(padt)[:, :, np.newaxis]

# ──────────────────────────────────────────────────────────────────────
# helper ─ detach ALL CUDA tensors held by a DeepSurvivalMachines model
# ──────────────────────────────────────────────────────────────────────
def _to_cpu_cache(dsm):
    """
    Move a DeepSurvivalMachines instance completely to CPU and delete
    the cached training / validation batches that train_dsm() stores.
    """
    if hasattr(dsm, "cpu"):               # DSM ≥ 0.2.2 implements .cpu()
        dsm.cpu()
    elif hasattr(dsm, "torch_model"):     # older versions
        dsm.torch_model.cpu()

    for attr in ("_data_cache", "_val_cache"):
        if hasattr(dsm, attr):
            delattr(dsm, attr)

"""
def train_dsm(model,
              x_train, t_train, e_train,
              x_valid, t_valid, e_valid,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100, cuda=False):
  logging.info('Pretraining the Underlying Distributions...')
  # For padded variable length sequences we first unroll the input and
  # mask out the padded nans.
  t_train_ = _reshape_tensor_with_nans(t_train)
  e_train_ = _reshape_tensor_with_nans(e_train)
  t_valid_ = _reshape_tensor_with_nans(t_valid)
  e_valid_ = _reshape_tensor_with_nans(e_valid)

  premodel = pretrain_dsm(model,
                          t_train_,
                          e_train_,
                          t_valid_,
                          e_valid_,
                          n_iter=10000,
                          lr=1e-2,
                          thres=1e-4, cuda = cuda or t_train.is_cuda)

  for r in range(model.risks):
    model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
    model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))

  model.double()
  optimizer = get_optimizer(model, lr)

  patience = 0
  oldcost = float('inf')

  nbatches = int(x_train.shape[0]/bs)+1

  dics = []
  costs = []
  i = 0
  for i in tqdm(range(n_iter)):
    for j in range(nbatches):

      xb = x_train[j*bs:(j+1)*bs]
      tb = t_train[j*bs:(j+1)*bs]
      eb = e_train[j*bs:(j+1)*bs]

      if xb.shape[0] == 0:
        continue

      if cuda:
        xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

      optimizer.zero_grad()
      loss = 0
      for r in range(model.risks):
        loss += conditional_loss(model,
                                 xb,
                                 _reshape_tensor_with_nans(tb),
                                 _reshape_tensor_with_nans(eb),
                                 elbo=elbo,
                                 risk=str(r+1))
      #print ("Train Loss:", float(loss))
      loss.backward()
      optimizer.step()

    valid_loss = 0
    for r in range(model.risks):
      if cuda:
        x_valid, t_valid_, e_valid_ = x_valid.cuda(), t_valid_.cuda(), e_valid_.cuda()

      valid_loss += conditional_loss(model,
                                     x_valid,
                                     t_valid_,
                                     e_valid_,
                                     elbo=False,
                                     risk=str(r+1))

    valid_loss = valid_loss.detach().cpu().numpy()
    costs.append(float(valid_loss))
    dics.append(deepcopy(model.state_dict()))

    if costs[-1] >= oldcost:
      if patience == 2:
        minm = np.argmin(costs)
        model.load_state_dict(dics[minm])

        del dics
        gc.collect()

        return model, i
      else:
        patience += 1
    else:
      patience = 0

    oldcost = costs[-1]

  minm = np.argmin(costs)
  model.load_state_dict(dics[minm])
  print("best cost:", costs[minm])

  del dics
  gc.collect()

  return model, i
"""
def train_dsm(model,
              x_train, t_train, e_train,
              x_valid, t_valid, e_valid,
              n_iter=10_000, lr=1e-3, elbo=True,
              bs=100, cuda=False):


    logging.info("Pre-training the underlying distributions …")

    # ── flatten padded RNN inputs ─────────────────────────────────────
    t_train_ = _reshape_tensor_with_nans(t_train)
    e_train_ = _reshape_tensor_with_nans(e_train)
    t_valid_ = _reshape_tensor_with_nans(t_valid)
    e_valid_ = _reshape_tensor_with_nans(e_valid)

    # ── pre-train the per-risk base distributions (1-D DSM) ──────────
    premodel = pretrain_dsm(model,
                            t_train_, e_train_,
                            t_valid_, e_valid_,
                            n_iter=10_000, lr=1e-2, thres=1e-4,
                            cuda=cuda or t_train.is_cuda)

    # copy pre-training parameters into the full model
    for r in range(model.risks):
        model.shape[str(r + 1)].data.fill_(float(premodel.shape[str(r + 1)]))
        model.scale[str(r + 1)].data.fill_(float(premodel.scale[str(r + 1)]))

    model.double()
    optimizer = get_optimizer(model, lr)

    # ── dataset sizes / batches ───────────────────────────────────────
    n_samples = (x_train[list(x_train.keys())[0]].shape[0]
                 if isinstance(x_train, dict) else x_train.shape[0])
    n_batches = n_samples // bs + 1

    # ── pin validation tensors to GPU once (optional but cheaper) ─────
    if cuda:
        if isinstance(x_valid, dict):
            x_valid = {m: t.cuda() for m, t in x_valid.items()}
        else:
            x_valid = x_valid.cuda()
        t_valid_, e_valid_ = t_valid_.cuda(), e_valid_.cuda()

    # ── early-stopping bookkeeping ────────────────────────────────────
    best_cost      = float("inf")
    best_state_cpu = None
    patience       = 0

    # ─────────────────────── EPOCH LOOP ───────────────────────────────
    for epoch in tqdm(range(n_iter)):
        # -- training over mini-batches --
        for b in range(n_batches):

            # slice batch
            if isinstance(x_train, dict):
                xb = {m: t[b*bs:(b+1)*bs] for m, t in x_train.items()}
            else:
                xb = x_train[b*bs:(b+1)*bs]
            tb = t_train[b*bs:(b+1)*bs]
            eb = e_train[b*bs:(b+1)*bs]

            if (xb[list(xb.keys())[0]].shape[0] == 0
                    if isinstance(xb, dict) else xb.shape[0] == 0):
                continue

            # move to GPU if requested
            if cuda:
                if isinstance(xb, dict):
                    xb = {m: t.cuda() for m, t in xb.items()}
                else:
                    xb = xb.cuda()
                tb, eb = tb.cuda(), eb.cuda()

            # forward / backward
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for r in range(model.risks):
                loss += conditional_loss(
                    model,
                    xb,
                    _reshape_tensor_with_nans(tb),
                    _reshape_tensor_with_nans(eb),
                    elbo=elbo,
                    risk=str(r + 1),
                )
            loss.backward()
            optimizer.step()

        # -- validation (no graph) --
        with torch.no_grad():
            val_loss = 0.0
            for r in range(model.risks):
                val_loss += conditional_loss(
                    model,
                    x_valid, t_valid_, e_valid_,
                    elbo=False, risk=str(r + 1)
                )
            val_loss = float(val_loss.cpu())

        # -- early stopping --
        if val_loss < best_cost:
            best_cost = val_loss
            best_state_cpu = {k: v.detach().cpu()
                              for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience == 2:       # same patience setting as original
                break

    # ── restore best weights and clean GPU memory ─────────────────────
    if best_state_cpu is not None:
        model.load_state_dict(best_state_cpu)

    #_to_cpu_cache(model)            # drop hidden cached batches
    torch.cuda.empty_cache()
    gc.collect()

    return model, epoch

"""
def train_dsm(model,
              x_train, t_train, e_train,
              x_valid, t_valid, e_valid,
              n_iter=10000, lr=1e-3, elbo=True,
              bs=100, cuda=False):



  logging.info('Pretraining the Underlying Distributions...')
  # For padded variable length sequences we first unroll the input and
  # mask out the padded nans.
  t_train_ = _reshape_tensor_with_nans(t_train)
  e_train_ = _reshape_tensor_with_nans(e_train)
  t_valid_ = _reshape_tensor_with_nans(t_valid)
  e_valid_ = _reshape_tensor_with_nans(e_valid)

  premodel = pretrain_dsm(model,
                          t_train_,
                          e_train_,
                          t_valid_,
                          e_valid_,
                          n_iter=10000,
                          lr=1e-2,
                          thres=1e-4, cuda = cuda or t_train.is_cuda)

  for r in range(model.risks):
    model.shape[str(r+1)].data.fill_(float(premodel.shape[str(r+1)]))
    model.scale[str(r+1)].data.fill_(float(premodel.scale[str(r+1)]))

  model.double()
  optimizer = get_optimizer(model, lr)

  patience = 0
  oldcost = float('inf')

  if isinstance(x_train, dict):
    n_samples = x_train[list(x_train.keys())[0]].shape[0]
  else:
    n_samples = x_train.shape[0]
  nbatches = int(n_samples/bs)+1
  #nbatches = int(x_train.shape[0]/bs)+1

  dics = []
  costs = []
  i = 0
  
  best_cost      = float('inf')
  best_state_cpu = None

  
  if cuda:
    if (isinstance(x_valid, dict)):
      x_valid = {mod: tensor.cuda() for mod, tensor in x_valid.items()}
    else:
      x_valid = x_valid.cuda()
    t_valid_, e_valid_ = t_valid_.cuda(), e_valid_.cuda()
  
  for i in tqdm(range(n_iter)):
    for j in range(nbatches):
      
      if (isinstance(x_train, dict)):
        xb = {mod: tensor[j*bs:(j+1)*bs] for mod, tensor in x_train.items()}
      else:
        xb = x_train[j*bs:(j+1)*bs]
      tb = t_train[j*bs:(j+1)*bs]
      eb = e_train[j*bs:(j+1)*bs]

      if xb[list(xb.keys())[0]].shape[0] == 0 if isinstance(xb, dict) else xb.shape[0] == 0:
        continue
      #if xb.shape[0] == 0:
      #  continue

      if cuda: 
        if isinstance(xb, dict):
          xb = {mod: tensor.cuda() for mod, tensor in xb.items()}
        else:
          xb = xb.cuda()
        tb, eb = tb.cuda(), eb.cuda()
      #if cuda:
      #  xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

      optimizer.zero_grad()
      loss = 0
      for r in range(model.risks):
        loss += conditional_loss(model,
                                 xb,
                                 _reshape_tensor_with_nans(tb),
                                 _reshape_tensor_with_nans(eb),
                                 elbo=elbo,
                                 risk=str(r+1))
      #print ("Train Loss:", float(loss))
      loss.backward()
      optimizer.step()
    with torch.no_grad():
        
      valid_loss = 0
      for r in range(model.risks):

        #if cuda:
        #  x_valid, t_valid_, e_valid_ = x_valid.cuda(), t_valid_.cuda(), e_valid_.cuda()

        valid_loss += conditional_loss(model,
                                      x_valid,
                                      t_valid_,
                                      e_valid_,
                                      elbo=False,
                                      risk=str(r+1))

      valid_loss = valid_loss.detach().cpu().numpy()
      if valid_loss < best_cost:
          best_cost      = valid_loss
          best_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
          patience = 0
      else:
          patience += 1
          if patience == 2:          # same as before
              break
  # ---------------------------------------

  if best_state_cpu is not None:
      model.load_state_dict(best_state_cpu)

  return model, i


      costs.append(float(valid_loss))
      dics.append(deepcopy(model.state_dict()))

      if costs[-1] >= oldcost:
        if patience == 2:
          minm = np.argmin(costs)
          model.load_state_dict(dics[minm])

          del dics
          gc.collect()

          return model, i
        else:
          patience += 1
      else:
        patience = 0

      oldcost = costs[-1]

  minm = np.argmin(costs)
  model.load_state_dict(dics[minm])

  del dics
  gc.collect()

  return model, i
  """