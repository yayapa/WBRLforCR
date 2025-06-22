from operator import index
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

def get_optimizer(models, lr, optimizer, **kwargs):
	parameters = list(models.parameters())

	if optimizer == 'Adam':
		return torch.optim.Adam(parameters, lr=lr, **kwargs)
	elif optimizer == 'SGD':
		return torch.optim.SGD(parameters, lr=lr, **kwargs)
	elif optimizer == 'RMSProp':
		return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
	else:
		raise NotImplementedError('Optimizer '+optimizer+' is not implemented')

def train_nfg(model, total_loss,
			  x_train, t_train, e_train,
			  x_valid, t_valid, e_valid,
			  n_iter = 1000, lr = 1e-3, weight_decay = 0.001,
			  bs = 100, patience_max = 3, cuda = False, contrastive_weight = 0.0):
	# Separate oprimizer as one might need more time to converge
	optimizer = get_optimizer(model, lr, model.optimizer, weight_decay = weight_decay)
	
	patience, best_loss, previous_loss = 0, np.inf, np.inf
	best_param = deepcopy(model.state_dict())
	
	if (isinstance(x_train, dict)):
		n_samples = x_train[list(x_train.keys())[0]].shape[0]
	else:
		n_samples = x_train.shape[0]
	
	#if isinstance(x_train, dict):
#		nbatches = int(x_train[list(x_train.keys())[0]].shape[0]/bs) + 1
#	else:
	#nbatches = int(x_train.shape[0]/bs) + 1
	nbatches = int(n_samples/bs) + 1
 
  
	
	#index = np.arange(len(x_train))
	index = np.arange(n_samples)
 
	t_bar = tqdm(range(n_iter))
	for i in t_bar:
		np.random.shuffle(index)
		model.train()
		
		# Train survival model
		for j in range(nbatches):
			#xb = x_train[index[j*bs:(j+1)*bs]]
			if isinstance(x_train, dict):
				xb = {mod: tensor[index[j*bs:(j+1)*bs]] for mod, tensor in x_train.items()}
			else:
				xb = x_train[index[j*bs:(j+1)*bs]]

			tb = t_train[index[j*bs:(j+1)*bs]]
			eb = e_train[index[j*bs:(j+1)*bs]]
   
			# Skip empty mini-batch.
			if xb[list(xb.keys())[0]].shape[0] == 0 if isinstance(xb, dict) else xb.shape[0] == 0:
				continue
			
			
   			#if xb.shape[0] == 0:
			#		continue
			
			if cuda:
				if isinstance(xb, dict):
					xb = {mod: tensor.cuda() for mod, tensor in xb.items()}
				else:
					xb = xb.cuda()
				tb, eb = tb.cuda(), eb.cuda()
    
				#xb, tb, eb = xb.cuda(), tb.cuda(), eb.cuda()

			optimizer.zero_grad()
			loss = total_loss(model,
							  xb,
							  tb,
							  eb,
							  contrastive_weight
         			) 
			loss.backward()
			optimizer.step()

		model.eval()
		xb, tb, eb = x_valid, t_valid, e_valid
		if cuda:
			if isinstance(xb, dict):
				xb = {mod: tensor.cuda() for mod, tensor in xb.items()}
			else:
				xb = xb.cuda()
			tb, eb = tb.cuda(), eb.cuda()
			#xb, tb, eb  = xb.cuda(), tb.cuda(), eb.cuda()

		valid_loss = total_loss(model,
								xb,
								tb,
								eb,
        						contrastive_weight
        ).item() 
		t_bar.set_description("Loss: {:.3f}".format(valid_loss))
		if valid_loss < previous_loss:
			patience = 0

			if valid_loss < best_loss:
				best_loss = valid_loss
				best_param = deepcopy(model.state_dict())

		elif patience == patience_max:
			break
		else:
			patience += 1

		previous_loss = valid_loss

	model.load_state_dict(best_param)
	return model, i

