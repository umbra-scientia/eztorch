"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import io
import os
import sys
import time
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

VERSION="1.0"

def Positive(t, epsilon=1e-16):
	return torch.maximum(t, torch.tensor(epsilon, device=t.device))

def SoftClamp(x, limit=2):
	k = limit - 1
	y = torch.minimum(x, torch.tanh(x - k) + k)
	y = torch.maximum(y, torch.tanh(x + k) - k)
	return y

def UnitSphere(x):
	if x == None: return None
	return x / Positive(torch.norm(x, dim=-1, keepdim=True))

def UnitVariance(x):
	if x == None: return None
	vari = torch.sum(x*x, dim=-1, keepdim=True) / max(x.shape[-1]-1, 1)
	return x / torch.sqrt(Positive(vari))

def Param(shape, mean=0, variance=1, device='cpu'):
	return nn.Parameter(torch.randn(shape, device=device, dtype=torch.float32) * math.sqrt(variance) + mean)

def Matrix(in_res, out_res, bias=False, device='cpu'):
	m = nn.Linear(in_res, out_res, bias=bias, device=device)
	m.weight.data.normal_(0, 1)
	m.weight.data /= math.sqrt(in_res)
	if bias: m.bias.data.zero_()
	return m

def Join(x, axis=-1):
	y = []
	for i in x:
		if i == None: continue
		y.append(i)
	if len(y) == 0: return None
	if len(y) == 1: return y[0]
	return torch.cat(y, axis=axis)

class CausalConv1d(nn.Conv1d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, device='cpu', **kwargs):
		self.__padding = (kernel_size - 1) * dilation
		super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias, device=device)

	def forward(self, input):
		result = super(CausalConv1d, self).forward(input)
		if self.__padding != 0:
			return result[:, :, :-self.__padding]
		return result

class VarianceNorm(nn.Module):
	def __init__(self, scale=1, device='cpu'):
		super(VarianceNorm, self).__init__()
		self.variance = nn.Parameter(torch.tensor(np.array(scale), dtype=torch.float32, device=device))
	def forward(self, x):
		return UnitVariance(x) * self.variance

def L1WeightDecay(x):
	return x - torch.sign(x)

def L2WeightDecayToZero(x):
	return x*0

def L2WeightDecayToUnit(x):
	return UnitSphere(x)

class Optimizer():
	def __init__(self, t, regularizer=L2WeightDecayToUnit):
		self.regularizer = regularizer
		self.grad_avg = True
		self.grad_var = True
		
		if self.grad_avg:
			self.grad_avg = torch.zeros(t.shape, device=t.device)
		if self.grad_var:
			self.grad_var = torch.zeros(t.shape, device=t.device)
			self.grad_var_comp = 0

	def update(self, t, learn_rate, alpha, weight_decay):
		alpha2 = alpha * alpha
		update_mask = torch.sign(torch.norm(t.grad, dim=-1, keepdim=True))
		gradient_step = t.grad
		if self.grad_avg != None:
			self.grad_avg = self.grad_avg + alpha*(t.grad - self.grad_avg) * update_mask
			gradient_step = self.grad_avg
		if self.grad_var != None:
			surprise = t.grad
			if self.grad_avg != None:
				surprise = surprise - self.grad_avg
			self.grad_var = self.grad_var + alpha2*(surprise*surprise - self.grad_var) * update_mask
			self.grad_var_comp = self.grad_var_comp + alpha2*(1 - self.grad_var_comp)
			gradient_step = gradient_step / torch.sqrt(Positive(self.grad_var / self.grad_var_comp))
		if self.regularizer != None:
			gradient_step = gradient_step + weight_decay * (t.data - self.regularizer(t.data))
		t.data -= learn_rate * gradient_step * update_mask

	def set_length(self, x):
		if x == True: x = 1
		if not x: x = 0
		self.length = x

def GetOptimizer(t):
	if not hasattr(t, "EzTorchOptimizer"):
		t.EzTorchOptimizer = Optimizer(t)
	return t.EzTorchOptimizer

def OptimizeTensor(t, learn_rate, alpha, weight_decay):
	opt = GetOptimizer(t)
	opt.update(t, learn_rate, alpha, weight_decay)
	t.grad = None

def OptimizeModule(mod, learn_rate, alpha, weight_decay, backprop_count=1):
	grad_scale = 1 / backprop_count
	for p in mod.parameters():
		if not p.requires_grad: continue
		if p.grad == None: continue
		p.grad *= grad_scale
		OptimizeTensor(p, learn_rate, alpha, weight_decay)

def save_param(p):
	x = {}
	if hasattr(p, "EzTorchOptimizer"):
		opt = p.EzTorchOptimizer
		x = {
			'grad_avg': opt.grad_avg,
			'grad_var': opt.grad_var,
			'grad_var_comp': opt.grad_var_comp
		}
	if p.grad != None:
		x['grad'] = p.grad
	x['data'] = p.data
	return x

def force_shape(p, t, x):
	if p.shape == x.shape:
		return x
	if len(p.shape) == 1:
		y = x[:min(p.shape[0], x.shape[0])]
		if t == None: t = torch.zeros(p.shape, device=p.device)
		t[:min(p.shape[0], x.shape[0])] = y
		return t
	if len(p.shape) == 2:
		y = x[:min(p.shape[0], x.shape[0]),:min(p.shape[1], x.shape[1])]
		if t == None: t = torch.zeros(p.shape, device=p.device)
		t[:min(p.shape[0], x.shape[0]),:min(p.shape[1], x.shape[1])] = y
		return t
	if len(p.shape) == 3:
		y = x[:min(p.shape[0], x.shape[0]),:min(p.shape[1], x.shape[1]),:min(p.shape[2], x.shape[2])]
		if t == None: t = torch.zeros(p.shape, device=p.device)
		t[:min(p.shape[0], x.shape[0]),:min(p.shape[1], x.shape[1]),:min(p.shape[2], x.shape[2])] = y
		return t
	print("Unable to force shapes:", p.shape, x.shape)
	exit()
	return x

def load_param(p, x):
	if "data" in x:
		p.data = force_shape(p, p.data, x['data'])
	if "grad" in x:
		p.grad = force_shape(p, p.grad, x['grad'])
	if "grad_avg" in x:
		opt = GetOptimizer(p)
		opt.grad_avg = force_shape(p, opt.grad_avg, x['grad_avg'])
	if "grad_var" in x:
		opt = GetOptimizer(p)
		opt.grad_var = force_shape(p, opt.grad_var, x['grad_var'])
		opt.grad_var_comp = x['grad_var_comp']

class Model(nn.Module):
	def __init__(self, device='cpu'):
		super(Model, self).__init__()
		self.device = device
		self.learn_rate = 0.001
		self.momentum = 0.9
		self.weight_decay = 0.01
		self.lr_decay = 0.001
		self.filename = None
		self.step = 0
		self.batch_losses = []
		self.losses = {}
		self.sample = 0
		self.time_offset = time.time()
		self.wandb = None
		self.wandb_id = None

	def elapsed_time(self):
		return time.time() - self.time_offset

	def wandb_init(self, config={}, wandb_lib=None, **kwargs):
		if wandb_lib == None:
			import wandb
			self.wandb = wandb
		else:
			self.wandb = wandb_lib
		resuming = True
		if self.wandb_id == None:
			self.wandb_id = self.wandb.util.generate_id()
			resuming = False
		config["Learning Rate"] = self.learn_rate
		config["Momentum"] = self.momentum
		config["Weight Decay"] = self.weight_decay
		config["LR Decay"] = self.lr_decay
		config["Parameters"] = self.param_count()
		config["EzTorch Version"] = VERSION
		self.wandb.init(id=self.wandb_id, resume=resuming, config=config, **kwargs)

	def save(self, fn, fc={}):
		parm = {}
		for k, v in self.named_parameters():
			parm[k] = save_param(v)
		fc['parm'] = parm
		fc['step'] = self.step
		fc['sample'] = self.sample
		fc['elapsed'] = time.time() - self.time_offset
		fc['batch_losses'] = self.batch_losses
		fc['wandb_id'] = self.wandb_id
		torch.save(fc, fn)
		self.filename = fn

	def load(self, fn, reject=None):
		try:
			f = torch.load(fn, map_location=torch.device(self.device))
		except:
			return False
		parm = {}
		if 'parm' in f: parm = f['parm']
		if type(parm) == type([]):
			i = 0
			for p in self.parameters():
				if i >= len(parm):
					print("WARNING: Only found %d parameters in file." % i)
					break
				load_param(p, parm[i])
				i = i + 1
		elif type(parm) == type({}):
			for k, v in self.named_parameters():
				if reject and k in reject: continue
				if k in parm:
					load_param(v, parm[k])
				else:
					print("WARNING: Missing parameter: %s" % k)
		else:
			return False
		if 'step' in f: self.step = f['step']
		if 'sample' in f: self.sample = f['sample']
		if 'batch_losses' in f:
			self.batch_losses = f['batch_losses']
			self.compute_losses_from_batch_()
		if 'wandb_id' in f: self.wandb_id = f['wandb_id']
		if 'elapsed' in f:
			self.time_offset = time.time() - f['elapsed']
		self.filename = fn
		return f

	def compute_losses_from_batch_(self):
		loss_total = {}
		loss_count = {}
		for losses in self.batch_losses:
			for name in losses:
				loss = losses[name]
				if name not in loss_total:
					loss_total[name] = loss
					loss_count[name] = 1
				else:
					loss_total[name] += loss
					loss_count[name] += 1
		loss = {}
		for name in loss_total:
			loss[name] = loss_total[name] / loss_count[name]
		self.losses = loss

	def backprop(self, losses, loss_weights=None):
		if type(losses) != type({}):
			if type(losses) != type([]):
				losses = [losses]
			new_losses = {}
			for i in range(len(losses)):
				generic_name = "loss"
				if i > 0: generic_name = "loss%d" % i
				new_losses[generic_name] = losses[i]
			losses = new_losses
		lossesf = {}
		backprop = 0
		for name in losses:
			loss = losses[name]
			if type(loss) == type(0):
				lossesf[name] = loss
			else:
				if torch.isnan(loss):
					raise ValueError("NaN loss")
					return len(self.batch_losses)
				lossesf[name] = loss.item()
			weight = 1
			if (loss_weights != None) and (name in loss_weights):
				weight = loss_weights[name]
			backprop = backprop + loss * weight
		backprop.backward()
		self.batch_losses.append(lossesf)
		self.compute_losses_from_batch_()
		return len(self.batch_losses)

	def optimize(self, aux_metrics={}):
		if self.wandb != None:
			aux_metrics["step"] = self.step
			if self.sample > 0:
				aux_metrics["sample"] = self.sample
			for ln in self.losses:
				aux_metrics[ln] = self.losses[ln]
			self.wandb.log(aux_metrics)
		lr = self.learn_rate
		if self.lr_decay > 0:
			lr = lr * math.pow(1-self.lr_decay, math.sqrt(self.step))
		OptimizeModule(self, lr, 1-self.momentum, self.weight_decay, len(self.batch_losses))
		self.batch_losses = []
		self.step += 1

	def progress_bar(self, cur_value, max_value, title=None, losses=None):
		w = 0
		try:
			w = os.get_terminal_size().columns
		except:
			w = 80
		if title == None:
			lhs = "Step %d [" % (self.step)
		else:
			lhs = title + " ["
		if losses == None:
			losses = ""
			for loss in self.losses:
				if losses != "":
					losses += ", "
				losses += " %s=%.04f" % (loss, self.losses[loss])
		rhs = "] (%d/%d)%s " % (cur_value, max_value, losses)
		w -= len(lhs)
		w -= len(rhs)
		wl = w
		if max_value > 0:
			wl = (w * cur_value) // max_value
		wl = min(wl, w)
		wr = w - wl
		sys.stdout.write("\r" + lhs)
		for i in range(wl): sys.stdout.write("#")
		for i in range(wr): sys.stdout.write(" ")
		sys.stdout.write(rhs)
		if cur_value == max_value:
			sys.stdout.write("\n")
		sys.stdout.flush()

	def param_count(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Attention(nn.Module):
	def __init__(self, res, heads, device='cpu'):
		super().__init__()
		self.res = res
		self.heads = heads
		self.device = device
		self.head_res = res // heads
		self.out_res = self.head_res * heads
		self.Q = Matrix(res, self.out_res, bias=False, device=self.device)
		self.K = Matrix(res, self.out_res, bias=False, device=self.device)
		self.V = Matrix(res, self.out_res, bias=False, device=self.device)
	def forward(self, x, y=None, mask=None):
		batch = x.shape[0]
		seq = x.shape[1]
		if y == None: y = x
		q = self.Q(x).reshape(batch, seq, self.heads, self.head_res).transpose(1, 2).reshape(batch * self.heads, seq, self.head_res)
		k = self.K(y).reshape(batch, seq, self.heads, self.head_res).transpose(1, 2).reshape(batch * self.heads, seq, self.head_res)
		v = self.V(y).reshape(batch, seq, self.heads, self.head_res).transpose(1, 2).reshape(batch * self.heads, seq, self.head_res)
		w = torch.bmm(q, k.transpose(-2, -1))
		w = w / math.sqrt(self.head_res)
		if mask != None: w = w + mask
		a = torch.bmm(F.softmax(w, dim=-1), v)
		return a.reshape(batch, self.heads, seq, self.head_res).transpose(1, 2).reshape(batch, seq, self.out_res)

class UnicornConv1d(nn.Module):
	def __init__(self, res, taps=[1, 2, [1, 4], [1, 16]], padding=True, device='cpu'):
		super().__init__()
		self.res = res
		self.device = device
		self.tap_res = res // len(taps)
		self.out_res = self.tap_res * len(taps)
		self.cumulative = False
		self.taps = []
		support = 0
		overhead = 0
		for n in taps:
			if type(n) != type([]):
				n = [n, n]
			n_min = min(n[0], n[1])
			n_max = max(n[0], n[1]) + 1
			support = max(support, n_max)
			self.taps.append([n_min, n_max])
			overhead += max(n_max - n_min - 2, 0)
			if overhead >= 4:
				self.cumulative = True
		self.support = support
		if padding == True: padding = support
		if padding == False: padding = 0
		self.kernel = Matrix(self.res, self.out_res, bias=False, device=self.device)
		self.padding = Param((1, padding, self.out_res), device=self.device)

	def forward(self, x, y=None):
		f = []
		u = self.kernel(x)
		padding = self.padding.repeat((x.shape[0], 1, 1))
		if y != None:
			if y.shape[0] < x.shape[0]:
				y = y.repeat((math.ceil(x.shape[0] / y.shape[0]), 1, 1))[:x.shape[0],:,:]
			padding = torch.cat([padding, y], axis=1)
		if padding.shape[1] > self.support:
			padding = padding[:,-self.support:,:]
		u = torch.cat([padding, u], axis=1)
		if padding.shape[1] < self.support:
			u = F.pad(u, (0, 0, self.support - padding.shape[1], 0, 0, 0))
		if self.cumulative:
			v = torch.cumsum(u, dim=1)
		for i in range(len(self.taps)):
			n = self.taps[i]
			count = n[1] - n[0]
			if count == 1:
				t = u[:,self.support-n[0]:x.shape[1]+self.support-n[0],self.tap_res*i:self.tap_res*(i+1)]
			elif count == 2:
				n0 = n[0] + 0
				n1 = n[0] + 1
				t0 = u[:,self.support-n0:x.shape[1]+self.support-n0,self.tap_res*i:self.tap_res*(i+1)]
				t1 = u[:,self.support-n1:x.shape[1]+self.support-n1,self.tap_res*i:self.tap_res*(i+1)]
				t = t0 + t1
			elif self.cumulative:
				t0 = v[:,self.support-n[0]:x.shape[1]+self.support-n[0],self.tap_res*i:self.tap_res*(i+1)]
				t1 = v[:,self.support-n[1]:x.shape[1]+self.support-n[1],self.tap_res*i:self.tap_res*(i+1)]
				t = t0 - t1
			else:
				t = 0
				for j in range(count):
					nj = n[0] + j
					t = t + u[:,self.support-nj:x.shape[1]+self.support-nj,self.tap_res*i:self.tap_res*(i+1)]
			f.append(t / count)
		return torch.cat(f, axis=-1)

class UnicornGate(nn.Module):
	def __init__(self, in_dim, out_dim, capacity=1, passthru=0.5, noise=0.01, device='cpu'):
		super().__init__()
		self.capacity = capacity
		self.device = device
		self.noise_scale = math.sqrt(noise)
		self.H = Matrix(out_dim+in_dim, capacity * out_dim, bias=True, device=self.device)
		self.Z = Matrix(out_dim+in_dim, capacity + 1, bias=True, device=self.device)
		self.Z.bias.data[0] = math.log(passthru / (1.0 - passthru))
	def forward(self, x, y):
		if self.training and (self.noise_scale > 0):
			z_shape = list(x.shape)
			z_shape[-1] = 1 + self.capacity
			z_noise = self.noise_scale * torch.randn(z_shape, dtype=torch.float32, device=self.device)
		else:
			z_noise = 0
		xy = torch.cat([x, y], axis=-1)
		z = F.softmax(self.Z(xy) + z_noise, dim=-1).unsqueeze(-1)
		h = SoftClamp(self.H(xy), limit=2)
		new_shape = list(h.shape)
		new_shape.append(new_shape[-1] // self.capacity)
		new_shape[-2] = self.capacity
		h = torch.cat([x.unsqueeze(-2), h.reshape(new_shape)], axis=-2)
		return torch.sum(z*h, dim=-2)

class UnicornCore(nn.Module):
	def __init__(self, res, capacity, kernel=[1], attention=True, passthru=1/2, device='cpu'):
		super().__init__()
		self.res = res
		self.device = device
		self.capacity = capacity
		heads = len(kernel)
		passthru2 = math.sqrt(passthru)

		self.conv_kernel = UnicornConv1d(res, kernel, device=self.device)
		self.conv_gate = UnicornGate(self.conv_kernel.out_res, self.res, self.capacity, passthru=passthru2, device=self.device)

		if attention:
			self.attention_pe = Matrix(self.conv_kernel.out_res, self.res, bias=False, device=self.device)
			self.attention = Attention(res, heads, device=self.device)
		else:
			self.attention = None

		self.mem_kernel = Matrix(self.res, self.res*4, bias=True, device=self.device)
		self.mem_gate = UnicornGate(self.res, self.res, self.capacity, passthru=passthru2, device=self.device)
		self.mem_shaper = F.gelu

	def forward(self, x, y=None, mask=None):
		ux = UnitVariance(x)
		uy = UnitVariance(y)
		h = self.conv_kernel(ux, uy)
		if self.attention != None:
			u = ux + self.attention_pe(h)
			v = Join([uy, u], axis=1)
			h = h + self.attention(u, v, mask=mask)
		x = self.conv_gate(x, h)

		h = self.mem_kernel(UnitVariance(x))
		h0, h1, h2, h3 = torch.split(h, 4, dim=-1)
		h = h0*self.mem_shaper(h2) + h1*self.mem_shaper(h3)
		x = self.mem_gate(x, h)
		return x

class Unicorn(nn.Module):
	def __init__(self, res, depth, capacity=4, kernel=[1, 2, 3, [1, 4], [5, 8], [9, 12], [1, 16], [1, 32]], attention=1/4, passthru=1/2, device='cpu'):
		super().__init__()
		self.res = res
		self.depth = depth
		self.capacity = capacity
		self.kernel = kernel
		self.heads = len(kernel)
		self.device = device
		self.mask_cache = None
		j = depth - math.ceil(depth * attention)
		p = math.pow(passthru, 1/max(depth, 1))
		self.layers = nn.ModuleList([UnicornCore(res, capacity, kernel, (i>=j), passthru=p, device=self.device) for i in range(depth)])

	def forward(self, x, y=None, mask=True):
		if mask == True:
			xlen = x.shape[1]
			ylen = 0
			if y != None: ylen = y.shape[1]
			if (self.mask_cache == None) or (self.mask_cache.shape[1] < xlen) or (self.mask_cache.shape[2] < xlen+ylen):
				if self.mask_cache != None:
					xlen = max(xlen, self.mask_cache.shape[1])
					ylen = max(ylen, self.mask_cache.shape[2] - self.mask_cache.shape[1])
				mask = torch.zeros([1, xlen, xlen+ylen], dtype=torch.float)
				indices = torch.triu_indices(xlen, xlen+ylen, offset=1+ylen)
				mask[:,indices[0],indices[1]] = float(-math.inf)
				self.mask_cache = mask.to(self.device)
			offset = self.mask_cache.shape[2] - self.mask_cache.shape[1] - ylen
			mask = self.mask_cache[:,:x.shape[1],offset:offset+x.shape[1]].repeat((x.shape[0]*self.heads, 1, 1))
		for i in range(len(self.layers)):
			yy = y
			if (y != None) and (len(y.shape) == 4):
				yy = y[:,:,:,i]
			x = self.layers[i](x, yy, mask=mask)
		return x
