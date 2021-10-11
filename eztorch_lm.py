import io
import os
import sys
import math
import json
import torch
import random
import zstandard
import numpy as np
import eztorch as ez
import torch.nn as nn
import torch.nn.functional as F

prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
config_file = None
filename = None
res = 512
depth = 6
capacity = 2
steps = 10000
tokenizer = None
wandb_entity = None
wandb_project = None
wandb_name = None
seq_length = 1024
batch_size = 1
device_batch_size = 1
learn_rate = 0.0005
momentum = 0.8
weight_decay = 0.01
lr_decay = 0.001
device_batch_size_override = None
stochastic_beams = 1
fixed_beams = 0
top_p = 0.9
output_length = 0
save_every = 0

i = 1
while i < len(sys.argv):
	if sys.argv[i] == "--config":
		i += 1
		config_file = sys.argv[i]
	elif sys.argv[i] == "--steps":
		i += 1
		steps = int(sys.argv[i])
	elif sys.argv[i] == "--prompt":
		i += 1
		prompt = sys.argv[i]
	elif sys.argv[i] == "--device-batch-size":
		i += 1
		device_batch_size_override = int(sys.argv[i])
	elif sys.argv[i] == "--stochastic-beam-search":
		i += 1
		stochastic_beams = int(sys.argv[i])
	elif sys.argv[i] == "--beam-search":
		i += 1
		fixed_beams = int(sys.argv[i])
	elif sys.argv[i] == "--top-p":
		i += 1
		top_p = float(sys.argv[i])
	elif sys.argv[i] == "--output-length":
		i += 1
		output_length = int(sys.argv[i])
	elif sys.argv[i] == "--save-every":
		i += 1
		save_every = float(sys.argv[i])
	else:
		print("Unknown command line option: %s" % sys.argv[i])
		exit()
	i += 1

if config_file == None:
	print("training usage:\n$ python eztorch_lm.py --config <filename.json>\n\t--steps %d\n\t--device-batch-size %d\n"%(steps, device_batch_size))
	print("generator usage:\n$ python eztorch_lm.py --config <filename.json>\n\t--prompt \"%s\"\n\t--beam-search %d\n\t--stochastic-beam-search %d\n\t--top-p %.02f\n\t--output-length %d\n"%(prompt,fixed_beams,stochastic_beams,top_p,output_length))
	exit()

with open(config_file, "rb") as fp:
	cfg = json.loads(fp.read())
	filename = config_file + ".ez"
	res = int(cfg["res"])
	depth = int(cfg["depth"])
	capacity = int(cfg["capacity"])
	tokenizer = cfg["tokenizer"]
	seq_length = int(cfg["seq_length"])
	batch_size = int(cfg["batch_size"])
	learn_rate = float(cfg["learn_rate"])
	momentum = float(cfg["momentum"])
	weight_decay = float(cfg["weight_decay"])
	lr_decay = float(cfg["lr_decay"])
	if "device_batch_size" in cfg:
		device_batch_size = int(cfg["device_batch_size"])
	if "wandb_entity" in cfg:
		wandb_entity = cfg["wandb_entity"]
	if "wandb_project" in cfg:
		wandb_project = cfg["wandb_project"]
	if "wandb_name" in cfg:
		wandb_name = cfg["wandb_name"]

if device_batch_size_override:
	device_batch_size = device_batch_size_override

wandb_config = None
if wandb_entity != None:
	wandb_config = {"entity":wandb_entity, "project":wandb_project, "name":wandb_name}

pile_path = "./lm/"
pile_url = "https://the-eye.eu/public/AI/pile/train/"

if torch.cuda.is_available():
	device = 'cuda'
else:
	torch.set_num_threads(1)
	device = 'cpu'

class UnicornLM(ez.Model):
	def __init__(self, unicorn, tokenizer=None, **kwargs):
		super().__init__(device=unicorn.device, **kwargs)
		self.res = unicorn.res
		self.tokenizer = None
		self.vocab_size = 256
		self.detokenize_prefix_data = []
		self.detokenize_prefix_len = 0
		if tokenizer:
			if type(tokenizer) == type(""):
				import youtokentome as yttm
				self.tokenizer = yttm.BPE(model=tokenizer)
			else:
				self.tokenizer = tokenizer
			self.vocab_size = self.tokenizer.vocab_size()
			dummy = "a"
			self.detokenize_prefix_data = self.tokenizer.encode(dummy)
			self.detokenize_prefix_len = len(dummy)

		self.unicorn = unicorn
		self.text_in = nn.Embedding(self.vocab_size, self.res, device=self.device)
		self.text_out = ez.Matrix(self.res, self.vocab_size, bias=True, device=self.device)

	def forward(self, x, y=None):
		x = self.prepare_input(x)
		y = self.prepare_input(y, x.shape[0])
		z = self.unicorn(x, y, mask=True)
		return self.text_out(z)
	
	def prepare_input(self, x, batch_size=1):
		if x == None: return x
		if type(x) == bytearray: x = list(x)
		if type(x) == type(""): x = [x]
		if type(x) == type([]):
			if len(x) == 0:
				x = torch.zeros((batch_size, 0), device=self.device, dtype=torch.long)
			else:
				if type(x[0]) == type(""): x = [self.tokenize(x) for x in x]
				x = torch.tensor(x, device=self.device, dtype=torch.long)
				if len(x.shape) <= 1: x = x.unsqueeze(0)	
		if len(x.shape) == 2: x = self.text_in(x)
		if x.shape[0] < batch_size:
			x = x.repeat((batch_size // x.shape[0], 1, 1))
		return x

	def tokenize(self, x, dropout=0):
		if self.tokenizer:
			if dropout == 0:
				return self.tokenizer.encode(x)
			return self.tokenizer.encode(x, dropout_prob=dropout)
		return bytearray(x.encode('utf-8', 'ignore'))

	def detokenize(self, x):
		if self.tokenizer:
			return self.tokenizer.decode(self.detokenize_prefix_data + x)[0][self.detokenize_prefix_len:]
		return bytes(x).decode('utf-8', 'ignore')

	def generate(self, prompt, length=64, top_p=0.95, fixed_beams=0, stochastic_beams=1, live=False):
		if type(prompt) == type(""):
			prompt = self.tokenize(prompt)
		if live == True: live = print
		beams = [{'p':0,'t':prompt}]
		max_beams = fixed_beams + stochastic_beams
		try:
			for k in range(length):
				new_beams = []
				for beam in beams:
					Pr = F.softmax(self(beam['t']), dim=-1)
					for i in range(self.vocab_size):
						b = {'p':beam['p']+math.log(max(Pr[0,-1,i].item(), 1e-6)), 't':list(beam['t'])+[i]}
						new_beams.append(b)
				if len(new_beams) > max_beams:
					new_beams.sort(key=lambda x: x['p'], reverse=True)
					new_beams_p = np.array([x['p'] for x in new_beams])
					new_beams_p = new_beams_p - new_beams_p.max()
					new_beams_p = np.exp(new_beams_p)
					new_beams_p = new_beams_p / new_beams_p.sum()
					sum_p = 0
					new_p = []
					new_id = []
					X_size = len(new_beams)
					X = sorted(range(X_size), key=lambda x: -new_beams_p[x])
					for i in range(X_size):
						p = new_beams_p[X[i]]
						new_id.append(X[i])
						new_p.append(p)
						sum_p += p
						if sum_p > top_p:
							if len(new_id) >= max_beams:
								break
					beams = [new_beams[i] for i in new_id[:fixed_beams]]
					new_id = new_id[fixed_beams:]
					new_p = new_p[fixed_beams:]
					if len(new_id) and stochastic_beams:
						new_p = np.array(new_p)
						new_p = new_p / new_p.sum()
						new_id = np.random.choice(new_id, size=(stochastic_beams), replace=False, p=new_p)
						beams.extend([new_beams[i] for i in new_id[:stochastic_beams]])
				else:
					beams = new_beams[:max_beams]
				if live != False:
					live(self.detokenize(beams[0]['t']), beams[0]['p'])
		except KeyboardInterrupt:
			pass
		if live != False:
			live(self.detokenize(beams[0]['t']), beams[0]['p'])
		return beams[0]['t']

model = UnicornLM(ez.Unicorn(res, depth, capacity, device=device), tokenizer=tokenizer)
print("model params: %.02fM"%(model.param_count() / 1000000))
model.train()
model.learn_rate = learn_rate
model.momentum = momentum
model.weight_decay = weight_decay
model.lr_decay = lr_decay
if filename: model.load(filename)

pile = None
pile_chunk = 0
def pile_readline():
	global pile
	global pile_path
	global pile_chunk
	while True:
		r = False
		if pile: r = pile.readline()
		if r: return r
		fn = "%02d.jsonl.zst" % pile_chunk
		chunk_fn = pile_path + fn
		if not os.path.exists(pile_path):
			os.mkdir(pile_path)
		if not os.path.exists(chunk_fn):
			import time
			import urllib.request
			def download_callback(count, block_size, total_size):
				global start_time
				if count == 0:
					start_time = time.time()
					return
				duration = time.time() - start_time
				progress_size = round(count * block_size)
				speed = round(progress_size / (1024 * duration))
				percent = min(round(count * block_size * 100 / total_size), 100)
				sys.stdout.write("\rDownloading '%s' (%d%%, %d MB, %d KB/s, %ds elapsed)" % (fn, percent, progress_size / (1024*1024), speed, duration))
				sys.stdout.flush()
			try:
				os.makedirs(pile_path, exist_ok=True)
				chunk_url = pile_url + fn
				urllib.request.urlretrieve(chunk_url, chunk_fn, download_callback)
			except:
				print("\nDownload failed.")
				return False
		fp = None
		try:
			fp = open(chunk_fn, 'rb')
		except OSError:
			continue
		if fp == None:
			return False
		zfp = zstandard.ZstdDecompressor().stream_reader(fp)
		pile = io.TextIOWrapper(zfp, encoding='utf-8')
		pile_chunk += 1

if steps:
	if model.sample: print("Skipping %d documents..."%model.sample)
	for i in range(model.sample): pile_readline()
	batch_count = max(batch_size // device_batch_size, 1)

	if wandb_config != None:
		model.wandb_init(**wandb_config, config={
			"Model Resolution": res,
			"Model Depth": depth,
			"Context Size": seq_length,
			"Optimizer Batch Size (Target)": batch_size,
			"Optimizer Batch Size (Actual)": device_batch_size*batch_count,
			"Device Batch Size": device_batch_size,
			"Tokens Per Step": seq_length*device_batch_size*batch_count,
		})

last_save = time.time()
try:
	should_save = False
	for i in range(steps*batch_count):
		should_save = True
		model.progress_bar(i, steps*batch_count)

		doc_in = []
		doc_out = []
		for i in range(device_batch_size):
			buf = []
			while len(buf) <= seq_length+1:
				model.sample += 1
				doc = pile_readline()
				if not doc:
					print("Exhausted all training data!")
					break
				doc = json.loads(doc)['text']
				buf.extend(model.tokenize(doc))
				buf.append(0)
			if not doc: break

			doc_in.append(buf[:seq_length])
			doc_out.append(buf[1:seq_length+1])
		if not doc: break

		x = torch.tensor(doc_in, device=model.device, dtype=torch.long)
		y_true = torch.tensor(doc_out, device=model.device, dtype=torch.long)
		y_pred = model(x)
		
		y_true = F.one_hot(y_true, model.vocab_size)
		y_pred = F.log_softmax(y_pred, dim=-1)
		loss = -torch.mean(torch.sum(y_pred * y_true, dim=-1)) / math.log(2)

		if model.backprop(loss) >= batch_count:
			model.optimize()
			if filename and (save_every > 0):
				now = time.time()
				if (now-last_save) >= save_every:
					last_save = now
					print("Saving...")
					model.save(filename)

except KeyboardInterrupt:
	print("Interrupted")

if filename and should_save:
	print("Saving...")
	model.save(filename)

if output_length:
	model.generate(prompt, length=output_length, top_p=top_p, fixed_beams=fixed_beams, stochastic_beams=stochastic_beams, live=True)
