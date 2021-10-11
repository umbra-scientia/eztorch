### EzTorch - A quick and easy boilerplate for deep learning with torch.

A deep learning model that inherits `torch.nn.Module` can inherit `eztorch.Model` instead, and instantly gain easy-to-use save and load functions with integrated optimization and metric reporting.
```python
model.load("filename.ez")
model.wandb_init(entity="your-user-name", project="your-project-name", name="your-model-name")
model.backprop(loss)
model.optimize()
model.save("filename.ez")
```
The integrated EzTorch optimizer synergistically combines AdamW and AdaBelief while eliminating most of their hyperparameters. Only the learning rate and momentum factor need to be tuned. A warmup effect occurs naturally.

The WandB run ID is automatically saved with the model parameters. These features are battle tested and flexible enough for advanced use cases encountered in real research.
```python
# Full batch gradient descent with aux losses and regularizers.
model.learn_rate = 0.0005
model.momentum = 0.8
model.wandb_init(
	entity="your-user-name",
	project="your-project-name",
	name="your-model-name",
	config={"Custom Model Information": True}
)
for minibatch in full_batch:
	loss, aux, reg = compute_loss(model(minibatch.x), minibatch.y)
	model.backprop({
		"main loss": loss,
		"auxiliary loss": aux,
		"regularizer": reg
	}, loss_weights={"regularizer": 1e-4})
model.optimize()
```

There are also some modern deep learning building blocks and other cool little things to make life nicer.
-  `eztorch.Unicorn(res, depth, ...)` Spatial sequence modeling. (Replaces Transformer and LSTM.)
   -  `eztorch.Unicorn.forward(x, y=None, mask=None)` x: sequence input, y: context input
-  `eztorch.UnicornGate(y_res, x_res, ...)` Residual learning stabilizer. (Replaces GRU gates in GTrXL & Unicorn.)
   -  `eztorch.UnicornGate.forward(x, y)` x: pass-thru input, y: new context input
-  `eztorch.UnicornConv1d(res, taps=[1, 2, [1, 4], [1, 16]])` Efficient non-uniform convolution. Cost is independent of receptive field size.
   -  `eztorch.UnicornConv1d.forward(x, y=None)` x: signal input, y: padding input
   -  `eztorch.UnicornConv1d.out_res` Last dimension of output shape.
-  `eztorch.Attention(res, heads)` Multi-head attention. (No post-projection.)
   -  `eztorch.Attention.forward(x, y=None, mask=None)` x: self attention input, y: cross attention input
   -  `eztorch.Attention.out_res` Last dimension of output shape.
-  `eztorch.UnitSphere(x)` x: input tensor (Scales vectors to unit length.)
-  `eztorch.UnitVariance(x)` x: input tensor (Scales vectors to unit variance.)
-  `eztorch.VarianceNorm(scale)` Activation whitening. (Scales vectors to a learnable variance.)
   -  `eztorch.VarianceNorm.forward(x)` x: input tensor
-  `eztorch.SoftClamp(x, limit=2)` Linear on `[1-limit, limit-1]` but clamps to `[-limit, limit]` using tanh.
- `eztorch.Model` A drop-in replacement for `torch.nn.Module` with EzTorch features.
  - `eztorch.Model.save(filename, extra_data={})` Save model weights, gradients, optimizer states, report IDs, ...
  - `eztorch.Model.load(filename, reject=None)` Load weights, optimizer state, report IDs, ... (returns `extra_data`)
  - `eztorch.Model.backprop(losses, loss_weights=None)` Compute and store loss gradients for every parameter.
  - `eztorch.Model.optimize(aux_metrics={})` Optimize parameters using stored gradients, and update dashboard.
  - `eztorch.Model.progress_bar(cur_value, max_value, title=None, losses=None)` Pretty-prints a progress bar.
  - `eztorch.Model.param_count()` Returns the total number of learnable parameters.
  - `eztorch.Model.elapsed_time()` Returns the total number of seconds the model has spent loaded into the memory of processes that eventually save the model. This helps when estimating non-preemptible time requirements from research and tuning done on preemptible instances.

### Bonus: eztorch_lm.py
Language modeling is a natural fit for the Unicorn architecture. The provided script can train a model on The Pile. There is an example configuration JSON file in `lm/example.json`

Training usage:
```sh
$ python eztorch_lm.py --config <filename.json>
	--steps 10000
	--device-batch-size 16
```

Generator usage:
```sh
$ python eztorch_lm.py --config <filename.json>
	--prompt "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
	--beam-search 1
	--stochastic-beam-search 4
	--top-p 0.95
	--output-length 256
```
