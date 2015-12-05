
opts = mx.TrainingOptions()



info("Initializing parameters...")
arg_names, param_names, aux_names = mx._init_model(model, mnist_provider, 
	mx.UniformInitializer(0.01), true)
model.arg_params

# setup kvstore
kvstore = opts.kvstore
if isa(kvstore, Base.Symbol)
	info("Creating KVStore...")
	kvstore, update_on_kvstore = mx._create_kvstore(kvstore, length(model.ctx), model.arg_params)
end
if !update_on_kvstore
    updater = mx.get_updater(optimizer)
end
# mx.copy_params_from(model.pred_exec, model.arg_params, model.aux_params)
# set up input data structures

num_dev = 1

##### modified fit function #####

slices      = mx._split_inputs(batch_size,1)

data = mnist_provider;
data_names   = [x[1] for x in mx.provide_data(data)]
label_names  = [x[1] for x in mx.provide_label(data)]

data_arrays  = [mx.SlicedNDArray[(slices[1], exec.arg_dict[name])]
              for name in data_names]
label_arrays = [mx.SlicedNDArray[(slices[1], exec.arg_dict[name])]
              for name in label_names]

param_idx    = filter(i -> in(arg_names[i], param_names), 1:length(arg_names))

param_arrays = [mx.NDArray[exec.arg_arrays[i]] for i in param_idx]
grad_arrays  = [mx.NDArray[exec.grad_arrays[i]] for i in param_idx]
aux_arrays   = [mx.NDArray[exec.aux_arrays[i]] for i = 1:length(aux_names)]

op_state = mx.OptimizationState(batch_size)
optimizer.state = op_state

# if !update_on_kvstore
# updater = get_updater(optimizer)
# end

# if !isa(kvstore, Void)
# if update_on_kvstore
#   set_optimizer(kvstore, optimizer)
# end

# info("Initializing KVStore...")
# # init kv with gradients
# for idx = 1:length(param_arrays)
#   param_on_devs = param_arrays[idx]

#   init!(kvstore, idx, self.arg_params[param_names[idx]])

#   if update_on_kvstore
#     # pull weights back
#     pull!(kvstore, idx, param_on_devs, priority=-idx)
#   end
# end
# end

# set up output and labels in CPU for evaluation metric
output_shapes = [tuple(size(x)[1:end-1]...,batch_size) for x in exec.outputs]
cpu_dev = model.ctx[1]
cpu_output_arrays = [mx.zeros(shape, cpu_dev) for shape in output_shapes]
cpu_label_arrays  = [mx.empty(shape, cpu_dev) for (name,shape) in mx.provide_label(data)]

# invoke callbacks on epoch 0
mx._invoke_callbacks(model, opts.callbacks, op_state, mx.AbstractEpochCallback)

# now start training...
# for i_epoch = 1:opts.n_epoch
time_start = time()
mx.reset!(opts.eval_metric)

# op_state.curr_epoch = i_epoch
# op_state.curr_batch = 0

# invoke callbacks on iteration 0
mx._invoke_callbacks(model, opts.callbacks, op_state, mx.AbstractBatchCallback)


for batch in mx.eachbatch(data)
# batch_array = mx.AbstractDataBatch[]
# for batch in mx.eachbatch(data)
# 	push!(batch_array, batch)
# end
# batch = batch_array[1]
  mx.load_data!(data, batch, data_arrays)
  mx.load_label!(data, batch, label_arrays)

  # forward and backward
  for islice in slices
    mx.forward(exec, is_train=true)
    # copy outputs into cpu ndarray, for evaluation metric
    for (cpu_out, dev_out) in zip(cpu_output_arrays, exec.outputs)
      copy!(slice(cpu_out, islice), dev_out)
    end

    mx.backward(exec)
  end

  op_state.curr_iter  += 1
  op_state.curr_batch += 1
  optimizer.state = op_state

  # update parameters
  for idx = 1:length(param_names)
    # gradient synchronization
    if !isa(kvstore, Void)
      # push gradient, priority is negative index
      push!(kvstore, idx, grad_arrays[idx], priority=-idx)
      if update_on_kvstore
        # pull back the weights
        pull!(kvstore, idx, param_arrays[idx], priority=-idx)
      else
        # pull back the sum-ed gradients, to the same locations
        pull!(kvstore, idx, grad_arrays[idx], priority=-idx)
      end
    end

    if !update_on_kvstore
      # manual updating
      for i_dev = 1:num_dev
        # create a fake index, so that the updater create states
        # for different param AND different devices, TODO(mli)
        # use a better solution later
        fake_idx = idx * num_dev + i_dev
        updater(fake_idx, grad_arrays[idx][i_dev], param_arrays[idx][i_dev])
      end
    end
  end

  # invoke callbacks after finishing each iteration
  mx._invoke_callbacks(model, opts.callbacks, op_state, mx.AbstractBatchCallback)

  # update evaluation metric on training set
  mx.load_label!(data, batch, cpu_label_arrays)
  mx.update!(opts.eval_metric, cpu_label_arrays, cpu_output_arrays)
end # end of one epoch

time_stop = time()
info(mx.format("== Epoch {1:0>3d} ==========", 1))
info("## Training summary")
for (name, value) in get(opts.eval_metric)
  info(mx.format("{1:>15s} = {2:.4f}", name, value))
end
info(mx.format("{1:>15s} = {2:.4f} seconds", "time", time_stop-time_start))
