using MXNet
#using MNIST
include("nll.jl")

mlp = @mx.chain mx.Variable(:data)=>
  mx.FullyConnected(name=:fc1, num_hidden=128)=>
  mx.Activation(name=:relu1, act_type = :relu)=>
  mx.FullyConnected(name=:fc2, num_hidden=64)=>
  mx.Activation(name=:relu2, act_type=:relu)=>
  mx.FullyConnected(name=:fc3, num_hidden=10)=>
  mx.SoftmaxOutput(name=:softmax)


###########################################
# get data 
# manually get the data from train_provider
batch_size = 10000
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

shape = mx.provide_data(train_provider)[1][2]

# create Julia arrays for training
a = zeros(shape[1], 0); 
labels = zeros(0); 
for batch in mx.eachbatch(train_provider)
	dataTmp = mx.get_data(train_provider, batch)[1]
	a = hcat(a, copy(dataTmp))
	labels = vcat(labels, copy(mx.get_label(train_provider, batch)[1]))
end
a_ND = mx.zeros(size(a))
copy!(a_ND, a)
labels_ND = mx.zeros(size(labels))
copy!(labels_ND, labels)

# create Julia arrays for testing
b = zeros(shape[1],0);
labels_eval = zeros(0);
for batch in mx.eachbatch(eval_provider)
	dataTmp = mx.get_data(eval_provider, batch)[1]
	b = hcat(b, copy(dataTmp))
	labels_eval = vcat(labels_eval, copy(mx.get_label(eval_provider, batch)[1]))
end
labels_eval_ND = mx.zeros(size(labels_eval))
copy!(labels_eval_ND, labels_eval)

new_batch_size = 100
mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = new_batch_size)
eval_provider = mx.ArrayDataProvider(b, labels_eval, batch_size = new_batch_size)



# batch_new = mx.ArrayDataBatch()
# for batch in mx.eachbatch(eval_provider)
#   batch_new = println(batch)
# end

#############################################
# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

input_shapes = mx.provide_data(mnist_provider)

# fit model
mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider)

# set up the executor, pass onto model
exec = mx.simple_bind(mlp, model.ctx[1]; grad_req=MXNet.mx.GRAD_WRITE, input_shapes...)
mx.copy_params_from(exec, model.arg_params, model.aux_params)
# mx.list_arguments(mlp)

### visualization 
# open("visualize.dot", "w") do io
#   println(io, mx.to_graphviz(mlp))
# end
# run(pipeline(`dot -Tsvg visualize.dot `, stdout="visualize.svg"))
# run(pipeline(`dot -Tpdf visualize.dot `, stdout="visualize.pdf"))

function get_xgrad(exec)
  # mx.forward(exec)
  # mx.backward(exec)
  fc1_weights = copy(exec.arg_arrays[2])
  grad_activation = copy(exec.grad_arrays[3])
  hgrad = zeros(length(grad_activation));
  xgrad = zeros(shape[1]);
  for i = 1:length(grad_activation)
    if grad_activation[i] > 0.0001 
      hgrad[i] = 1
    end    
  end

  for j = 1:shape[1]
    xgrad[j] = sum(grad_activation .* hgrad .* fc1_weights[j,:])
  end

  return(xgrad)

end

xgrad = get_xgrad(exec)




batch = mx.ArrayDataBatch(1:100)



# mx.copy_params_from(exec, model.arg_params, model.aux_params)

copy(exec.arg_arrays[8])[1:5]

# function get_info()
#   mx.every_n_batch(50, call_on_0=true) do state :: mx.OptimizationState
#     #info("Testing")
#     println(state.curr_batch)
#     mx.copy_params_from(exec, model.arg_params, model.aux_params)
#     print(copy(exec.arg_arrays[8])[1:5])
#   end
# end

# mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider, callbacks = [get_info()])
# # copy the parameters to executer 
# mx.copy_params_from(exec, model.arg_params, model.aux_params)


# ###### make prediction ######
probs = mx.predict(model, eval_provider)
pred = mx.zeros(size(probs))
copy!(pred, probs)
# get nll
nll = NLL()
mx.reset!(nll)
mx.update!(nll, [labels_eval_ND], [pred])
nll

###############################
##### show weights
# fc1_weights = copy(model.arg_params[:fc1_weight])


# for batch in mx.eachbatch(mnist_provider)
#   mx.forward(exec, is_train=true)
#   mx.backward(exec)
#   # println(copy(exec.grad_arrays[3]))
# end
# ##### show gradients
# exec.grad_arrays
# copy(exec.grad_arrays[8])


##### Adversarial Robust Training ####
input_shapes = mx.provide_data(mnist_provider);
i = 0
nll_old = 100000;
nll_new = 50000;
mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = new_batch_size);
anew = a; 
nll = NLL();
while (abs(nll_new - nll_old) > 10 | i < 20)


  nll_old = nll_new;
  i = i+1;

  model = mx.FeedForward(mlp, context=mx.cpu())
  optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)
  exec = mx.simple_bind(mlp, model.ctx[1]; grad_req=MXNet.mx.GRAD_WRITE, input_shapes...)

	# fit parameters
  mx.srand!(1234)
	mx.fit(model, optimizer, mnist_provider, n_epoch=3, eval_data=eval_provider)
  mx.copy_params_from(exec, model.arg_params, model.aux_params)
  
  # to get gradient, given current weight, for each batch
  for batch in mx.eachbatch(mnist_provider)
    idx = batch.idx
    data = mx.get_data(mnist_provider, batch)[1]
    # update exec with the current data
    exec.arg_arrays[1] = data
    exec.arg_dict[:data] = data

    # update the gradient using backward call
    mx.backward(exec)
    xgrad = get_xgrad(exec)
    # find most adversarial delta x
    # update x_i with x_i + rho * sign(grad) [l-infinity]
    anew[:, idx] = a[:, idx] + 0.01 * repmat(sign(xgrad), 1, new_batch_size)
    # println(copy(exec.grad_arrays[3]))
  end

  # anew = a + 0.01 * rand(size(a));

  mnist_provider = mx.ArrayDataProvider(anew, labels, batch_size = new_batch_size)
	probs = mx.predict(model, mnist_provider)
  pred = mx.zeros(size(probs))
  copy!(pred, probs)

  # update NLL
  mx.reset!(nll)
  mx.update!(nll, [labels_ND], [pred])
  nll_new = nll.nll_sum

  println(mx.format("NLL on training: {1:.2f}", nll_new))

end 



