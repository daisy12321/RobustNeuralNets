using MXNet
#using MNIST
include("nll.jl")

mlp = @mx.chain mx.Variable(:data) 			        =>
  mx.FullyConnected(name=:fc1, num_hidden=128)  =>
  mx.Activation(name=:relu1, act_type = :relu)  =>
  mx.FullyConnected(name=:fc2, num_hidden=64)   =>
 mx.Activation(name=:relu2, act_type=:relu)     =>
 mx.FullyConnected(name=:fc3, num_hidden=10)    =>
  mx.SoftmaxOutput(name=:softmax)


###########################################
# get data 
# manually get the data from train_provider
batch_size = 100
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)

shape = mx.provide_data(train_provider)[1][2]

# create Julia arrays for training
# very slow, any way to speed up?
a = zeros(shape[1], 0); 
labels = zeros(0); 
for batch in mx.eachbatch(train_provider)
	dataTmp = mx.get_data(train_provider, batch)[1]
	a = hcat(a, copy(dataTmp))
	labels = vcat(labels, copy(mx.get_label(train_provider, batch)[1]))
end
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

mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = batch_size)
eval_provider = mx.ArrayDataProvider(b, labels_eval, batch_size = batch_size)



#############################################
# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

input_shapes = mx.provide_data(mnist_provider)

# set up the executor, pass onto model
exec = mx.simple_bind(mlp, model.ctx[1]; grad_req=MXNet.mx.GRAD_WRITE, input_shapes...)
mx.list_arguments(mlp)
model.pred_exec = exec

# fit model
mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider)


###### make prediction ######
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
copy(model.arg_params[:fc2_weight])
##### show gradients
model.pred_exec.grad_arrays
copy(model.pred_exec.grad_arrays[2])
copy(exec.grad_arrays[2])
# Q. why all zeros?




##### Adversarial Robust Training ####
######### Need help #########
# while not converge
# Q1. how to get the weights or objective to check convergence?
# Q2. how to get gradient with respect to x_i?
# Q3. how to update the data in a provider?

i = 0
nll_old = 100000
nll_new = 50000
while (abs(nll_new - nll_old) > 1 | i < 5)

  nll_old = nll_new
  i = i+1;
  # find most adversarial delta x
	# update x_i with x_i + rho * sign(grad) [l-infinity]
  # for now do a random noise
	a = a + 0.01 * rand(size(a));

	mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = batch_size);
	# fit parameters
	mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider)

	probs = mx.predict(model, mnist_provider)
  pred = mx.zeros(size(probs))
  copy!(pred, probs)

  # update NLL

  mx.reset!(nll)
  mx.update!(nll, [labels_ND], [pred])
  nll_new = nll.nll_sum

  println(mx.format("NLL on training: {1:.2f}", nll_new))

end 



