using MXNet
using MNIST
# mlp = @mx.chain mx.Variable(:data) =>
#   mx.MLP([128, 64, 10])            =>
#   mx.SoftmaxOutput(name=:softmax)

mlp = @mx.chain mx.Variable(:data) 			   =>
  mx.FullyConnected(name=:fc1, num_hidden=128)  =>
  mx.Activation(name=:relu1, act_type = :relu)  =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
 mx.Activation(name=:relu2, act_type=:relu)   =>
 mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100


# method 1 - use package
include(Pkg.dir("MXNet", "examples", "mnist", "mnist-data.jl"))
train_provider, eval_provider = get_mnist_providers(batch_size)



# method 2 - manually get the data from train_provider
a = zeros(784, 0); 
labels = zeros(0); 

# create Julia arrays for training
for batch in mx.eachbatch(train_provider)
	dataTmp = mx.get_data(train_provider, batch)[1]
	a = hcat(a, copy(dataTmp))
	labels = vcat(labels, copy(mx.get_label(train_provider, batch)[1]))
end

b = zeros(784,0);
labels_eval = zeros(0);
# create Julia arrays for testing
for batch in mx.eachbatch(eval_provider)
	dataTmp = mx.get_data(eval_provider, batch)[1]
	b = hcat(b, copy(dataTmp))
	labels_eval = vcat(labels_eval, copy(mx.get_label(eval_provider, batch)[1]))
end


# a_NDArray = mx.zeros(784, 60000)
# # add white noise
# copy!(a_NDArray, a + 0.01 * rand(size(a)))
# # find most adversarial delta x
# labels_NDArray = mx.zeros(60000)
# copy!(labels_NDArray, labels)

mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = batch_size)
eval_provider = mx.ArrayDataProvider(b, labels_eval, batch_size = batch_size)

#############################################
# setup model
model = mx.FeedForward(mlp, context=mx.cpu())

# optimizer
optimizer = mx.SGD(lr=0.1, momentum=0.9, weight_decay=0.00001)

# fit on example data with example set-up
# mx.fit(model, optimizer, mnist_provider, n_epoch=5, eval_data=mnist_provider)

#########################################
############ manual fit #################

input_shapes = mx.provide_data(mnist_provider)

# set up the executor
exec = mx.simple_bind(mlp, model.ctx[1]; grad_req=MXNet.mx.GRAD_ADD, input_shapes...)
mx.list_arguments(mlp)
model.pred_exec = exec

# include("fit.jl")
mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider)

# show weights
copy(model.arg_params[:fc2_weight])
# show gradients
model.pred_exec.grad_arrays
copy(model.pred_exec.grad_arrays[2])

##### Adversarial Robust Training ####
######### Need help #########
# while not converge
# Q1. how to get the weights or objective to check convergence?
# Q2. how to get gradient with respect to x_i?
# Q3. how to update the data in a provider?

probs = zeros(10, 10000)
for i in 1:10
	# update x_i with x_i + rho * sign(grad) [l-infinity]
	a = a + 0.01 * rand(size(a));
	# find most adversarial delta x

	mnist_provider = mx.ArrayDataProvider(a, labels, batch_size = batch_size);
	# fit parameters
	mx.fit(model, optimizer, mnist_provider, n_epoch=1, eval_data=eval_provider)

	probs = mx.predict(model, eval_provider)
	# get weights and update
end 


mx.reset!(nll)
nll = NLL()
labels_eval_ND = mx.zeros(size(labels_eval))
copy!(labels_eval, labels_eval_ND)
pred = mx.zeros(size(probs))
copy!(probs, pred)
mx.update!(nll, [labels_eval_ND], [pred])
nll

# Negative Log-likelihood
type NLL <: mx.AbstractEvalMetric
  nll_sum  :: Float64
  n_sample :: Int

  NLL() = new(0.0, 0)
end

function mx.update!(metric :: NLL, labels :: Vector{mx.NDArray}, preds :: Vector{mx.NDArray})
  @assert length(labels) == length(preds)
  nll = 0.0
  for (label, pred) in zip(labels, preds)
    @mx.nd_as_jl ro=(label, pred) begin
      nll -= sum(log(max(broadcast_getindex(pred, round(Int,label+1), 1:length(label)), 1e-20)))
    end
  end

  nll = nll / length(labels)
  metric.nll_sum += nll
  metric.n_sample += length(labels[1])
end

function mx.get(metric :: NLL)
  nll  = metric.nll_sum / metric.n_sample
  perp = exp(nll)
  return [(:NLL, nll), (:perplexity, perp)]
end

function mx.reset!(metric :: NLL)
  metric.nll_sum  = 0.0
  metric.n_sample = 0
end



#dataArray = Array[]
#a = mx.NDArray
#labels = mx.NDArray

a = zeros(784, 0)
labels = zeros(0)
lab = mx.NDArray[]
i = 0;
for batch in mx.eachbatch(train_provider)
	i = i+1;
  	#push!(dataArray,copy(mx.get_data(train_provider, batch)))
  	dataTmp = mx.get_data(train_provider, batch)[1]
  	if i == 2 
  		print(copy(dataTmp))
  	end
	a = hcat(a, copy(dataTmp))
	# copy!(a, a + 0.01 * rand(size(a)))
	#println(size(copy(mx.get_label(train_provider, batch)[1])))
  	#labels = mx.get_label(train_provider, batch)[1]
  	#push!(lab, mx.get_label(train_provider, batch)[1])
  	
  	labels = vcat(labels, copy(mx.get_label(train_provider, batch)[1]))
end



a_NDArray = mx.zeros(784, 60000)
# add white noise
copy!(a_NDArray, a + 0.01 * rand(size(a)))
# find most adversarial delta x
labels_NDArray = mx.zeros(60000)
copy!(labels_NDArray, labels)


mnist_provider = mx.ArrayDataProvider(:data => a_NDArray, :softmax_label => labels_NDArray, batch_size = batch_size)

# fit parameters
mx.fit(model, optimizer, mnist_provider, n_epoch=5, eval_data=eval_provider)


probs = mx.predict(model, eval_provider)

# collect all labels from eval data
labels = Array[]
for batch in eval_provider
    push!(labels, copy(mx.get(eval_provider, batch, :softmax_label)))
end
labels = cat(1, labels...)

# Now we use compute the accuracy
correct = 0
for i = 1:length(labels)
    # labels are 0...9
    if indmax(probs[:,i]) == labels[i]+1
        correct += 1
    end
end
accuracy = 100correct/length(labels)
println(mx.format("Accuracy on eval set: {1:.2f}%", accuracy))





# trainX, trainY = traindata()
# trainX_NDArray = mx.zeros(size(trainX))
# copy!(trainX_NDArray, trainX)
# trainY_NDArray = mx.zeros(size(trainY))
# copy!(trainY_NDArray, trainY)


# testX, testY = testdata()
# testX_NDArray = mx.zeros(size(testX))
# copy!(testX_NDArray, testX)
# testY_NDArray = mx.zeros(size(testY))
# copy!(testY_NDArray, testY)


# filenames = mx.get_mnist_ubyte()
# mnist_provider = mx.MNISTProvider(image=dataTmp2,#filenames[:train_data],
#                                 label=labels,#filenames[:train_label],
#                                 data_name=:data, label_name=:softmax_label,
#                                 batch_size=batch_size, shuffle=true, flat=true, silent=true)

