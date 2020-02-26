# Robust Artificial Neural Networks

A design and implementation of artificial neural networks robust to data uncertainties via adversarial training

Artificial neural network is a popular and successful machine learning method. However, when data are subject to uncertainty, the model can be unstable and overfit the training data. In this project, I explored the possibility of immunizing ANNs against uncertainties in data via robust optimization. I derived the robust formulation and established its connection with regularization and adversarial training. As a preliminary computational experiment (but not the focus of the paper), I implemented the method and observe that Robust-ANN offers slightly higher out-of-sample accuracy in the MNIST dataset.
