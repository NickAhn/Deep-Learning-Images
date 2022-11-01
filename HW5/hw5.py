import torch
# %matplotlib inline
from matplotlib import pyplot as plt

# Input and output vectors are given. 
inp = [ 0.7300, -1.0400, -1.2300,  1.6700, -0.6300,  1.4300, -0.8400,  0.1500,
         -2.3000,  3.1000, -1.4500, -1.8100,  1.8700, -0.1100, -0.2800,  1.1200,
         -0.4200,  2.8900]
out = [ 1.43,  10.1,  8.3,  1.03,  10.21, -0.1,  8.92,  5.1,
         -7.53, 34.72,  7.61,  3.2,  2.19,  7.15,  7.69, -0.18,
          8.81, 23.1]


# Define the polynomial model of degree 3, i.e., having 3 weights and 1 bias. 
# Also define the loss function

def model(t, w1, w2, w3, b):
    return (t**3 * w1) + (t**2 * w2) + (t * w3) + b

def loss_fn(pred, act):
    diff = (pred - act)**2
    return diff.mean()

# Define gradient manually wrt the exisiting parameters
# Note: You need to define appropriate derivative functions to define the gradient
# Use the defined gradient function to define the training function 
# Note: You cannot use autograd and optimizers
# Run it on the input and output vector with appropriate learning rate and number of iterations
# Plot the learned curve


def dloss_fn(pred, act):
    dsq_diffs = 2*(pred - act) / pred.size(0)
    return dsq_diffs

# 3 * t**2 * w1 + 2 * t * w2 + 1 * t_u
def dmodel_dw1(t, w1, w2, w3, b):
    return 3 * w1 * t**2

def dmodel_dw2(t, w1, w2, w3, b):
    return 2 * t * w2

def dmodel_dw3(t, w1, w2, w3, b):
    return t

def dmodel_db(t, w1, w2, w3, b):
    return 1.0

def grad_fn(t, t_c, t_p, w1, w2, w3, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw1 = dloss_dtp * dmodel_dw1(t, w1, w2, w3, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t, w1, w2, w3, b)
    dloss_dw3 = dloss_dtp * dmodel_dw3(t, w1, w2, w3, b)
    dloss_db = dloss_dtp * dmodel_db(t, w1, w2, w3, b)
    return torch.stack([
        dloss_dw1.sum(),
        dloss_dw2.sum(),
        dloss_dw3.sum(),
        dloss_db.sum()
    ])



params = torch.tensor([1.0, 1.0, 1.0, 0.0], requires_grad=True)
t_u = torch.Tensor(inp) # convert inp to tensor and normalize
t_c = torch.Tensor(out) # Convert out to tensor and normalize

# Normalizing Tensors
t_u  = (t_u - t_u.mean()) / t_u.std()
t_c  = (t_c - t_c.mean()) / t_c.std()


# Use PyTorch's autograd to automatically compute the gradients 
# Define the training function
# Note: You cannot use optimizers.
# Run it on the input and output vector with appropriate learning rate and number of iterations
# Plot the learned curve

params = torch.tensor([1.0, 1.0, 1.0, 0], requires_grad=True)

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        if params.grad is not None:
            params.grad_zero()
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        
        with torch.no_grad():
            params -= learning_rate * params.grad
            
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params

training_loop(100, 0.01, params, t_u, t_c)
print(params)

# # Use PyTorch's autograd to automatically compute the gradients 
# # Use optimizers to abstract how parameters get updated
# # Define the training function
# # Run it on the input and output vector with appropriate learning rate, number of iterations, and SGD optimizer
# # Plot the learned curve
# import torch.optim as optim

# learning_rate = 1e-5
# params = torch.tensor([1.0, 1.0, 1.0, 0], requires_grad=True)
# optimizer = optim.SGD([params], lr=learning_rate) # SGD = stochastic gradient descent

# def training_loop(n_epochs, optimizer, params, t_u, t_c):
#     for epoch in range(1, n_epochs+1):
#         t_p = model(t_u, *params)
#         loss = loss_fn(t_p, t_c)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
            
#         if epoch % 500 == 0:
#             print('Epoch %d, Loss %f' % (epoch, float(loss)))

# training_loop(
#     n_epochs=500,
#     optimizer=optimizer,
#     params=params,
#     t_u=t_u,
#     t_c=t_c
# )

# fig = plt.figure(dpi=600)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(t_u.numpy(), t_p.detach().numpy(), 'o')
# plt.plot(t_u.numpy(), t_c.numpy(), 'o')