import torch
import polyfourier
import taichi as ti
ti.init(arch=ti.cuda)

num_points = 100
feature_dim = 3
output_dim = 2

# The parameters should be organized as a tensor of 
# shape (num_points, feature_dim, output_dim)
init_shape = (num_points, feature_dim, output_dim)
params = torch.nn.Parameter(torch.randn(init_shape, device='cuda'))
t_array = torch.linspace(0, 1, num_points).reshape(-1, 1).cuda()

# type_name should be 'poly', 'fourier' and 'poly_fourier'
fit_model = polyfourier.get_fit_model(type_name='poly')


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD([params], lr=0.01)

# Training loop for 10- iterations
for epoch in range(1000):
    # Forward pass
    output = fit_model(params, t_array, feature_dim)

    # Initialize target as ones
    target = torch.ones(output.shape).cuda()
    loss = loss_fn(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # print("Gradient of params:", params.grad)

# Verify final gradients
# print("Final gradient of params:", params.grad)
print(output)