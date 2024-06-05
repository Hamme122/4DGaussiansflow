import torch
import polyfourier
import taichi as ti

# Initialize Taichi with CUDA
ti.init(arch=ti.cuda)

# Set up parameters
num_points = 100
feature_dim = 5
output_dim = 10

# Initialize the DDDMModel with the type_name as 'poly'
fit_model = polyfourier.DDDMModel(type_name="poly", feat_dim=feature_dim,num_points= num_points, output_dim= output_dim)

fit_model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fit_model.get_dddm_parameters(), lr=0.001)

for epoch in range(1000):
    # Forward pass
    output = fit_model()
    
    # Initialize target as ones
    target = torch.zeros(output.shape).cuda()
    
    # Compute the loss
    loss = loss_fn(output, target)
    
    # Backward pass
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()        # Backpropagate the loss
    optimizer.step()       # Update the model parameters
    
    # Print the loss for each epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Print the final output
print(output)
