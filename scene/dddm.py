import torch
from typing import Callable
from polyfourier.poly import Polynomial
from polyfourier.fourier import Fourier
from polyfourier.poly_fourier import PolyFourier

class DDDMModel(torch.nn.Module):
    def __init__(self, type_name: str = "poly", feat_dim: int = 3, poly_factor: float = 1.0, Hz_factor: float = 1.0):
        super(DDDMModel, self).__init__()
        self.type_name = type_name
        self.feat_dim = feat_dim
        self.poly_factor = poly_factor
        self.Hz_factor = Hz_factor
        self.create_model()

    def create_model(self):
        if self.type_name == "fourier":
            self.trajectory_func = Fourier(
                self.feat_dim,
                Hz_base_factor=self.Hz_factor
            )
        elif self.type_name == "poly_fourier":
            self.trajectory_func = PolyFourier(
                self.feat_dim,
                poly_base_factor=self.poly_factor,
                Hz_base_factor=self.Hz_factor
            )
        elif self.type_name == "poly":
            self.trajectory_func = Polynomial(
                self.feat_dim,
                poly_base_factor=self.poly_factor,
            )
        else:
            self.trajectory_func = None
            print("Trajectory type not found")
    
    def forward(self, factors, timestamp, degree=1):
        if self.trajectory_func:
            return self.trajectory_func(factors, timestamp, degree)
        else:
            raise ValueError("Trajectory function not properly initialized")

    def get_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list
 