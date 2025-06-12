import torch
import pandas
import yfinance
# print("PyTorch version:", torch.__version__)
import torch.nn as nn

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

linear_layer = nn.Linear(
    in_features=3,
    out_features=2
)

output = linear_layer(input_tensor)
print(output)