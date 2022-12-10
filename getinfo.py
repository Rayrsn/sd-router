import torch

from torchinfo import summary
#
model = torch.load('D:\stable-diffusion-webui\models\Stable-diffusion\sd-v1-4.ckpt', map_location='cpu')

# # display stable diffusion model information with torchinfo
# summary(model, input_size=(1, 3, 512, 896), depth=4, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), col_width=25)
#
# # display stable diffusion model information with torchsummary
# # from torchsummary import summary
# # summary(model, input_size=(1, 3, 512, 896))

# (“input_size”, “output_size”, “num_params”, “kernel_size”, “mult_adds”, “trainable” )
summary(model, input_size=(4), batch_dim=1, depth=1)