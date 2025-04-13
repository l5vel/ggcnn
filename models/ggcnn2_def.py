import torch
import torch.nn as nn
import torch.nn.functional as F


class GGCNN2(nn.Module):
    def __init__(self, input_channels=1, filter_sizes=None, l3_k_size=5, dilations=None):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [16,  # First set of convs
                            16,  # Second set of convs
                            32,  # Dilated convs
                            16]  # Transpose Convs

        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            # 4 conv layers.
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Dilated convolutions.
            nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[0], stride=1, padding=(l3_k_size//2 * dilations[0]), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[2], filter_sizes[2], kernel_size=l3_k_size, dilation=dilations[1], stride=1, padding=(l3_k_size//2 * dilations[1]), bias=True),
            nn.ReLU(inplace=True),

            # Output layers
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, xc, yc):
        # 1. Ensure input and model are on the same device
        xc = xc.to(self.features[0].weight.device)  # Move input to the first layer's weight device
        for param in self.parameters():
            param.data = param.data.to(xc.device) # Move all parameters to the same device as the input.
        # ... (rest of your forward pass)
        return self.compute_loss(xc, yc)

    def compute_loss(self, xc, yc):
        # ... (your compute_loss function)
        # Ensure targets are same dtype and device as predictions
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self.original_forward(xc)

        y_pos = y_pos.type_as(pos_pred).to(pos_pred.device)
        y_cos = y_cos.type_as(cos_pred).to(pos_pred.device)
        y_sin = y_sin.type_as(sin_pred).to(pos_pred.device)
        y_width = y_width.type_as(width_pred).to(width_pred.device)

        p_loss = F.mse_loss(pos_pred, y_pos, reduction='mean')  # Explicit mean
        cos_loss = F.mse_loss(cos_pred, y_cos, reduction='mean')
        sin_loss = F.mse_loss(sin_pred, y_sin, reduction='mean')
        width_loss = F.mse_loss(width_pred, y_width, reduction='mean')

        # 3. Combine Losses
        total_loss = p_loss + cos_loss + sin_loss + width_loss

        # 4. Construct Output Dictionary
        loss_dict = {
            'loss': total_loss.mean(),  # Mean for DataParallel
            'losses': {
                'p_loss': p_loss.mean(),
                'cos_loss': cos_loss.mean(),
                'sin_loss': sin_loss.mean(),
                'width_loss': width_loss.mean()
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

        # 5. Ensure Device Consistency (Critical!)
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_dict[key] = value.to(xc.device)  # Move to input device

        return loss_dict

