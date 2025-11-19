from src.modeling.components import AbstractModelComponent
import torch
import torch.nn as nn

class FrameIntakeBlock(AbstractModelComponent):
    """A block that takes in a batch of dimensions (batch_size, max_sentence_time, brain_regions, channels, x, y) and shape (N, T, 4, 3, 8, 8), and outputs a batch of dimensions (batch_size, max_sentence_time, pseudo_RGB_channels, x, y) and shape (N, T, 3, 224, 224).

    The IntakeBlock first divides the tensor into 4 tensors of shape (N, T, 1, 3, 8, 8). It removes the extra dimension, yielding a tensor of shape (N, T, 3, 8, 8). Then, it spatially upsamples each of the 4 tensors while pooling the 5 channels into 3 channels, yielding (N, T, 3, 32, 32). Next, it concatenates the 4 tensors in a window-pane fashion, yielding a tensor of shape (N, T, 3, 64, 64). Finally, it upsamples the tensor to (N, T, 3, 224, 224).
    """
    def __init__(self, dropout_rate=0.2, use_attention=True):
        super(FrameIntakeBlock, self).__init__()
        self.upsample_inferior_44 = UpsampleBrainRegionBlock(in_channels=3, out_channels=3, output_size=32, use_attention=use_attention, dropout_rate=dropout_rate)
        self.upsample_superior_44 = UpsampleBrainRegionBlock(in_channels=3, out_channels=3, output_size=32, use_attention=use_attention, dropout_rate=dropout_rate)
        self.upsample_inferior_6v = UpsampleBrainRegionBlock(in_channels=3, out_channels=3, output_size=32, use_attention=use_attention, dropout_rate=dropout_rate)
        self.upsample_superior_6v = UpsampleBrainRegionBlock(in_channels=3, out_channels=3, output_size=32, use_attention=use_attention, dropout_rate=dropout_rate)
        self.upsample_x = UpsampleBrainRegionBlock(in_channels=3, out_channels=3, output_size=224, use_attention=False, dropout_rate=0.0)


    def forward(self, x, tokens=None, kv_cache=None):
        """Forward pass of the IntakeBlock.

        Args:
            x (torch.Tensor): A batch of dimensions (batch_size, max_sentence_time, brain_regions, channels, x, y) and shape (N, T, 4, 3, 8, 8).

        Returns:
            torch.Tensor: A batch of dimensions (batch_size, max_sentence_time, RGB_channels, x, y) and shape (N, T, 3, 224, 224).
        """
        # Reshape for 2D convolutional operations, (N, T, 4, 3, 8, 8) -> (N*T, 4, 3, 8, 8)
        # TODO: try with only (N, T, 4, 1, 8, 8) with only threshold crossing
        N = x.shape[0]
        T = x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4], x.shape[5])

        # Divide the tensor into 4 tensors of shape (N*T, 1, 3, 8, 8)
        inferior_6v = x[:, 0, :, :, :]
        superior_6v = x[:, 1, :, :, :]
        inferior_44 = x[:, 2, :, :, :]
        superior_44 = x[:, 3, :, :, :]

        # Remove the extra dimension, (N*T, 1, 3, 8, 8) -> (N*T, 3, 8, 8)
        inferior_6v = inferior_6v.squeeze(2)
        superior_6v = superior_6v.squeeze(2)
        inferior_44 = inferior_44.squeeze(2)
        superior_44 = superior_44.squeeze(2)

        # Upsample each of the 4 tensors, (N*T, 3, 8, 8) -> (N*T, 3, 32, 32)
        inferior_6v = self.upsample_inferior_6v(inferior_6v)
        superior_6v = self.upsample_superior_6v(superior_6v)
        inferior_44 = self.upsample_inferior_44(inferior_44)
        superior_44 = self.upsample_superior_44(superior_44)

        # Concatenate the 4 tensors in a window-pane fashion, (N*T, 3, 32, 32) -> (N*T, 3, 64, 64)

        # Concatenate superior 6v on top of inferior 6v, (N*T, 3, 32, 32) -> (N*T, 3, 32, 64)
        area_6v = torch.cat((superior_6v, inferior_6v), dim=3)
        # Concatenate superior 44 on top of inferior 44, (N*T, 3, 32, 32) -> (N*T, 3, 32, 64)
        area_44 = torch.cat((superior_44, inferior_44), dim=3)
        # Concatenate area 44 to the left of area 6v, (N*T, 3, 32, 64) -> (N*T, 3, 64, 64)
        x = torch.cat((area_44, area_6v), dim=2)

        # Expand the tensor x, (N*T, 3, 64, 64) -> (N*T, 3, 224, 224)
        x = self.upsample_x(x)

        # Reshape for 3D convolutional operations, (N*T, 3, 224, 224) -> (N, T, 3, 224, 224)
        x = x.view(N, T, x.shape[1], x.shape[2], x.shape[3])

        return x

    @property
    def input_shape(self):
        return (4, 3, 8, 8)

    @property
    def output_shape(self):
        return (3, 224, 224)

    @property
    def is_output_layer(self):
        return False

    def name(self):
        return "FrameIntakeBlock"

class UpsampleBrainRegionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, use_attention=True, dropout_rate=0.0):
        super(UpsampleBrainRegionBlock, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.expander1 = UpsampleBlock(in_channels=in_channels, out_channels=out_channels, output_size=output_size // 2, use_attention=use_attention)
        self.expander2 = UpsampleBlock(in_channels=out_channels, out_channels=out_channels, output_size=output_size, use_attention=False)

    def forward(self, x):
        x = self.dropout(x)
        x = self.expander1(x)
        x = self.expander2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, use_attention=False):
        super(UpsampleBlock, self).__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialSelfAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU() #nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.GELU() #nn.ReLU()
        self.upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        if self.use_attention:
            x = self.attention(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.upsample(x)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return out + x  # Skip Connection