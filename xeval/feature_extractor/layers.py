import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()

        self._height, self._width = input_size
        self._hidden_dim = hidden_dim

        self._conv = nn.Conv2d(in_channels=input_dim + self._hidden_dim,
                               out_channels=4 * self._hidden_dim,
                               kernel_size=kernel_size,
                               padding=(kernel_size[0] // 2, kernel_size[1] // 2))

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self._conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self._hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self._hidden_dim, self._height, self._width)).cuda(),
                Variable(torch.zeros(batch_size, self._hidden_dim, self._height, self._width)).cuda())


class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels_list, kernel_size):
        super(ConvLSTM, self).__init__()

        self._num_layers = len(hidden_channels_list)

        cell_list = []
        for i in range(0, self._num_layers):
            cur_input_dim = input_channels if i == 0 else hidden_channels_list[i - 1]

            cell_list.append(ConvLSTMCell(input_size=input_size,
                                          input_dim=cur_input_dim,
                                          hidden_dim=hidden_channels_list[i],
                                          kernel_size=kernel_size))
            input_size = [size // 2 for size in input_size]

        self._cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self._num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self._cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                  cur_state=[h, c])
                output_inner.append(F.max_pool2d(h, 2))

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            last_state_list.append([h, c])

        return last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self._num_layers):
            init_states.append(self._cell_list[i].init_hidden(batch_size))
        return init_states


class ImagesEncoder(nn.Module):
    def __init__(self, feature_size, input_size, input_channels, hidden_channels_list, kernel_size):
        super().__init__()

        self._conv_lstm = ConvLSTM(
            input_size=input_size,
            input_channels=input_channels,
            hidden_channels_list=hidden_channels_list,
            kernel_size=kernel_size,
        )
        self._pooling = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(hidden_channels_list[-1] * 2, feature_size)

    def forward(self, inputs, lengths):
        inputs = inputs.permute(0, 1, 4, 2, 3)
        states = self._conv_lstm(inputs)
        h, c = states[-1]
        x = torch.cat([h, c], dim=1)
        x = self._pooling(x).view(x.shape[0], -1)
        return self._fc(x)
