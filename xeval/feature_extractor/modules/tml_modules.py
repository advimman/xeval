import torch.nn as nn
import torch
import math


class MMLSTMCell(nn.Module):
    """Multi Modal LSTM cell"""
    def __init__(self, input_size, hidden_size, n_layers, n_modalities, dropout=None):
        r"""input_size is input size of modality (all modalities have the same inpuit size),
        n_layers is the same for each modality
        hidden_size is the same for each modality
        """
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_modalities = n_modalities

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.init_x_weights()
        self.init_hidden_weights()

    def init_x_weights(self):
        """For each modality we init separate weights"""
        self.w_ih = []
        for _ in range(self.n_modalities):
            ih = []
            for i in range(self.n_layers):
                if i == 0:
                    layer = nn.Linear(self.input_size, 4 * self.hidden_size)
                else:
                    layer = nn.Linear(4 * self.hidden_size, 4 * self.hidden_size)
                ih.append(layer)
            self.w_ih.append(nn.ModuleList(ih))
        self.w_ih = nn.ModuleList(self.w_ih)

    def init_hidden_weights(self):
        """For all modalities we share weights for hidden layers"""
        hh = []
        for i in range(self.n_layers):
            if i == 0:
                layer = nn.Linear(self.hidden_size, 4 * self.hidden_size)
            else:
                layer = nn.Linear(4 * self.hidden_size, 4 * self.hidden_size)
            hh.append(layer)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input_list, hidden_list):
        """input_list - list of X for each modality
        hidden_list - list of hidden_states for each modality
        """
        res = []
        for m in range(self.n_modalities):
            hy, cy = [], []
            for i in range(self.n_layers):
                hx, cx = hidden_list[m][0][i], hidden_list[m][1][i]
                gates = self.w_ih[m][i](input_list[m]) + self.w_hh[i](hx)
                i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                c_gate = torch.tanh(c_gate)
                o_gate = torch.sigmoid(o_gate)

                ncx = (f_gate * cx) + (i_gate * c_gate)
                nhx = o_gate * torch.tanh(ncx)
                cy.append(ncx)
                hy.append(nhx)
                if self.dropout is not None:
                    nhx = self.dropout(nhx)

            hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
            res.append((hy, cy))
        return res


class MMLSTM(nn.Module):
    """Modalities should have the same dims
    inputs dim = [Modality, Time, Batch, X_data]

    Outputs:
    y dims = [n_modalities, time, batch, x_dim]
    hidden = list of tuples. Len(list) == n_modalities. Each tuple dim = [n_layers, batch, x_dim]
    """
    def __init__(self, input_size, hidden_size, n_layers, n_modalities, dropout = None):
        super().__init__()
        self.cell = MMLSTMCell(input_size = input_size, hidden_size = hidden_size, n_layers = n_layers, n_modalities = n_modalities, dropout = dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = None
        self.n_modalities = n_modalities
        self.soft = nn.Softmax(dim = -1)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_list):
        use_cuda = input_list[0].is_cuda
        if use_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        res = [(torch.zeros(self.batch_size, self.hidden_size).type(dtype), torch.zeros(self.batch_size, self.hidden_size).type(dtype))] * self.n_modalities
        return res

    def forward(self, input_list, hidden_list = None):
        out = []

        if self.batch_size is None:
            self.batch_size = input_list[0].shape[1]
        if hidden_list is None:
            hidden_list = self.init_hidden(input_list)

        time_window = input_list[0].shape[0]
        for i in range(time_window):
            input_list_tmp = [x[i,:,:] for x in input_list]
            hidden_list = self.cell(input_list_tmp, hidden_list)

            # caculate output for each modality
            y = [self.soft(self.linear_out(hc[0][-1])) for hc in hidden_list] # list of outputs for each modality
            y = torch.cat([y_single.unsqueeze(dim = 0) for y_single in y], dim = 0).unsqueeze(dim = 0)
            out.append(y)
        out = torch.cat(out, dim = 0).permute(1,0,2,3)

        return out, hidden_list
