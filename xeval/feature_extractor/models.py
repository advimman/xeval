import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50

from ..core.constants import FEATURE_EXTRACTOR_INPUT_SIZE
from ..core.utils import load_class
from .modules import tml_modules as tml_nn
from .modules import modules as modules_nn


class AEModelBase(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, inputs, lengths):
        features = self._encoder(inputs, lengths)
        x = self._decoder(features, lengths)
        return x

    def get_features(self, inputs, lengths):
        return self._encoder(inputs, lengths)


class LSTMEncoder(nn.Module):
    def __init__(self, lstm_hidden_size, lstm_num_layers, is_lstm_bidirectional,
                 lstm_dropout, features_size):
        super().__init__()
        self._lstm_impl = nn.LSTM(
            input_size=FEATURE_EXTRACTOR_INPUT_SIZE,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=is_lstm_bidirectional,
            dropout=lstm_dropout,
            batch_first=True
        )
        num_directions = 1 if not is_lstm_bidirectional else 2
        self._fc = nn.Linear(2 * lstm_num_layers * num_directions * lstm_hidden_size, features_size)

    def forward(self, inputs, lengths):
        x = inputs[0]
        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        x = torch.nn.utils.rnn.pack_padded_sequence(x[perm_idx], seq_lengths, batch_first=True)
        _, (h, c) = self._lstm_impl(x)
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        h = h.contiguous().view(h.shape[0], -1)
        c = c.contiguous().view(c.shape[0], -1)
        x = torch.cat([h, c], dim=1)
        return self._fc(x)[inverse_idx]


class LSTMDecoder(nn.Module):
    def __init__(self, features_size, lstm_hidden_size, lstm_num_layers, is_lstm_bidirectional, lstm_dropout):
        super().__init__()
        self._lstm_impl = nn.LSTM(
            input_size=features_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=is_lstm_bidirectional,
            dropout=lstm_dropout)

        num_directions = 1 if not is_lstm_bidirectional else 2
        self._fc = nn.Linear(lstm_hidden_size * num_directions, FEATURE_EXTRACTOR_INPUT_SIZE)

    def forward(self, features, lengths):
        x = features.unsqueeze(1).repeat(1, lengths.max(), 1)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        x = torch.nn.utils.rnn.pack_padded_sequence(x[perm_idx], seq_lengths, batch_first=True)

        x, _ = self._lstm_impl(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return self._fc(x)[inverse_idx],


class CNNEncoder(nn.Module):
    def __init__(self, cnn_channels_list, cnn_kernel_size, cnn_padding, features_size):
        super().__init__()

        cnn_channels_list.append(features_size)
        input_channels = FEATURE_EXTRACTOR_INPUT_SIZE
        layers = []
        for output_channels in cnn_channels_list:
            layer = nn.Conv1d(input_channels, output_channels, cnn_kernel_size, padding=cnn_padding)
            layers.append(layer)
            layers.append(nn.ReLU())
            input_channels = output_channels

        self._cnn_layers = nn.ModuleList(layers)
        self._pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs, lengths):
        x = inputs[0]
        x = x.permute(0, 2, 1)

        for layer in self._cnn_layers:
            x = layer(x)
        x = self._pooling(x).squeeze(2)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, features_size, cnn_channels_list, cnn_kernel_size, cnn_padding):
        super().__init__()

        cnn_channels_list.append(FEATURE_EXTRACTOR_INPUT_SIZE)
        input_channels = features_size
        layers = []
        for output_channels in cnn_channels_list:
            layer = nn.Conv1d(input_channels, output_channels, cnn_kernel_size, padding=cnn_padding)
            layers.append(layer)
            layers.append(nn.ReLU())
            input_channels = output_channels

        layers = layers[:-1]
        self._cnn_layers = nn.ModuleList(layers)

    def forward(self, features, lengths):
        x = features.unsqueeze(1).repeat(1, lengths.max(), 1)
        x = x.permute(0, 2, 1)

        for layer in self._cnn_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        return x,

class PCAEncoder(nn.Module):
    def __init__(self, fc_size_list, input_size, features_size):
        super().__init__()
     
        fc_size_list.append(features_size)

        layers = []
        for i, output_size in enumerate(fc_size_list):
            layer = nn.Linear(input_size, output_size)
            layers.append(layer)
                   
        layers.append(nn.BatchNorm1d(output_size))
        self._fc_layers = nn.ModuleList(layers)
        self._features_size = features_size
        
    def forward(self, inputs, lengths):
        x = inputs
        for layer in self._fc_layers:
            x = layer(x) 
            
        return x

    
class PCADencoder(nn.Module):
    def __init__(self, fc_size_list, output_sizes, features_size):
        super().__init__()
     
        fc_size_list.append(output_sizes[0] * output_sizes[1] * output_sizes[2])
        input_size = features_size
        
        layers = []
        for output_size in fc_size_list:
            layer = nn.Linear(input_size, output_size)
            layers.append(layer)
            input_size = output_size
            
        layers.append(nn.Sigmoid())
        self._fc_layers = nn.ModuleList(layers)
        self._features_size = features_size
        self._output_sizes = output_sizes
        
    def forward(self, inputs, lengths):
        x = inputs
        for layer in self._fc_layers:
            x = layer(x) 
            
        x = x.view(x.size(0), 
                   self._output_sizes[0],
                   self._output_sizes[1], 
                   self._output_sizes[2])    
        return x

    
class Resnet34Encoder(nn.Module):
    def __init__(self, fc_size_list, features_size):
        super().__init__()
        self.cnn = resnet34(pretrained=True)
        self.cnn.fc = nn.Dropout(0)
        self.cnn.avgpool = nn.AdaptiveAvgPool2d(3)
        
        fc_size_list.append(features_size)

        layers = []
        input_channels = 4608
        for output_channels in fc_size_list:
            layer = nn.Linear(input_channels, output_channels)
            layers.append(layer)
            layers.append(nn.ReLU())
            input_channels = output_channels
        
        self._fc_layers = nn.ModuleList(layers)
        self._features_size = features_size
        
    def forward(self, inputs, lengths):
        x = inputs
        x = self.cnn(x)
        for layer in self._fc_layers:
            x = layer(x) 
        return x
    
    
class CNN2DDecoder(nn.Module):
    def __init__(self, features_size, cnn_channels_list, cnn_new_channels_list, cnn_kernel_size, cnn_padding):
        super().__init__()
        layers = []         
        input_channels = features_size
        
        layers = []
        for output_channels in cnn_channels_list:
            layer = nn.ConvTranspose2d(input_channels, output_channels, cnn_kernel_size, stride=2, padding=1, output_padding=1)
            layers.append(layer)
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU())
            input_channels = output_channels
        
        cnn_new_channels_list.append(3)
        for output_channels in cnn_new_channels_list:
            layer = nn.Conv2d(input_channels, output_channels, cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2)
            layers.append(layer)
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU())
            input_channels = output_channels
        
        layers = layers[:-2]
        layers.append(nn.Sigmoid())
        self._cnn_layers = nn.ModuleList(layers)

    def forward(self, features, lengths):
        x = features
        x = x.view(x.size(0), x.size(1), 1 , 1)
        for layer in self._cnn_layers:
            x = layer(x)
        return x


class LSTMAutoEncoder(AEModelBase):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(LSTMEncoder(**encoder_kwargs), LSTMDecoder(**decoder_kwargs))


class CNNAutoEncoder(AEModelBase):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(CNNEncoder(**encoder_kwargs), CNNDecoder(**decoder_kwargs))


class PCAAutoEncoder(AEModelBase):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(PCAEncoder(**encoder_kwargs), PCADencoder(**decoder_kwargs))
     
    def forward(self, inputs, lengths):
        inputs = inputs[1]
        inputs = self.preprocessor(inputs)
        return self.post_processor(super(PCAAutoEncoder, self).forward(inputs, 1))
        
    def preprocessor(self, inputs):
        """Resize X to feature_size; resize pictures to feature_size."""
        inputs = inputs.view(inputs.shape[0], -1)
        return inputs
    
    def post_processor(self, output):
        output = output.permute(0, 2, 3, 1)
        return output
        
    def get_features(self, inputs, lengths):
        inputs = inputs[1]
        features = []
        for el in inputs:
            inputs = self.preprocessor(el)
            features.append(self._encoder(inputs, 1).mean(0))
        return features   
        
        
class Resnet34AutoEncoder(AEModelBase):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(Resnet34Encoder(**encoder_kwargs), CNN2DDecoder(**decoder_kwargs))
     
    def forward(self, inputs, lengths):
        inputs = inputs[1]
        inputs = self.preprocessor(inputs)
        return self.post_processor(super(Resnet34AutoEncoder, self).forward(inputs, 1))
        
    def preprocessor(self, inputs):
        """Resize X to feature_size; resize pictures to feature_size."""
        inputs = inputs.permute(0, 3, 1, 2)
        return inputs
    
    def post_processor(self, output):
        output = output.permute(0, 2, 3, 1)
        return output
    
    def get_features(self, inputs, lengths):
        inputs = inputs[1]
        features = []
        for el in inputs:
            inputs = self.preprocessor(el)
            features.append(self._encoder(inputs, 1).mean(0))
        return features

        
class TimeSeriesRegressor(nn.Module):
    def __init__(self, time_distance, features_size, fc_layers, encoder):
        super().__init__()
        self._time_distance = time_distance
        self._encoder = encoder
        layers = []
        input_features = features_size
        for _ in range(fc_layers):
            layers.append(nn.Linear(input_features, FEATURE_EXTRACTOR_INPUT_SIZE))
            layers.append(nn.ReLU())
            input_features = FEATURE_EXTRACTOR_INPUT_SIZE

        self._fc_layers = nn.Sequential(*layers[:-1])

    def forward(self, inputs, lengths):
        if self.training:
            inputs = [modality.clone() for modality in inputs]
            for modality in inputs:
                for i in range(len(modality)):
                    s = modality[i, lengths[i] - 1].sum()
                    if s == 0:
                        print(modality[i].sum())
                    modality[i, lengths[i] - self._time_distance:] = 0

            lengths = lengths.clone() - self._time_distance

        features = self._encoder(inputs, lengths)
        return self._fc_layers(features),

    def get_features(self, inputs, lengths):
        return self._encoder(inputs, lengths)


class AttentionEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, layers_number):
        super().__init__()

        self._feature_size = feature_size
        self._linear_layers = nn.ModuleList(
            [nn.Linear(FEATURE_EXTRACTOR_INPUT_SIZE + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(layers_number)])
        self._final_fc = nn.Linear(FEATURE_EXTRACTOR_INPUT_SIZE + hidden_size * (layers_number - 1), feature_size)

    def forward(self, inputs, lengths):
        x = inputs[0]

        for linear_layer, attention_layer in zip(self._linear_layers, self._attention_layers):
            vectors = linear_layer(x)
            vectors = F.relu(vectors)
            weights = F.softmax(attention_layer(vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)

            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            x = torch.cat([x, vector], dim=2)

        x = x.mean(dim=1)
        return self._final_fc(x)


class MultimodalEncoder(nn.Module):
    def __init__(self, state_encoder_class=None, state_encoder_kwargs=None, images_encoder_class=None, images_encoder_kwargs=None):
        super().__init__()

        assert state_encoder_class or images_encoder_class
        state_encoder_kwargs = state_encoder_kwargs if state_encoder_kwargs is not None else {}
        images_encoder_kwargs = images_encoder_kwargs if images_encoder_kwargs is not None else {}
        self._state_encoder = None if state_encoder_class is None else load_class(state_encoder_class)(**state_encoder_kwargs)
        self._images_encoder = None if images_encoder_class is None else load_class(images_encoder_class)(**images_encoder_kwargs)

    def forward(self, inputs, lengths):
        result = []
        if self._state_encoder is not None:
            result.append(self._state_encoder([inputs[0]], lengths))

        if self._images_encoder is not None:
            result.append(self._images_encoder(inputs[1], lengths))

        return torch.cat(result, dim=1)


class TML(nn.Module):
    """Deep Multimodal Representation Learning from Temporal Data
       https://arxiv.org/abs/1704.03152
    """
    def __init__(self, feature_size,
                       lstm_input_size,
                       lstm_hidden_size,
                       lstm_n_modalities,
                       lstm_n_layers,
                       lstm_dropout=None,
                       cnn_pretrained=True):
        super().__init__()
        self.feature_size = feature_size
        self.X_size = FEATURE_EXTRACTOR_INPUT_SIZE
        self.n_modalities = 2

        self.lstm_encoder = tml_nn.MMLSTM(input_size=lstm_input_size,
                                      hidden_size=lstm_hidden_size,
                                      n_layers=lstm_n_layers,
                                      n_modalities = lstm_n_modalities,
                                      dropout = lstm_dropout)
        self.lstm_decoder = tml_nn.MMLSTM(input_size=lstm_hidden_size,
                                      hidden_size=lstm_input_size,
                                      n_layers=lstm_n_layers,
                                      n_modalities = lstm_n_modalities,
                                      dropout = lstm_dropout)
        self.x_enc_resizer = nn.Linear(self.X_size, self.feature_size)
        self.x_dec_resizer = nn.Linear(self.feature_size, self.X_size)
        var_dim = 256 // self.feature_size
        self.x_dec_cnn = modules_nn.DecoderBlock(in_channels=1,
                                            out_channels=3,
                                            kernel_size=(var_dim, var_dim),
                                            stride=var_dim,
                                            padding=0,
                                            conv_channels=16)

        self.cnn = resnet18(pretrained=cnn_pretrained)
        self.cnn.fc = nn.Linear(2048, feature_size)

    def preprocessor(self, inputs):
        """Resize X to feature_size; resize pictures to feature_size."""
        x_vectors = []
        picture_vectors = []
        x, pictures = inputs[0], inputs[1]
        time_window = x.shape[1]

        for time in range(time_window):
            x_val = self.x_enc_resizer(x[:,time,:])
            x_vectors.append(x_val.unsqueeze(dim=1))
            pic_val = self.cnn(pictures[:, time, ::].permute(0, 3, 1, 2))
            picture_vectors.append(pic_val.unsqueeze(dim=1))

        res = [torch.cat(x_vectors, dim = 1), torch.cat(picture_vectors, dim=1)]
        return res

    def apply_encoder(self, inputs):
        inputs = [x.permute(1,0,2) for x in inputs] # make it (T, B, :)
        features_list, hidden_list = self.lstm_encoder(inputs)
        return features_list, hidden_list

    def post_processor(self, out_list):
        """Reconstruct from lstm decoder to X and Img."""
        # Handle X
        out_list[0] = self.x_dec_resizer(out_list[0])

        # Handle Images
        out_list[1] = out_list[1].unsqueeze(dim=-2).unsqueeze(dim=-2)
        out_list[1] = torch.cat([out_list[1]] * self.feature_size, dim=-2)
        dims_original = out_list[1].shape
        out_list[1] = out_list[1].view(dims_original[0]*dims_original[1],
                                       dims_original[2],
                                       dims_original[3],
                                       dims_original[4])
        out_list[1] = self.x_dec_cnn(out_list[1])
        dims_cnn = out_list[1].shape
        out_list[1] = out_list[1].view(dims_original[0],
                                       dims_original[1],
                                       dims_cnn[-3],
                                       dims_cnn[-2],
                                       dims_cnn[-1])

        return out_list

    def forward(self, inputs, lengths):
        """Inputs - list of inputs for n_modalities (first - aux, second - pic).
           Model internal modality_dim = [Time, Batch, X].
        """
        inputs = self.preprocessor(inputs)
        features_list, hidden_list = self.apply_encoder(inputs)
        out_list, _ = self.lstm_decoder(features_list, hidden_list)

        # swap axis
        features_list = self.swap_t_b(features_list)
        out_list = self.swap_t_b(out_list)
        hidden_list = self.swap_t_b_tuples(hidden_list)

        out_list = self.post_processor(out_list)

        res = [features_list, out_list, hidden_list]
        return res

    def get_features(self, inputs, lengths):
        inputs = self.preprocessor(inputs)
        features_list, hidden_list = self.apply_encoder(inputs)
        features_list = self.swap_t_b(features_list)
        hidden_list = self.swap_t_b_tuples(hidden_list)
        return hidden_list[0][0][:, -1, :]

    @staticmethod
    def swap_t_b(list_):
        return [x.permute(1,0,2) for x in list_]

    @staticmethod
    def swap_t_b_tuples(list_):
        return [(h1.permute(1,0,2), h2.permute(1,0,2)) for (h1,h2) in list_ ]


class DecoupleDynamicsModel(nn.Module):
    """
    Paper Decoupling Dynamics and Reward for Transfer Learning
    https://arxiv.org/pdf/1804.10689

    ------
    2 variants:
    - original paper on 1 modality (X)
    - modified variant with images
    """
    def __init__(self, use_images,
                       use_X,
                       action_size,
                       feature_size,
                       lstm_input_size,
                       lstm_hidden_size,
                       lstm_n_layers=1,
                       lstm_is_bidirectional=False,
                       lstm_dropout=0,
                       cnn_pretrained=True):
        super().__init__()
        self.use_images = use_images
        self.use_X = use_X
        self.X_size = FEATURE_EXTRACTOR_INPUT_SIZE - action_size
        self.action_size = action_size
        self.feature_size = feature_size
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_n_layers = lstm_n_layers
        self.lstm_is_bidirectional = lstm_is_bidirectional
        self.lstm_dropout = lstm_dropout
        self.cnn_pretrained = cnn_pretrained

        self.encoder_mix_modalities = nn.Linear(self.feature_size * 2, self.feature_size)
        self.rnn = nn.LSTM(input_size = self.lstm_input_size,
                           hidden_size = self.lstm_hidden_size,
                           num_layers = self.lstm_n_layers,
                           dropout = self.lstm_dropout)
        self.rnn.flatten_parameters()

        self.action_linear = nn.Linear(self.feature_size + self.action_size, self.feature_size)

        if self.use_X:
            self.encoder_modules = nn.ModuleList([modules_nn.NeuralTree([self.X_size, 1024, 1024, self.feature_size])])
            self.decoder_modules = nn.ModuleList([modules_nn.NeuralTree([self.feature_size, 1024, 1024, self.X_size])])
            self.im_index = 1
        else:
            self.encoder_modules = nn.ModuleList([])
            self.decoder_modules = nn.ModuleList([])
            self.im_index = 0
        if self.use_images:
            self.encoder_modules.append(self.generate_encoder_cnn())
            self.decoder_modules.append(self.generate_decoder_cnn())

    def generate_encoder_cnn(self):
        cnn = resnet18(pretrained = self.cnn_pretrained)
        cnn.fc = nn.Linear(2048, self.feature_size)
        return cnn

    def generate_decoder_cnn(self):
        var_dim = 256 // self.feature_size
        cnn = modules_nn.DecoderBlock(in_channels=1,
                                 out_channels=3,
                                 kernel_size=(var_dim, var_dim),
                                 stride=var_dim,
                                 padding=0,
                                 conv_channels=16)
        return cnn

    def encoder(self, state):
        res = []
        len_states = len(state)
        for m in range(len_states):
            val = self.encoder_modules[m](state[m])
            res.append(val)

        res = torch.cat(res, dim=-1)
        if len_states > 1:
            res = self.encoder_mix_modalities(res)
        return res

    def decoder(self, embeds):
        res = []
        for m in range(len(self.decoder_modules)):
            x = embeds
            if m == self.im_index:
                x = embeds.unsqueeze(dim=-2).unsqueeze(dim=-2)
                x = x.expand(embeds.shape[0],
                             1,
                             self.feature_size,
                             embeds.shape[-1])


            val = self.decoder_modules[m](x)
            res.append(val)

        return res

    def formate_states(self, inputs, t, return_st1):
        if self.use_X:
            s_t0 = [inputs[0][:, t, :]]
            if return_st1:
                s_t1 = [inputs[0][:, t+1, :]]
        else:
            s_t0 = []
            if return_st1:
                s_t1 = []

        if self.use_images:
            s_t0.append(inputs[self.im_index][:, t, ::].permute(0, 3, 1, 2))
            if return_st1:
                s_t1.append(inputs[self.im_index][:, t+1, ::].permute(0, 3, 1, 2))
        if return_st1:
            return s_t0, s_t1
        else:
            return s_t0

    def forward(self, inputs, lengths):
        """Original paper:
            t0 - (t)
            t1 - (t+1)
            a - action
            s - state
            z - embeding
            _pred - prediction of forward model
        """
        res = []
        for t in range(max(lengths)-1):
            s_t0, s_t1 = self.formate_states(inputs=inputs, t=t, return_st1=True)
            a_t0 = inputs[-1][:, t, :]
            z_t0 = self.encoder(s_t0)
            z_t1 = self.encoder(s_t1)

            # Forward model
            z_t0_with_actions = self.action_linear(torch.cat([z_t0,a_t0], dim=-1))
            z_t1_pred, (h,c) = self.rnn(z_t0_with_actions.unsqueeze(dim=1))
            z_t1_pred = z_t1_pred.squeeze(dim=1)
            s_t1_pred = self.decoder(z_t1_pred)
            s_t0_pred = self.decoder(z_t0)
            res_dict = {"state_t0": s_t0,
                        "state_t0_pred": s_t0_pred,
                        "state_t1": s_t1,
                        "state_t1_pred": s_t1_pred,
                        "embed_t1": z_t1,
                        "embed_t1_pred": z_t1_pred,
                        "action_t0": a_t0}
            res.append(res_dict)
        return res

    def get_features(self, inputs, lengths):
        time_window = inputs[0].shape[1]
        res_z = []

        for t in range(time_window):
            s_t0 = self.formate_states(inputs=inputs, t=t, return_st1=False)
            z_t0 = self.encoder(s_t0)
            res_z.append(z_t0.unsqueeze(dim=0))

        # take mean across all t
        embed = torch.cat(res_z, dim=0)
        embed = torch.mean(embed, dim=0)
        return embed


class ContextRegressor(nn.Module):
    def __init__(self, context_size, features_size, fc_layers, encoder):
        super().__init__()
        self._context_size = context_size
        self._encoder = encoder

        self._fc_layers = []

        for _ in range(context_size * 2):
            layers = []
            input_features = features_size
            for _ in range(fc_layers):
                layers.append(nn.Linear(input_features, FEATURE_EXTRACTOR_INPUT_SIZE))
                layers.append(nn.ReLU())
                input_features = FEATURE_EXTRACTOR_INPUT_SIZE

            self._fc_layers.append(nn.Sequential(*layers[:-1]))

        self._fc_layers = nn.ModuleList(self._fc_layers)

    def get_features(self, inputs, lengths):
        return self._encoder(inputs, lengths)

    def forward(self, inputs, lengths):
        if self.training:
            use_cuda = lengths.is_cuda
            inputs_without_context = []
            for modality in inputs:
                shape = list(modality.shape)
                shape[1] -= self._context_size * 2

                modality_without_context = torch.zeros(*shape)
                if use_cuda:
                    modality_without_context = modality_without_context.cuda()

                inputs_without_context.append(modality_without_context)

                for example_index, example in enumerate(modality):
                    inputs_without_context[-1][example_index, :lengths[example_index] - self._context_size * 2] = \
                        example[self._context_size:lengths[example_index] - self._context_size]

            inputs = inputs_without_context
            lengths = lengths - self._context_size * 2

        features = self._encoder(inputs, lengths)

        result = []
        for layer in self._fc_layers:
            result.append(layer(features).unsqueeze(1))

        result = torch.cat(result, dim=1)
        return result,


class AlmostIdentityModule(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self._output_size = output_size
        self._fake_parameter = nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, inputs):
        input_size = inputs.size()[-1]
        if input_size < self._output_size:
            inputs = inputs.repeat(1, math.ceil(self._output_size / input_size))
        inputs = inputs[:, :self._output_size] * self._fake_parameter
        return inputs


class AlmostIdentityEncoder(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self._impl = AlmostIdentityModule(features_size)

    def forward(self, inputs, lengths):
        inputs = inputs[0][:, 0]  # use only the first modality and the first timeframe
        outputs = self._impl(inputs)
        return outputs


class AlmostIdentityDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._impl = AlmostIdentityModule(FEATURE_EXTRACTOR_INPUT_SIZE)

    def forward(self, inputs, lengths):
        outputs = self._impl(inputs).unsqueeze(1),
        return outputs


class AlmostIdentityAutoEncoder(AEModelBase):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__(AlmostIdentityEncoder(**encoder_kwargs), AlmostIdentityDecoder(**decoder_kwargs))
