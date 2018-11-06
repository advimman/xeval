import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools

from ..core.constants import ACTION_INDS


class ReconstructionMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicted_modalities, input_modalities, lengths):
        loss = 0
        count = 0
        for predicted_modality, input_modality in zip(predicted_modalities, input_modalities):
            for predicted_modality_example, input_modality_example, length in zip(predicted_modality,
                                                                                  input_modality,
                                                                                  lengths):
                loss += F.mse_loss(predicted_modality_example[:length], input_modality_example[:length])
                count += 1
        return loss / count


class TimeSeriesRegressionMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicted_modalities, input_modalities, lengths):
        loss = 0
        count = 0
        for predicted_modality, input_modality in zip(predicted_modalities, input_modalities):
            for predicted_modality_example, input_modality_example, length in zip(predicted_modality,
                                                                                  input_modality,
                                                                                  lengths):
                loss += F.mse_loss(predicted_modality_example, input_modality_example[length - 1])
                count += 1
        return loss / count

    
class AELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, model_output, input_modalities, lengths):
        loss = 0
        loss += self.mse(model_output, input_modalities[1])
        return loss 


class TMLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.corr_lambda = kwargs['corr_lambda']
        self.modalities_lambda = kwargs['modalities_lambda']

    def forward(self, model_output, input_modalities, lengths):
        features_modalities = model_output[0]
        predicted_modalities = model_output[1]
        hidden_states_modalities = model_output[2]

        fused_reconstruction_loss_val = self.fused_reconstruction_loss(predicted_modalities, input_modalities, lengths)
        corr_loss_val = self.corr_loss(hidden_states_modalities, lengths)

        final_loss = fused_reconstruction_loss_val - self.corr_lambda * corr_loss_val
        return final_loss

    def corr_loss(self, hidden_states_modalities, lengths):
        """Computes correlation between all modality pairs"""
        # first_index = modality, second_index = hidden state
        # the dims opf H1 and H2 are [n_layers, batch, x_dim]
        hidden = [hidden_states_modalities[0][0], hidden_states_modalities[1][0]]
        batch_size = hidden[0].shape[1]

        if batch_size > 1:
            v_hidden = [x - torch.mean(x, dim = 1) for x in hidden]
        else:
            v_hidden = hidden

        # for each pair compute correlation
        res = [self.correlation_func(index_tuple = comb, hidden = hidden, v_hidden = v_hidden) \
                for comb in itertools.combinations(range(len(hidden)), 2)]

        res = sum(res) / len(res)
        return res

    def fused_reconstruction_loss(self, predicted_modalities, input_modalities, lengths):
        """(x,y) are different modalities. Reconstruction of x and y from (x,y)"""
        loss = 0
        i = 0
        loss_functions = [nn.MSELoss(), self.img_loss]

        for m, (predicted_modality, input_modality) in enumerate(zip(predicted_modalities, input_modalities)):
            for predicted_modality_example, input_modality_example, length in zip(predicted_modality, input_modality, lengths):
                loss += loss_functions[m](predicted_modality_example[length-1], input_modality_example[length-1])
                i+= 1

        res = loss / i
        return res

    def img_loss(self, preds, labels):
        res = nn.MSELoss()(preds, labels.permute(2,0,1))
        return self.modalities_lambda * res

    @staticmethod
    def correlation_func(hidden, v_hidden, index_tuple):
        EPS = 1e-7
        i,j = index_tuple
        res = torch.sum(torch.mul(v_hidden[i] ,v_hidden[j]), dim = 1)
        res /= (torch.norm(hidden[i], dim = 1) * torch.norm(hidden[j], dim = 1) + EPS)
        return res.sum()


class DecoupleDynamicsModelLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._lambda = kwargs
        self.use_X = kwargs['use_X']
        self.mse = nn.MSELoss()

    def reconstruction_loss(self, y, pred):
        pred = pred.detach()
        return self.mse(y, pred)

    def state_loss(self, y, pred):
        pred = pred.detach()
        return self.mse(y, pred)

    def forward_model_loss(self, y, pred):
        pred = pred.detach()
        return self.mse(y, pred)

    def inverse_loss(self, y, pred):
        pred = pred.detach()
        return self.mse(y, pred)

    def forward(self, model_output, input_modalities, lengths):
        size = len(model_output)
        res = 0

        for model_dict in model_output:
            # action only between X values
            if self.use_X:
                model_dict['action_t0_pred'] = (model_dict["state_t1"][0] - model_dict["state_t0"][0])[:,tuple(ACTION_INDS)]
                inverse_loss_val = self.inverse_loss(model_dict["action_t0"], model_dict["action_t0_pred"])
            else:
                inverse_loss_val = 0
            forward_model_loss_val = self.forward_model_loss(model_dict["embed_t1"], model_dict["embed_t1_pred"])

            res += self._lambda['lambda_forward'] * forward_model_loss_val + \
                   self._lambda['lambda_inverse'] * inverse_loss_val

            for m in range(len(model_dict['state_t0'])):
                reconstruction_loss_val = self.reconstruction_loss(model_dict["state_t0"][m], model_dict["state_t0_pred"][m])
                state_loss_val = self.reconstruction_loss(model_dict["state_t1"][m], model_dict["state_t1_pred"][m])
                res += self._lambda['lambda_decoder'] * (reconstruction_loss_val + state_loss_val)

        return res / size


class ContextReconstructionMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, predicted_modalities, input_modalities, lengths):
        context_size = predicted_modalities[0].shape[1] // 2
        loss = 0
        count = 0
        for predicted_modality, input_modality in zip(predicted_modalities, input_modalities):
            for predicted_modality_example, input_modality_example, length in zip(predicted_modality, input_modality,
                                                                                  lengths):
                loss += F.mse_loss(predicted_modality_example[:context_size],
                                   input_modality_example[:context_size])
                loss += F.mse_loss(predicted_modality_example[context_size:],
                                   input_modality_example[length - context_size:length])
                count += context_size * 2

        return loss / count

