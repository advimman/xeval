import os
import logging
import numpy as np
import torch

from ..core.utils import move_data_to_device

LOGGER = logging.getLogger()


def train_ae_feature_extractor(model, train_loader, val_loader, optimizer,
                               loss, epoch_count, model_save_path, use_cuda):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    best_val_loss = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5, cooldown=1)
    for epoch_id in range(epoch_count):
        for step, (inputs, lengths) in enumerate(train_loader):
            inputs = move_data_to_device(inputs, use_cuda)
            lengths = move_data_to_device(lengths, use_cuda)
            pred = model(inputs, lengths)
            loss_value = loss(pred, inputs, lengths)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_value = loss_value.cpu().data.numpy()

            LOGGER.info("[Training] Epoch %d, step %d, loss %f", epoch_id, step, loss_value)

        val_loss = 0
        val_examples_count = 0
        with torch.no_grad():
            for step, (inputs, lengths) in enumerate(val_loader):
                inputs = move_data_to_device(inputs, use_cuda)
                lengths = move_data_to_device(lengths, use_cuda)
                pred = model(inputs, lengths)
                loss_value = loss(pred, inputs, lengths).cpu().data.numpy()
                val_loss += loss_value * len(lengths)
                val_examples_count += len(lengths)
                LOGGER.info("[Validation] Epoch %d, step %d, loss %f", epoch_id, step, loss_value)

        val_loss /= val_examples_count
        scheduler.step(val_loss, epoch_id)
        LOGGER.info("[Validation] Epoch %d, mean loss %f. Previous best loss is %f", epoch_id, val_loss, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))
