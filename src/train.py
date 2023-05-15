import logging
import os
from pathlib import PosixPath
from typing import Any, Tuple

import hydra
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
# TODO: Save model graph with FileWriter
# TODO: Add train params to tensorboard
# TODO: from tqdm import tqdm - for train batches
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Blip2ForConditionalGeneration

log = logging.getLogger(__name__)

# region Classes
__modes__ = ('train', 'test')


class ImagePromptDatatset(Dataset):
    def __init__(self, dataset, processor, mode='train'):
        self.dataset = dataset
        n_records = len(dataset)
        split = int(n_records * .1)  # 10%

        assert mode in __modes__
        self.mode = mode

        if self.mode == 'train':
            self.ids = np.arange(split, n_records)
            # self.ids = np.arange(split, split*2, dtype=np.int64)

        elif self.mode == 'test':
            self.ids = np.arange(0, split, dtype=np.int64)

        self.processor = processor
        self.max_length = 128

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = self.dataset[int(self.ids[idx])]
        encoding = self.processor(images=item['image'], text=item['prompt'], padding='max_length',
                                  max_length=self.max_length, truncation=True, return_tensors='pt')
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        # input_ids = encoding['input_ids']
        # input_ids[input_ids == 1] = -100
        # encoding['input_ids'] = input_ids
        return encoding
# endregion


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_blip2(dir: str, dtype: str, offload_state_dict: bool) -> Tuple[AutoProcessor, Blip2ForConditionalGeneration]:
    dtypes = {'bfloat16': torch.bfloat16,
              'float16': torch.float16, 'float32': torch.float32}
    assert dtype in dtypes.keys()

    processor = AutoProcessor.from_pretrained(dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
        dir,
        offload_state_dict=offload_state_dict,
        torch_dtype=dtypes[dtype])

    return processor, model


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device) -> Tuple[float, float]:
    epoch_total_loss = 0

    # Train
    for idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device, torch.bfloat16)
        # pixel_values.requires_grad = True

        outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        epoch_total_loss += loss.item()
        loss.backward()

        optimizer.step()

    # Validation
    with torch.no_grad():
        val_loss = 0.0
        for idx, batch in enumerate(test_dataloader):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device, torch.bfloat16)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values, labels=input_ids)
            val_loss += outputs.loss.item()

    return epoch_total_loss, val_loss


@hydra.main('../conf', 'config.yaml', version_base='1.2')
def main(cfg: Any):
    # Env setup
    log.info('Configuring environment')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.backends.cudnn.allow_tf32 = cfg.use_tf_core
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf_core
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = cfg.use_tf_core
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = cfg.use_tf_core
    np.random.seed(cfg.seed)
    torch.seed = cfg.seed
    original_directory = PosixPath(hydra.utils.get_original_cwd())
    cfg.model.dir = (original_directory / cfg.model.dir).absolute()

    # Model load
    log.info('Loading model')
    processor, model = load_blip2(**cfg.model)

    # Data load
    log.info('Data setup')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(cfg.seed)

    dataset = load_dataset(cfg.data.dataset.name, cfg.data.dataset.part)
    train_dataset = ImagePromptDatatset(
        dataset=dataset['train'], processor=processor, mode='train')
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, generator=generator, batch_size=cfg.data.dataloader.batch_size, num_workers=cfg.data.dataloader.num_workers, pin_memory=True)
    test_dataset = ImagePromptDatatset(
        dataset=dataset['train'], processor=processor, mode='test')
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, generator=generator, batch_size=cfg.data.dataloader.batch_size, num_workers=cfg.data.dataloader.num_workers, pin_memory=True)

    # Train
    log.info('Training configuration')
    print_trainable_parameters(model)
    # Freeze all layers - we simply don't have the gpu memory (or power!!) to train all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last n layers of vision model
    # n_last_trainable = 1
    # n_layers = len(model.vision_model.encoder.layers)
    # for idx in range(n_layers - n_last_trainable, n_layers):
    #     for param in model.vision_model.encoder.layers.parameters():
    #         param.requires_grad = True

    # Unfreeze  q former
    n_last_trainable = len(model.qformer.encoder.layer)
    #  annoyingly uses 'layer' instead of 'layers' like the vision model
    n_layers = len(model.qformer.encoder.layer)
    for idx in range(n_layers - n_last_trainable, n_layers):
        for param in model.qformer.encoder.layer[idx].parameters():
            param.requires_grad = True

    print_trainable_parameters(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    # optimizer = bnb.optim.adam.Adam8bit(model.parameters(), lr=5e-5)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=5e-3)
    writer = SummaryWriter()
    checkpoint_path = PosixPath('./checkpoints')

    log.info('Starting train loop')
    for epoch in range(cfg.train.epochs):
        train_loss, val_loss = train_epoch(
            model, optimizer, train_dataloader, test_dataloader, device)
        if cfg.train.do_checkpoint == True:
            checkpoint = checkpoint_path / f'epoch_{epoch:02d}'
            model.save_pretrained(checkpoint)

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Test Loss', val_loss, epoch)
        log.info(
            f'Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Validation Score: N/A')
    log.info('Saving final model')
    model.save_pretrained('blip2', max_shard_size='1GB')


if __name__ == '__main__':
    main()
