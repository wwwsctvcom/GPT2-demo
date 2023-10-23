import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from utils.tools import get_cur_time


class Trainer:

    def __init__(self, args, model, tokenizer, optimizer, scheduler, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model.to(self.device)

    def train(self,
              train_data_loader: DataLoader = None,
              test_data_loader: DataLoader = None):
        best_val_loss = float("inf")
        for epoch in range(self.args.epochs):
            epoch_ave_train_loss = self.train_epoch(train_data_loader)

            if test_data_loader is None:
                raise ValueError("test date loader is not set !")
            epoch_ave_test_loss = self.train_epoch(test_data_loader)
            logger.info(f"epoch average train loss: {epoch_ave_train_loss}, "
                        f"epoch average test loss: {epoch_ave_test_loss}")

            if epoch_ave_test_loss < best_val_loss:
                best_val_loss = epoch_ave_test_loss
                model_path = get_cur_time() + "best"
                self.save_model(model_path)
                logger.info(f"saving best model to {model_path}")

    def save_model(self, model_path: str = None):
        if not Path(model_path).exists():
            Path(model_path).mkdir()
        self.model = self.model.module if hasattr(self.model, 'module') else self.model
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def train_epoch(self, train_data_loader=None):
        total_loss = 0
        self.model.train()
        epoch_correct_num, epoch_total_num = 0, 0

        with tqdm(enumerate(train_data_loader), total=len(train_data_loader)) as pbar:
            for step, (input_ids, labels) in pbar:
                try:
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model.forward(input_ids, labels=labels)
                    logits = outputs.logits
                    loss = outputs.loss
                    loss = loss.mean()

                    # calculate acc
                    batch_correct_num, batch_total_num = self.calculate_acc(logits, labels,
                                                                            ignore_index=self.args.ignore_index)
                    epoch_correct_num += batch_correct_num
                    epoch_total_num += batch_total_num
                    batch_acc = batch_correct_num / batch_total_num

                    total_loss += loss.item()
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    # gradient clip
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # update parameters
                        self.optimizer.step()
                        # update lr
                        self.scheduler.step()
                        # clear gradient
                        self.optimizer.zero_grad()

                    pbar.set_description(
                        desc="loss: %.3f, acc: %.3f, lr: %.3f" % (loss.item() * self.args.gradient_accumulation_steps,
                                                                  batch_acc, self.scheduler.get_lr()[0]))
                    pbar.update(1)

                    del input_ids, outputs

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        logger.info("WARNING: ran out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logger.info(str(exception))
                        raise exception
        epoch_mean_loss = total_loss / len(train_data_loader)
        return epoch_mean_loss

    def test_epoch(self, test_data_loader):
        total_loss = 0
        self.model.eval()
        try:
            with torch.no_grad():
                for batch_idx, (input_ids, labels) in enumerate(test_data_loader):
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model.forward(input_ids, labels=labels)
                    loss = outputs.loss
                    loss = loss.mean()

                    total_loss += loss.item()
                    del input_ids, outputs
            epoch_mean_loss = total_loss / len(test_data_loader)
            return epoch_mean_loss
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    @staticmethod
    def calculate_acc(logit, labels, ignore_index=-100):
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = labels[..., 1:].contiguous().view(-1)

        _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index, 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
        non_pad_mask = labels.ne(ignore_index)
        n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return n_correct, n_word
