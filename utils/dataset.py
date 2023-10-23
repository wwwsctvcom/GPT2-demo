import torch
import platform
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class ChatDataset(Dataset):

    def __init__(self, data_path, tokenizer, data_type: str = "train", max_seq_len: int = 150):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.lines = []
        with open(data_path, 'rb') as reader:
            data = reader.read().decode("utf-8")
            if platform.system().lower() == 'windows':
                train_data = data.split("\r\n\r\n")
            elif platform.system().lower() == 'linux':
                train_data = data.split("\n\n")
            # choose data_num samples
            if self.data_type == "train":
                self.train_data = train_data[:1000]
            elif self.data_type == "test":
                self.train_data = train_data[-500:]
        self.max_seq_len = max_seq_len

        # create token
        sep_id = self.tokenizer.sep_token_id
        cls_id = self.tokenizer.cls_token_id

        for index, dialogue in enumerate(tqdm(self.train_data)):
            utterances = []
            if platform.system().lower() == 'windows':
                utterances = dialogue.split("\r\n")
            elif platform.system().lower() == 'linux':
                utterances = dialogue.split("\n")

            input_ids = [cls_id]  # dialogue begin with [CLS]
            for utterance in utterances:
                input_ids += self.tokenizer.encode(utterance, add_special_tokens=False)
                input_ids.append(sep_id)  # add [SEP] after per utterance
            self.lines.append(input_ids)

    def __getitem__(self, index):
        input_ids = self.lines[index]
        input_ids = input_ids[:self.max_seq_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.lines)


def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels
