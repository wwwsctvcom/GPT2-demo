import argparse
import torch
import transformers
from loguru import logger
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from utils.dataset import ChatDataset, collate_fn
from utils.tools import seed_everything, get_cur_time
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from torch.nn import DataParallel


def set_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name_or_path', default="", type=str, required=False, help='')
    parser.add_argument('--config', default='config/config.json', type=str, required=False, help='')

    # dataset
    parser.add_argument('--data_path', default='data/train.txt', type=str, required=False, help='')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False,
                        help='')
    parser.add_argument('--max_seq_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='for ignore_index label token do not calculate gradient')
    parser.add_argument('--num_workers', type=int, default=0, help="data loader worker number")

    # train
    parser.add_argument('--device', default="", type=str, required=False, help='')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', default=2, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='')
    parser.add_argument('--lr', default=2.6e-5, type=float, required=False, help='leaning rate')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='weight decay')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--patience', type=int, default=0, help="for early stopping, if set 0, will not early stopping")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up step')

    # model saved
    parser.add_argument('--save_model_path', default='model', type=str, required=False,
                        help='model to save !')
    args = parser.parse_args()
    return args


def main():
    args = set_args()

    seed_everything(42)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using device:{}'.format(args.device))

    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id
    logger.info(f"loading tokenizer from vocab file {args.vocab_path} done !")

    config = GPT2Config.from_json_file(args.config)
    model = GPT2LMHeadModel(config=config)
    logger.info("loading model from config {args.config} !")

    assert model.config.vocab_size == tokenizer.vocab_size

    if torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
        logger.info("DataParallel training !")

    # calculate parameters number
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('Number of model parameters: {} !'.format(num_parameters))

    # Trainer
    logger.info("Loading data set...")
    train_dataset = ChatDataset(data_path=args.data_path,
                                tokenizer=tokenizer,
                                data_type="train",
                                max_seq_len=args.max_seq_len)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    test_dataset = ChatDataset(data_path=args.data_path,
                               tokenizer=tokenizer,
                               data_type="test",
                               max_seq_len=args.max_seq_len)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    # optimizer and scheduler, t_total: every gradient_accumulation_steps is a step * epochs
    t_total = len(train_data_loader) // args.gradient_accumulation_steps * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("Start Training...")
    trainer = Trainer(args=args,
                      model=model,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      device=args.device)
    trainer.train(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

    # saving path
    trainer.save_model("./model/" + get_cur_time() + "/best")


if __name__ == '__main__':
    main()
