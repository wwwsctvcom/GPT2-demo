import torch
import argparse
from utils.tools import top_k_top_p_filtering
from loguru import logger
from transformers import GPT2LMHeadModel
from transformers import BertTokenizerFast
import torch.nn.functional as F


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='model/epoch40', type=str, required=False, help='')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='')
    parser.add_argument('--device', default='0', type=str, required=False, help='')

    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help="penalty for repeat word")
    parser.add_argument('--max_len', type=int, default=25, help='max length for per utterance')
    parser.add_argument('--max_history_len', type=int, default=3, help="max length for dialogue history")
    return parser.parse_args()


def main():
    args = set_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device:{}'.format(args.device))
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path,
                                  sep_token="[SEP]",
                                  pad_token="[PAD]",
                                  cls_token="[CLS]")

    # loading pretrained model
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model = model.to(args.device)
    model.eval()


    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('Start chatting with ChatBot...')

    while True:
        try:
            text = input("user:")
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            input_ids = [tokenizer.cls_token_id]  # begin with [CLS] for per input

            # choose newest max_history_len utterance
            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            input_ids = torch.tensor(input_ids).long().to(args.device)
            input_ids = input_ids.unsqueeze(0)
            response = []
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                # add repetition_penalty for had generated token,
                # to reduce the probability of generating repeat word.
                # divide is to reduce the probability, if the probability bigger the probability is smaller
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # set [UNK] infinite small
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial: 表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # if generate [SEP], represent the ending of the sequence
                    break
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot:" + "".join(text))
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
