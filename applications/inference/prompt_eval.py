import argparse
import logging
import torch
import sys
import os

from transformers import (
    AutoModelForCausalLM, )

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_pretrain",
        type=str,
        help="Path to pretrain model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_sft",
        type=str,
        help="Path to SFT model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_ppo",
        type=str,
        help="Path to PPO model",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model, tokenizer, device, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("===========================================================")
        print("================== prompt start ===========================")
        print("========== Pre-train model ==========")
        r_pretrain = generate(model[0],
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_pretrain)
    
        # Here we use the simplest greedy search. You can also use other methods, 
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # Examples details can be found at https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py#L22
        print("============= SFT model =============")
        r_sft = generate(model[1],
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_sft)

        if args.model_name_or_path_ppo:
            print("============= PPO model =============")
            r_ppo = generate(model[2],
                                    tokenizer,
                                    inputs,
                                    num_beams=1,
                                    num_return_sequences=args.num_return_sequences,
                                    max_new_tokens=args.max_new_tokens)
            print_utils(r_ppo)
        print("====================prompt end=============================")
        print("===========================================================\n")



def main():
    args = parse_args()

    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_pretrain,
                                  fast_tokenizer=True)

    model = []

    model_pretrain = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_pretrain,
                                     tokenizer, None)
    model_pretrain.to(device)
    model.append(model_pretrain)

    model_sft = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_sft,
                                     tokenizer, None)
    model_sft.to(device)
    model.append(model_sft)

    if args.model_name_or_path_ppo:
        model_ppo = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_ppo,
                                     tokenizer, None)
        model_ppo.to(device)
        model.append(model_ppo)


    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            "Human: Please tell me about Microsoft in a few sentence? Assistant:",
            "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
            "Human: Write a short poem about a wise frog. Assistant:",
            "Human: Who was president of the United States in 1955? Assistant:",
            "Human: How does a telescope work? Assistant:",
            "Human: Why do birds migrate south for the winter? Assistant:"
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]

    prompt_eval(args, model, tokenizer, device, prompts)


if __name__ == "__main__":
    main()
