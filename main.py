import json 
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm

from prompt import COT_PROMPT_5


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []

    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sent = sent.split('</s>')[0]
        sent = sent.split('\n\n\n')[0]
        sent = sent.split("\n\n")[0]
        sents.append(sent)
    return sents

def generate_result(model, tokenizer, input_txt):
    print(f"Input text: {input_txt}\n")
    input_txt = COT_PROMPT_5 + f"{input_txt}\n\n### Response:"
    input_ids = tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor(
                [input_ids]).to(model.device)
    outputs = model.generate(context_enc,
                            num_return_sequences=1,
                            max_length=512, 
                            use_cache=True,
                            temperature=0.1,
                            do_sample=True,
                            eos_token_id=[tokenizer.eos_token_id]
                    )

    output_text = decode(outputs,tokenizer,raw_text_len)[0]
    print(f"\nOutput text: {output_text}\n")
    return output_text


if __name__ == "__main__":

    # load datasets
    dataset = load_dataset("webis/tldr-17", split="train")
    print(f"len(dataset): {len(dataset)}")

    dataset = dataset.select(range(10000))
    print(f"len(dataset): {len(dataset)}")
    # print(dataset[0])

    # load model
    model_name = "/mnt/bd/fazzie-models/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()


    progress_bar = tqdm(range(len(dataset)))

    for i, sample in enumerate(dataset):

        input = sample['summary']

        outputs = generate_result(model, tokenizer, input)

        results = {"results" : outputs, "summary": sample['summary'], "subreddit": sample['subreddit']}

        with open("result.jsonl", "a+") as fout:
            fout.write(json.dumps(results)+'\n')
        progress_bar.update(1)

