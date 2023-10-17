import json 
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from datasets import load_dataset

from prompt import COT_PROMPT_5



def generate_result(model, tokenizer, inputs):


    inputs = tokenizer(inputs, 
                        padding="longest",
                        return_tensors="pt").cuda()

    with torch.no_grad():
        outputs = model.generate(
                        inputs, 
                        num_return_sequences=1,
                        max_length=2048, 
                        use_cache=True,
                        temperature=0.1,
                        do_sample=True,
                        eos_token_id=[tokenizer.eos_token_id],
        )

    outputs = tokenizer.decode(outputs, skip_special_tokens=False)


    print(outputs[0])

if __name__ == "__main__":

    # load datasets
    dataset = load_dataset("webis/tldr-17", split="train")

    dataset = dataset.take(1000)

    print(f"len(dataset): {len(dataset)}")
    print(dataset[0])

    dataset = dataset.map(lambda x: {"summary": x["summary"], "subreddit": x["subreddit"], "inputs": COT_PROMPT_5.format(x["summary"])}, batched=True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

    # load model
    model_name = "/mnt/bd/fazzie-models/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    for i, batch in enumerate(dataloader):

        if i > 5 :
            break

        inputs = batch['inputs']

        outputs = generate_result(model, tokenizer, inputs)

        results = {"results" : outputs}

        with open("result.jsonl", "a+") as fout:
            fout.write(json.dumps(results)+'\n')

