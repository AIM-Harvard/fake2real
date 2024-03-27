import pandas as pd
import numpy as np
import torch
import transformers
from tqdm import tqdm
import os
import csv
import tqdm
import json
import sys


SEED_VAL = np.random.randint(0, 60)
MODEL = "meta-llama/Llama-2-70b-chat-hf"

SYSTEM_PROMPT = '''<s>[INST] <<SYS>>
You are a helpful physician assistant. You come up with example sentences from doctor's notes and health records.
<</SYS>>'''
def generate_sequence(prompt):
    return pipeline(
        prompt,
        do_sample=True,
        top_k=50,
        top_p = 0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # max_length=200,
        max_new_tokens=2000,
        temperature=.9, 
        repetition_penalty=1.17,
    )

if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    egsents = []

    for i in range(len(train)):
        egsents.append(' | '.join([train.loc[i, 'Premise'], train.loc[i, 'Hypothesis'], train.loc[i, 'Relation']]))
    NLIprompt = '\nThe following are sentence pairs (Premise and Hypothesis) from doctor notes. Each sentence pair has a relation (entailment, contradiction, neutral). Here is what they should look like:\nPremise | Hypothesis | Relation\n{sents}\nWrite 10 new pairs of sentences for each relation type.[/INST]'
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    llama_params = '70B_20Percent'
    for i in tqdm.tqdm(range(0, 2247, 3)): #int(len(egsents)/2)
        sents = egsents[i:i+3]
        prompt = SYSTEM_PROMPT + NLIprompt.format(sents='\n'.join(sents))
        synth_data_unformatted = []
        if os.path.exists('medNLI_synthetic'+llama_params+'.json'):
            with open('medNLI_synthetic'+llama_params+'.json', 'r') as f:
                synth_data_unformatted = json.load(f)
        synth_data_unformatted.append(generate_sequence(prompt)[0]['generated_text'])

        with open('medNLI_synthetic'+llama_params+'.json', 'w') as f:
            json.dump(synth_data_unformatted, f)
