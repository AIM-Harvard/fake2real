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
import random

SEED_VAL = 42 #np.random.randint(0, 60)
MODEL = "meta-llama/Llama-2-70b-chat-hf"
random.seed(SEED_VAL)

SYSTEM_PROMPT = '''<s>[INST] <<SYS>>
You are a helpful physician assistant. You come up with examples from doctor's notes and health records.
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
        max_new_tokens=500,
        temperature=.9, 
        repetition_penalty=1.17,
    )

if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    egsents = []
    train['concat'] = train['Assessment'] + '\n'+ train['S'] + '\n' + train['O']
    train = train[~(pd.isna(train['Summary']))]
    train = train[~(pd.isna(train['concat']))]
    train['concat'] = train['concat'].astype(str)
    train.reset_index(inplace=True, drop=True)
    for i in range(len(train)):
        egsents.append((train.loc[i, 'concat'], train.loc[i, 'Summary']))
    
    egsents = [eg for eg in egsents if len(eg[0].split())>=300 and len(eg[0].split())<=500] # limit token length
    egsents = random.sample(egsents, 300) # take 300

    NLIprompt = """\nThe following are doctor notes and a corresponding summary.
Here are two examples of note / summary pairs:
{summaries}
Write 1 new note and its summary.[/INST]"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    llama_params = '70B'
    for i in tqdm.tqdm(range(600)):
        note_summary = random.sample(egsents, 2)
        sents = [str(j+1)+'. Note:\n'+ns[0]+'\n\n'+'Summary:\n'+ns[1] for j, ns in enumerate(note_summary)]
        prompt = SYSTEM_PROMPT + NLIprompt.format(summaries='\n\n'.join(sents))
        if i == 0: 
            print(prompt)
            print('='*100)
            print(len(prompt.split()))
        synth_data_unformatted = []
        if os.path.exists('./raw_synthetic/SUMM_synthetic'+llama_params+'.json'):
            with open('./raw_synthetic/SUMM_synthetic'+llama_params+'.json', 'r') as f:
                synth_data_unformatted = json.load(f)
        synth_data_unformatted.append(generate_sequence(prompt)[0]['generated_text'])

        with open('./raw_synthetic/SUMM_synthetic'+llama_params+'.json', 'w') as f:
            json.dump(synth_data_unformatted, f)
