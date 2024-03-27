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
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SEED_VAL = 42 #np.random.randint(0, 60)
MODEL = "meta-llama/Llama-2-70b-chat-hf"

SYSTEM_PROMPT = '''<s>[INST] <<SYS>>
You are a helpful physician assistant. You come up with example sentences from doctor's notes and with the same label.
<</SYS>>'''

NLIprompt = """\nThe following are doctor notes and a corresponding label indicating relationship between the Assessment and Plan from doctor's note.
Here are three examples of Assessment and Plan from doctor's note and the relationship between them:
{summaries}
Write 1 new note like this with the same label.[/INST]"""
llama_params = '70B'

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

def get_egsents(df):
    # sample three random rows
    sample_df = df.sample(n=3)
    # print(sample_df)
    # Concatenate the values as strings
    result_strings = []
    for idx, row in sample_df.iterrows():
        result = f"example {idx+1}. Assessment: {row['Assessment']}\n\nPlan Subsection: {row['Plan Subsection']}\n\nRelationship is: {row['Relation']}\n"
        result_strings.append(result)

    # Combine the results into one string
    final_result = "".join(result_strings)

    # print(final_result)
    return final_result


if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    egsents = []

    direct = train[train['Relation'] == 'Direct']#.sample(400)
    indirect = train[train['Relation'] == 'Indirect']#.sample(400)
    not_related = train[train['Relation'] == 'Not Relevant']#.sample(400)
    neither = train[train['Relation'] == 'Neither']#.sample(400)

    # populate 3200 examplars, 800*4
    for relation in [direct, indirect, not_related, neither]:
        for i in range(1200):
            egsents.append(get_egsents(relation))
    # for i in range(0,3200):
    #     if i < 800:
    #         egsents.append(get_egsents(direct))
    #     elif i < 1600:
    #         egsents.append(get_egsents(indirect))
    #     elif i < 2400:
    #         egsents.append(get_egsents(not_related))
    #     else:
    #         egsents.append(get_egsents(neither))
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # Define the directory and file path
    directory = './raw_synthetic/'
    file_name = 'ap_synthetic_fullTrain' + llama_params + '.json'
    file_path = os.path.join(directory, file_name)

    # Check and create directory if not exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # If the file exists, load the data, otherwise initialize an empty list
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            synth_data_unformatted = json.load(f)
    else:
        synth_data_unformatted = []

    buffer = []

    for j, i in enumerate(tqdm.tqdm(egsents)):
        prompt = SYSTEM_PROMPT + NLIprompt.format(summaries=i)
        buffer.append(generate_sequence(prompt)[0]['generated_text'])
        
        # Save when buffer reaches 500 items or on the last iteration
        if (j + 1) % 500 == 0 or j == len(egsents) - 1:
            synth_data_unformatted.extend(buffer)
            with open(file_path, 'w') as f:
                json.dump(synth_data_unformatted, f)
            buffer.clear()
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # for j, i in enumerate(tqdm.tqdm(egsents[:])):
    #     synth_data_unformatted = []
    #     prompt = SYSTEM_PROMPT + NLIprompt.format(summaries=i)
    #     if os.path.exists(file_path):
    #         with open(file_path, 'r') as f:
    #             synth_data_unformatted = json.load(f)
    #     synth_data_unformatted.append(generate_sequence(prompt)[0]['generated_text'])


    #     # Write to the file.
    #     with open(file_path, 'w') as f:
    #         json.dump(synth_data_unformatted, f)
