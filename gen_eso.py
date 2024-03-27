import pandas as pd
from azure.identity import DefaultAzureCredential
import openai
import tqdm
import random
import os
import json

random.seed(42)
SYS_MES = "You are an oncologist who helps write notes for datasets."
ESOPH_PROMPT = """The following are examples of parts of doctor notes. The grade that follows describes the CTCAE score of esophagitis based on the text.
{docs}
Write {num} new partial notes and their CTCAE grades for each grade. These will be used to make a new dataset, so they should be completely unique but mimic the style of the examples."""

ESOPH_0_PROMPT = """The following are examples of parts of doctor notes. They describe patients with no presence of esophagitis, so the CTCAE score would be 0.
{docs}
Write {num} new partial notes that do not have esophagitis either implicitly or explicitly. These will be used to make a new dataset, so they should be completely unique but mimic the style of the examples."""

# # # # Request credential
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default",)

# Setup parameters
openai.api_type = "azure_ad"
openai.api_key = token.token #"88333da25
openai.api_base = "https://bwh-openai-service.openai.azure.com/"
openai.api_version = "2023-05-15"

def make_prompt(system_message, user_message):
    return [{"role": "system", "content": system_message},
            {"role": "user", "content": user_message}]

def gpt_response(messages, model = "gpt-3.5-turbo", temperature=0.1, max_tokens=1000, top_p=0.95, frequency_penalty=0, presence_penalty=0):
    response = openai.ChatCompletion.create(
        model=model,
        engine="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        # stop=None
        )['choices'][0]['message']['content']
    return response


def format_example(df):
    result_strings = []
    for _, row in df.iterrows():
        result_strings.append(f"Note: {row['gpt_section']}\nCTCAE grade: {row['grade']}")
    return result_strings


if __name__ == '__main__':
    train = pd.read_csv('./data/train.csv')
    train['grade'] = train['grade'].astype(int)
    g0 = train[train['grade']==0][:5]
    g1 = train[train['grade']==1][:5]
    g2 = train[train['grade']==2][:5]
    g3 = train[train['grade']==3][:5]
    g0.reset_index(inplace=True, drop=True)
    g1.reset_index(inplace=True, drop=True)
    g2.reset_index(inplace=True, drop=True)
    g3.reset_index(inplace=True, drop=True)

    g0_eg = format_example(g0)
    g1_eg = format_example(g1)
    g2_eg = format_example(g2)
    g3_eg = format_example(g3)

    random.shuffle(g0_eg)
    random.shuffle(g1_eg)
    random.shuffle(g2_eg)
    random.shuffle(g3_eg)
    directory = './raw_synthetic/'
    file_name = 'esoph_synthetic_20_ask5_gpt4.json'
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

    for j, (eg0, eg1, eg2, eg3) in enumerate(tqdm.tqdm(zip(g0_eg, g1_eg, g2_eg, g3_eg), total=len(g1_eg))):
        exemplars = '\n\n'.join([eg0, eg1, eg2, eg3])
        prompt_str = ESOPH_PROMPT.format(docs=exemplars, num=str(5))
        # print(prompt_str)
        # assert(1==2)
        prompt = make_prompt(SYS_MES, prompt_str)
        buffer.append(gpt_response(prompt))
        
        # Save when buffer reaches 10 items or on the last iteration
        if (j + 1) % 10 == 0 or j == len(g0_eg) - 1:
            synth_data_unformatted.extend(buffer)
            with open(file_path, 'w') as f:
                json.dump(synth_data_unformatted, f)
            buffer.clear()

#     train = pd.read_csv('./data/train.csv')
#     train['grade'] = train['grade'].astype(int)
#     g0 = train[train['grade']==0][:50]

#     g0.reset_index(inplace=True, drop=True)


#     g0_eg = format_example(g0)


#     random.shuffle(g0_eg)

#     directory = './raw_synthetic/'
#     file_name = 'esoph_synthetic_50x100_give3ask5s_gpt35.json'
#     file_path = os.path.join(directory, file_name)

#     # Check and create directory if not exists
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # If the file exists, load the data, otherwise initialize an empty list
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as f:
#             synth_data_unformatted = json.load(f)
#     else:
#         synth_data_unformatted = []

#     buffer = []


# for j in tqdm.tqdm(range(100)):
#     # Randomly sample 3 elements from g0_eg
#     sampled_egs = random.sample(g0_eg, 3)
#     exemplars = '\n\n'.join(sampled_egs)

#     prompt_str = ESOPH_0_PROMPT.format(docs=exemplars, num=str(5))
#     prompt = make_prompt(SYS_MES, prompt_str)
#     buffer.append(gpt_response(prompt))
    
#     # Save when buffer reaches 10 items or on the last iteration
#     if j == 99 or len(buffer) >= 10:  # Check for the last iteration or buffer size
#         synth_data_unformatted.extend(buffer)
#         with open(file_path, 'w') as f:
#             json.dump(synth_data_unformatted, f)
#         buffer.clear()
