import sys
import openai
import tiktoken
import torch
import json
import random
import nltk
from time import sleep
from copy import deepcopy

from string import ascii_letters
import traceback
from textwrap import dedent

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMPipeline:
    def __init__(self, model_type, model_name_or_path) -> None:
        hf_pipe = None
        if model_type == 'openai':
            model = openai.ChatCompletion
            tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                torch_dtype="auto", 
                trust_remote_code=True, 
            )
            assert torch.cuda.is_available(), "This model needs a GPU to run ..."
            device = torch.cuda.current_device()
            model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            hf_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device
            )

        self.model = model
        self.tokenizer = tokenizer
        self.hf_pipe = hf_pipe
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
    
    def patch_kwargs_with_default(self, kwargs):
        kwargs_copy = deepcopy(kwargs)
        if self.model_type == 'openai':
            default_args = {
                "max_tokens": 500,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "n": 1,
                "stream": False,
            }
        else:
            default_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
        for k in default_args:
            if k not in kwargs_copy:
                kwargs_copy[k] = default_args[k]
        return kwargs_copy

    def query(self, prompt, **kwargs):
        kwargs = self.patch_kwargs_with_default(kwargs)

        if self.model_type == 'openai':
            kwargs["messages"] = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            answer = None
            while answer is None:
                try:
                    response = self.model.create(model=self.model_name_or_path, **kwargs)
                    answer = response["choices"][0]["message"]["content"]
                except ConnectionError as e:
                    answer = None
                    print(f"error: {e}, response: {response}")
                    sleep(60)
                except Exception as ex:
                    print('Unknown Error:', ex)
                    print(traceback.format_exc())
                    sleep(60)
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]

            output = self.hf_pipe(messages, **kwargs)
            answer = output[0]['generated_text']
        return answer


def make_phi3_qa_prompt(sents, question):
    prompt_template_qa = dedent('''
    Having the question and context. Give one concrete answer on this question.
    ## Question: {question}
    ## Context: {context}
    ## Answer:
    ''').strip()
    prompt_template_phi3 = '<|user|>\n{}<|end|>\n<|assistant|>\n'
    
    ctx = ' '.join(sents)
    p = prompt_template_qa.format(
        context=ctx,
        question=question
    )
    p = prompt_template_phi3.format(p)
    
    return p


def sentence_is_good(sentence):
    ascii_letters_s = set(ascii_letters)
    words = nltk.word_tokenize(sentence)
    if not words:
        return False
    def word_is_ascii(w):
        return len(set(w) - ascii_letters_s) == 0
    n_ascii_words = sum([int(word_is_ascii(w)) for w in words])
    return n_ascii_words / len(words) >= 0.75
