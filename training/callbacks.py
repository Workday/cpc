import os
import sys

from sentence_splitter import split_text_into_sentences
from munch import Munch
from tqdm import tqdm
import json

from model_llama import LlamaBiForMNTPandSentEmbeddingsV2_w_q_token
from model_llama import mean_pooling
from args import parse_args
from transformers import AutoConfig, AutoTokenizer
import torch
from peft import PeftModel
from datasets import load_dataset
from argparse import ArgumentParser
from loss import cos_sim
import numpy as np
import colorprint3
import tiktoken
from transformers import TrainerCallback
from util.preprocessing import compress_sample


class EvaluatorCallback(TrainerCallback):
    def __init__(self, source_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_path = source_path

    def on_evaluate(self, args, state, control, model, tokenizer, logs=None, **kwargs):
        tiktoken_tokenizer = tiktoken.encoding_for_model("gpt-4")
        compressed_target_tokens = 2100
        if state.is_local_process_zero:
            inference_dir = os.path.join(args.output_dir, "inference")
            if not os.path.exists(inference_dir):
                os.makedirs(inference_dir)

            with open(self.source_path) as fin:
                samples = json.load(fin)
            model.eval()
            for s in tqdm(samples):
                if len(tiktoken_tokenizer.encode(s['context'])) < compressed_target_tokens:
                    s['compressed_context'] = s['context']
                    continue
                sents = compress_sample(
                    model,
                    tokenizer,
                    tiktoken_tokenizer,
                    s,
                    compressed_target_tokens
                )

                s['compressed_context'] = ' '.join(sents)
            for sample in samples:
                del sample['encodings']
            with open(f'{inference_dir}/{state.global_step}.json', 'w') as fout:
                json.dump(samples, fout)
            model.train()
