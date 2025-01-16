import os
import json

import tiktoken

from model.common import ModelType, build_model
from model.llama import LlamaBiForMNTPandSentEmbeddingsV2
from model.mistral import MistralBiForMNTPandSentEmbeddingsV2

from model.model import load_model_and_tokenizer

from util.util import SentenceEmbeddingType
from util.preprocessing import compress_sample, SamplePreprocessor


class PromptCompressorCPC:
    def __init__(self, model_type, use_question_as_suffix=False, use_openai_tokenizer_to_measure_length=False):
        configs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
        configs = {
            ModelType.MISTRAL: os.path.join(configs_dir, 'cpc-1.0-mistral.json'),
            ModelType.LLAMA: os.path.join(configs_dir, 'cpc-1.0-llama.json'),
        }
        configs = {
            ModelType.MISTRAL: {
                'config_path': os.path.join(configs_dir, 'cpc-1.0-mistral.json'),
                'lora_name_or_path': 'deadcode99/cpc-1.0-mistral-7b-ds-v5-iter66-lora-bidirectional-attn',
                'tokenizer_name_or_path': 'deadcode99/cpc-1.0-mistral-7b-tokenizer',
            },
            ModelType.LLAMA: {
                'config_path': os.path.join(configs_dir, 'cpc-1.0-llama.json'),
                'lora_name_or_path': 'deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn',
                'tokenizer_name_or_path': 'deadcode99/cpc-1.0-llama-1b-tokenizer',
            },
        }

        assert model_type in configs, f"Unsupported model type: {model_type}. Supported configuration are: {configs.keys()}"

        cfg = configs[model_type]

        self.model, self.tokenizer = load_model_and_tokenizer(**cfg)

        self.model.eval()

        with open(cfg['config_path']) as fin:
            train_conf = json.load(fin)

        self.processor = SamplePreprocessor(
            tokenizer=self.tokenizer,
            max_context_len=train_conf['max_seq_length'],
            use_question_as_suffix=use_question_as_suffix,
            sentence_embedding_type=SentenceEmbeddingType.AVG,
        )

        self.openai_tokenizer = None
        if use_openai_tokenizer_to_measure_length:
            self.openai_tokenizer = tiktoken.encoding_for_model('gpt-4')

    def _get_tokenizer_for_length_measure(self):
        if self.openai_tokenizer is not None:
            return self.openai_tokenizer
        return self.tokenizer

    def compress(self, context, question, compression_target_tokens, boost_sents_regexp=None):
        encodings = self.processor(
            context=context,
            question=question,
            question_for_suffix=question,
        )
        sents = compress_sample(
            model=self.model,
            tokenizer=self.tokenizer,
            openai_tokenizer=self._get_tokenizer_for_length_measure(),
            sample={
                'encodings': encodings,
            },
            compression_target_tokens=compression_target_tokens,
            boost_match_regex=boost_sents_regexp,
        )

        return ' '.join(sents)
