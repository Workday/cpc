from transformers import AutoModelForCausalLM, LlamaConfig
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm
from peft import PeftConfig

from transformers.models.llama.modeling_llama import LlamaForCausalLM

from model.llama import LlamaBiForMNTPandSentEmbeddingsV2_w_q_token
from model.mistral import MistralBiForMNTPandSentEmbeddingsV2
from model.qwen2 import Qwen2BiForMNTPandSentEmbeddingsV2_w_q_token
from args import parse_args
from transformers import AutoConfig, AutoTokenizer
import torch
from peft import PeftModel

from model.common import get_model_mock_class, MODEL_TYPE_2_CLS, ensure_model_type

from util.util import SpecTokenType, SentenceEmbeddingType


class MultiLoraModel(nn.Module):
    def __init__(self, model_args, tokenizer):
        super().__init__()

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.token,
            "trust_remote_code": model_args.trust_remote_code,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, **config_kwargs
            )
        else:
            raise NotImplementedError("Not implemented")
    
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        config.contrastive_loss_scale = 20

        model_type = ensure_model_type(model_args.model_name_or_path)

        sentence_embedding_type = SentenceEmbeddingType.AVG

        bi_model_cls = MODEL_TYPE_2_CLS[sentence_embedding_type][model_type]
        
        self.bi_model = bi_model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            attn_implementation=model_args.attn_implementation,
        )
        config_causal = deepcopy(config)
        
        causal_mock_model_cls = get_model_mock_class(model_type)
        
        self.causal_model = causal_mock_model_cls(
            config_causal,
            self.bi_model.model,
            self.bi_model.lm_head
        )
    
    def generate_causal(self, *args, **kwargs):
        self.causal_model.set_adapter('causal')
        for l in self.bi_model.model.layers:
            l.is_causal = True

        return self.causal_model.generate(*args, **kwargs)

    def forward_bidirectional(self, *args, **kwargs):
        self.bi_model.model.set_adapter('default')
        for l in self.bi_model.model.layers:
            l.is_causal = False
        
        return self.bi_model(*args, **kwargs)

    def load_adapters(self, bi_lora_path, causal_lora_path):
        self.bi_model.model = PeftModel.from_pretrained(self.bi_model.model, bi_lora_path)
        self.causal_model = PeftModel.from_pretrained(self.causal_model, causal_lora_path, 'causal')
