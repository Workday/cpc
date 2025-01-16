import torch

from model.llama import (
    LlamaBiForMNTPandSentEmbeddingsV2,
    LlamaBiForMNTPandSentEmbeddingsV2_w_q_token
)

from model.mistral import (
    MistralBiForMNTPandSentEmbeddingsV2,
    MistralBiForMNTPandSentEmbeddingsV2_w_q_token,
)

from model.qwen2 import (
    Qwen2BiForMNTPandSentEmbeddingsV2,
    Qwen2BiForMNTPandSentEmbeddingsV2_w_q_token
)

from args import parse_args

from munch import Munch
from transformers import AutoTokenizer, AutoConfig

from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    Qwen2ForCausalLM
)

from typing import Optional, List

from util.util import SpecTokenType, SentenceEmbeddingType

from peft import get_peft_model, LoraConfig


class ModelType:
    MISTRAL='mistral'
    LLAMA='llama'
    QWEN2='qwen2'


MODEL_TYPE_2_CLS = {
    SentenceEmbeddingType.AVG: {
        ModelType.MISTRAL: MistralBiForMNTPandSentEmbeddingsV2,
        ModelType.LLAMA: LlamaBiForMNTPandSentEmbeddingsV2,
        ModelType.QWEN2: Qwen2BiForMNTPandSentEmbeddingsV2,
    },
}


def ensure_model_type(model_name_or_path):
    if 'mistral' in model_name_or_path.lower():
        return ModelType.MISTRAL
    elif 'llama' in model_name_or_path.lower():
        return ModelType.LLAMA
    elif 'qwen2' in model_name_or_path.lower():
        return ModelType.QWEN2
    else:
        raise ValueError('Unsupported model type: should be on of ["Mistral", "Llama", "Qwen2"]')


class LlamaForCausalLMMock(LlamaForCausalLM):
    def __init__(self, config, model_body, lm_head):
        super().__init__(config)

        self.model = model_body
        self.lm_head = lm_head


class MistralForCausalLMMock(MistralForCausalLM):
    def __init__(self, config, model_body, lm_head):
        super().__init__(config)

        self.model = model_body
        self.lm_head = lm_head


class Qwen2ForCausalLMMock(Qwen2ForCausalLM):
    def __init__(self, config, model_body, lm_head):
        super().__init__(config)

        self.model = model_body
        self.lm_head = lm_head


def get_model_mock_class(model_type: int):
    if model_type == ModelType.LLAMA:
        return LlamaForCausalLMMock
    if model_type == ModelType.MISTRAL:
        return MistralForCausalLMMock
    if model_type == ModelType.QWEN2:
        return Qwen2ForCausalLMMock
    raise ValueError(f'Unsupported mock model type: {model_type}')


def build_model(config_path, tokenizer_name_or_path):
    model_args, data_args, training_args, custom_args = parse_args(
        config_path
    )
    return build_model_from_args(model_args, custom_args, tokenizer_name_or_path)


def build_model_from_args(model_args, custom_args, tokenizer_name_or_path):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "padding_side": "left"
    }

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # blank, eos, mask
    if tokenizer.mask_token is None:
        if custom_args.mask_token_type == "blank":
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(
                f"mask_token_type {custom_args.mask_token_type} is not supported."
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.padding_side  = 'left'

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

    config.contrastive_loss_scale = custom_args.contrastive_loss_scale

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model_typename = ensure_model_type(model_args.model_name_or_path)

    sentence_embedding_type = SentenceEmbeddingType.AVG

    model_cls = MODEL_TYPE_2_CLS[sentence_embedding_type][model_typename]

    model = model_cls.from_pretrained(
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

    return Munch(
        # model stuff
        config=config,
        model=model,
        tokenizer=tokenizer,
        model_type=model_typename,
        # args
        model_args=model_args,
        custom_args=custom_args,
    )


def resize_model_embeddings_to_fit_tokenizer(model, tokenizer):
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model
