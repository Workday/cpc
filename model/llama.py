import torch

from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from torch import nn
from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available

from peft import PeftModel


from transformers.modeling_outputs import CausalLMOutputWithPast

from training.loss import HardNegativeNLLLoss
import training.loss as losses_module

from typing import List, Optional, Tuple, Union

from peft import PeftModel
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from dataclasses import dataclass

from util.torch_util import  mean_pooling, cos_sim

from util.util import SpecTokenType


@dataclass
class LMOutputWithContrastiveLoss(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


logger = logging.get_logger(__name__)


def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.1"
    )


class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,
    "sdpa": ModifiedLlamaSdpaAttention,
}


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class LlamaBiModel(LlamaModel):
    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.43.1"
            )
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=device
        )  # in original implementation - torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :, None, None, :
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ..., :mask_length
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        return causal_mask


class LlamaBiForMNTP(LlamaForCausalLM):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)


class LlamaBiForMNTPandSentEmbeddingsV2(LlamaBiForMNTP):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.sim_loss = HardNegativeNLLLoss(
            scale=config.contrastive_loss_scale
        )
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_mntp(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return dict(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def forward_sentence_embeddnigs(
        self,
        input_ids_not_masked: torch.LongTensor = None,
        attention_mask_not_masked: torch.LongTensor = None,
        
        positive_tokens_masks: torch.LongTensor = None,
        negative_tokens_masks: torch.LongTensor = None,

        questions_input_ids: torch.LongTensor = None,
        questions_attention_mask: torch.LongTensor = None,
        
        answers_input_ids: torch.LongTensor = None,
        answers_attention_mask: torch.LongTensor = None,
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids_not_masked - positive (pos) sample. all unmasked
            attention_mask_not_masked - positive sample attention mask (0 for spesial tokens and padding)
            
            masked_sentence_in_context_input_ids - target (x) sample. context unmasked. 1 random sent masked with some prob
            masked_sentence_in_context_mask - maskof the "masked with prob" sentence in context. like 0 0 0 1 1 1 1 0 0 0 0

            sentence_data__input_ids - 1 "random" sentence (w/o context) for each sample that is masked in input_ids_one_sent_masked
            sentence_data__attention_mask - attention_mask for previous
        ```"""

        assert input_ids_not_masked is not None
        assert attention_mask_not_masked is not None

        assert positive_tokens_masks is not None
        assert negative_tokens_masks is not None

        assert questions_input_ids is not None
        assert questions_attention_mask is not None

        assert answers_input_ids is not None
        assert answers_attention_mask is not None

        assert positive_sentences_input_ids is not None
        assert positive_sentences_attention_mask is not None

        # pos - ..context [sent_unmasked] context..
        # with torch.no_grad():
        hid_states = self.model(
            input_ids=input_ids_not_masked,
            attention_mask=attention_mask_not_masked,
        )[0]

        pos_emb = mean_pooling(hid_states, positive_tokens_masks)

        neg_embs = []
        for neg_mask in negative_tokens_masks.split(1, dim=1):
            neg_mask = neg_mask.squeeze(1)
            neg_emb = mean_pooling(hid_states, neg_mask)
            neg_embs.append(neg_emb)

        neg_emb = torch.cat(neg_embs, dim=0)

        with torch.no_grad():
            hid_states_query = self.model(
                input_ids=questions_input_ids,
                attention_mask=questions_attention_mask,
            )[0]
        
        query_emb = mean_pooling(hid_states_query, questions_attention_mask)

        loss_contrastive = self.sim_loss(query_emb, pos_emb, neg_emb)

        return loss_contrastive

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        input_ids_not_masked: torch.LongTensor = None,
        attention_mask_not_masked: torch.LongTensor = None,
        
        positive_tokens_masks: torch.LongTensor = None,
        negative_tokens_masks: torch.LongTensor = None,

        questions_input_ids: torch.LongTensor = None,
        questions_attention_mask: torch.LongTensor = None,
        
        answers_input_ids: torch.LongTensor = None,
        answers_attention_mask: torch.LongTensor = None,
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        is_train=True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if is_train:
            result = self.forward_mntp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            loss_contrastive = self.forward_sentence_embeddnigs(
                input_ids_not_masked=input_ids_not_masked,
                attention_mask_not_masked=attention_mask_not_masked,
                
                positive_tokens_masks=positive_tokens_masks,
                negative_tokens_masks=negative_tokens_masks,

                questions_input_ids=questions_input_ids,
                questions_attention_mask=questions_attention_mask,
                
                answers_input_ids=answers_input_ids,
                answers_attention_mask=answers_attention_mask,
                
                positive_sentences_input_ids=positive_sentences_input_ids,
                positive_sentences_attention_mask=positive_sentences_attention_mask,
            )
            result['loss_contrastive'] = loss_contrastive
            return LMOutputWithContrastiveLoss(**result)
        else:
            hid_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            return hid_states


class LlamaBiForMNTPandSentEmbeddingsV2_w_q_token(LlamaBiForMNTP):
    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.sim_loss = HardNegativeNLLLoss(
            scale=config.contrastive_loss_scale
        )
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward_mntp(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return dict(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def forward_sentence_embeddnigs(
        self,
        input_ids_not_masked: torch.LongTensor = None,
        attention_mask_not_masked: torch.LongTensor = None,
        
        positive_tokens_masks: torch.LongTensor = None,
        negative_tokens_masks: torch.LongTensor = None,

        questions_input_ids: torch.LongTensor = None,
        questions_attention_mask: torch.LongTensor = None,
        question_token_ids_mask: torch.LongTensor = None,
        
        answers_input_ids: torch.LongTensor = None,
        answers_attention_mask: torch.LongTensor = None,
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids_not_masked - positive (pos) sample. all unmasked
            attention_mask_not_masked - positive sample attention mask (0 for spesial tokens and padding)
            
            masked_sentence_in_context_input_ids - target (x) sample. context unmasked. 1 random sent masked with some prob
            masked_sentence_in_context_mask - maskof the "masked with prob" sentence in context. like 0 0 0 1 1 1 1 0 0 0 0

            sentence_data__input_ids - 1 "random" sentence (w/o context) for each sample that is masked in input_ids_one_sent_masked
            sentence_data__attention_mask - attention_mask for previous
        ```"""

        assert input_ids_not_masked is not None
        assert attention_mask_not_masked is not None

        assert positive_tokens_masks is not None
        assert negative_tokens_masks is not None

        assert questions_input_ids is not None
        assert questions_attention_mask is not None
        assert question_token_ids_mask is not None

        assert answers_input_ids is not None
        assert answers_attention_mask is not None

        assert positive_sentences_input_ids is not None
        assert positive_sentences_attention_mask is not None

        # pos - ..context [sent_unmasked] context..
        # with torch.no_grad():
        hid_states = self.model(
            input_ids=input_ids_not_masked,
            attention_mask=attention_mask_not_masked,
        )[0]

        pos_emb = mean_pooling(hid_states, positive_tokens_masks)

        neg_embs = []
        for neg_mask in negative_tokens_masks.split(1, dim=1):
            neg_mask = neg_mask.squeeze(1)
            neg_emb = mean_pooling(hid_states, neg_mask)
            neg_embs.append(neg_emb)

        neg_emb = torch.cat(neg_embs, dim=0)

        # with torch.no_grad():
        hid_states_query = self.model(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
        )[0]
        
        query_emb = mean_pooling(hid_states_query, question_token_ids_mask)

        loss_contrastive = self.sim_loss(query_emb, pos_emb, neg_emb)

        return loss_contrastive

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
        input_ids_not_masked: torch.LongTensor = None,
        attention_mask_not_masked: torch.LongTensor = None,
        
        positive_tokens_masks: torch.LongTensor = None,
        negative_tokens_masks: torch.LongTensor = None,

        questions_input_ids: torch.LongTensor = None,
        questions_attention_mask: torch.LongTensor = None,
        question_token_ids_mask: torch.LongTensor = None,
        
        answers_input_ids: torch.LongTensor = None,
        answers_attention_mask: torch.LongTensor = None,
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        is_train=True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if is_train:
            result = self.forward_mntp(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            loss_contrastive = self.forward_sentence_embeddnigs(
                input_ids_not_masked=input_ids_not_masked,
                attention_mask_not_masked=attention_mask_not_masked,
                
                positive_tokens_masks=positive_tokens_masks,
                negative_tokens_masks=negative_tokens_masks,

                questions_input_ids=questions_input_ids,
                questions_attention_mask=questions_attention_mask,
                question_token_ids_mask=question_token_ids_mask,
                
                answers_input_ids=answers_input_ids,
                answers_attention_mask=answers_attention_mask,
                
                positive_sentences_input_ids=positive_sentences_input_ids,
                positive_sentences_attention_mask=positive_sentences_attention_mask,
            )
            result['loss_contrastive'] = loss_contrastive
            return LMOutputWithContrastiveLoss(**result)
        else:
            hid_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            return hid_states
