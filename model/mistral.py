from typing import List, Optional, Tuple, Union
import torch

from transformers import (
    MistralModel,
    MistralPreTrainedModel,
    MistralForCausalLM,
    MistralConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
    MistralMLP,
)
from torch import nn
from transformers.utils import logging
from model.attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from training.loss import HardNegativeNLLLoss
import training.loss as losses_module

from dataclasses import dataclass

from peft import PeftModel
from peft import LoraConfig, get_peft_model

import importlib.metadata
from packaging import version
from transformers.utils.import_utils import _is_package_available

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, StaticCache, SlidingWindowCache

from args import parse_args

from transformers import AutoTokenizer, AutoModel, AutoConfig
from util.torch_util import  mean_pooling, cos_sim


logger = logging.get_logger(__name__)


class ModifiedMistralAttention(MistralAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMistralFlashAttention2(MistralFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedMistralSdpaAttention(MistralSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


MISTRAL_ATTENTION_CLASSES = {
    "eager": ModifiedMistralAttention,
    "flash_attention_2": ModifiedMistralFlashAttention2,
    "sdpa": ModifiedMistralSdpaAttention,
}


class ModifiedMistralDecoderLayer(MistralDecoderLayer):
    def __init__(self, config: MistralConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


def is_transformers_attn_greater_or_equal_4_43_1():
    if not _is_package_available("transformers"):
        return False

    return version.parse(importlib.metadata.version("transformers")) >= version.parse(
        "4.43.1"
    )


class MistralBiModel(MistralModel):
    _no_split_modules = ["ModifiedMistralDecoderLayer"]

    def __init__(self, config: MistralConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "The current implementation of LlamaEncoderModel follows modeling_llama.py of transformers version >= 4.43.1"
            )
        MistralPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedMistralDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from forward() in transformers.models.mistral.modeling_mistral.MistralModel
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        use_cache: bool,
        output_attentions: bool,
    ):
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and use_cache:
                is_padding_right = (
                    attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                )
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )

            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask

            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.

        # cache_position must be valid here no matter which cache we use
        past_seen_tokens = cache_position[0] if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache
        if using_sliding_window_cache:
            target_length = max(sequence_length, self.config.sliding_window)
        # StaticCache
        elif using_static_cache:
            target_length = past_key_values.get_max_length()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.zeros(
                (sequence_length, target_length), dtype=dtype, device=device
            )  # causal_mask = torch.full(
            # (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            # )
            exclude_mask = torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            if self.config.sliding_window is not None:
                if (
                    not using_sliding_window_cache
                    or sequence_length > self.config.sliding_window
                ):
                    exclude_mask.bitwise_or_(
                        torch.arange(target_length, device=device)
                        <= (cache_position.reshape(-1, 1) - self.config.sliding_window)
                    )
            causal_mask *= exclude_mask
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                if attention_mask.dim() == 2:
                    mask_length = attention_mask.shape[-1]
                    padding_mask = (
                        causal_mask[:, :, :, :mask_length]
                        + attention_mask[:, None, None, :]
                    )
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[
                        :, :, :, :mask_length
                    ].masked_fill(padding_mask, min_dtype)

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


class MistralBiForMNTP(MistralForCausalLM):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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


@dataclass
class LMOutputWithContrastiveLoss(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MistralBiForMNTPandSentEmbeddings(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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
        attention_mask_not_masked: Optional[torch.Tensor] = None,

        sentence_data__input_ids: torch.LongTensor = None,
        sentence_data__attention_mask: Optional[torch.Tensor] = None,

        masked_sentence_in_context_input_ids: Optional[torch.Tensor] = None,
        masked_sentence_in_context_mask: Optional[torch.Tensor] = None,
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

        assert sentence_data__input_ids is not None
        assert sentence_data__attention_mask is not None

        assert masked_sentence_in_context_input_ids is not None
        assert masked_sentence_in_context_mask is not None

        # pos - ..context [sent_unmasked] context..
        with torch.no_grad():
            hid_states_pos = self.model(
                input_ids=input_ids_not_masked,
                attention_mask=attention_mask_not_masked,
            )[0]
        
        # neg - [sent_unmasked]
        with torch.no_grad():
            hid_states_neg = self.model(
                input_ids=sentence_data__input_ids,
                attention_mask=sentence_data__attention_mask,
            )[0]

        # x - ..context [sent_masked] context..
        hid_states_optimized = self.model(
            input_ids=masked_sentence_in_context_input_ids,
            attention_mask=attention_mask_not_masked,
        )[0]

        pos_emb = mean_pooling(hid_states_pos, masked_sentence_in_context_mask)
        neg_emb = mean_pooling(hid_states_neg, sentence_data__attention_mask)
        x_emb = mean_pooling(hid_states_optimized, masked_sentence_in_context_mask)
        loss_contrastive = self.sim_loss(x_emb, pos_emb, neg_emb)

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
        attention_mask_not_masked: Optional[torch.Tensor] = None,
        
        sentence_data__input_ids: torch.LongTensor = None,
        sentence_data__attention_mask: Optional[torch.Tensor] = None,

        masked_sentence_in_context_input_ids: Optional[torch.Tensor] = None,
        masked_sentence_in_context_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
            sentence_data__input_ids=sentence_data__input_ids,
            sentence_data__attention_mask=sentence_data__attention_mask,

            masked_sentence_in_context_input_ids=masked_sentence_in_context_input_ids,
            masked_sentence_in_context_mask=masked_sentence_in_context_mask,
        )
        result['loss_contrastive'] = loss_contrastive
        return LMOutputWithContrastiveLoss(**result)


class MistralBiForMNTPandSentEmbeddingsV2(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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


class MistralBiForMNTPandSentEmbeddingsV2_w_q_token(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # self.sim_loss = HardNegativeNLLLoss(
        #     scale=config.contrastive_loss_scale
        # )
        self.sim_loss = getattr(losses_module, config.contrastive_loss_name)(
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


class MistralBiForMNTPandSentEmbeddingsV2_w_q_token_non_contrastive(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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

        # neg_emb = torch.cat(neg_embs, dim=0)

        # with torch.no_grad():
        hid_states_query = self.model(
            input_ids=questions_input_ids,
            attention_mask=questions_attention_mask,
        )[0]
        
        query_emb = mean_pooling(hid_states_query, question_token_ids_mask)

        # loss_contrastive = self.sim_loss(query_emb, pos_emb, neg_emb)

        # return loss_contrastive
    
        pos_sim = similarity(query_emb, pos_emb)
        neg_sims = []
        for ne in neg_embs:
            neg_sim = similarity(query_emb, ne)
            neg_sims.append(neg_sim)

        loss_contrastive = torch.nn.BCELoss()(pos_sim, torch.ones_like(pos_sim))
        for ns in neg_sims:
            loss_contrastive += torch.nn.BCELoss()(ns, torch.zeros_like(ns))

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


class MistralBiForMNTPandSentEmbeddingsV3(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,
        
        instruction_mask: torch.LongTensor = None,
        question_mask: torch.LongTensor = None,
        separator_mask: torch.LongTensor = None,
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

        assert positive_sentences_input_ids is not None
        assert positive_sentences_attention_mask is not None

        assert instruction_mask is not None
        assert question_mask is not None
        assert separator_mask is not None

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
        
        query_emb = mean_pooling(hid_states, question_mask)

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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,
        
        instruction_mask: torch.LongTensor = None,
        question_mask: torch.LongTensor = None,
        separator_mask: torch.LongTensor = None,

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
                
                positive_sentences_input_ids=positive_sentences_input_ids,
                positive_sentences_attention_mask=positive_sentences_attention_mask,
        
                instruction_mask=instruction_mask,
                question_mask=question_mask,
                separator_mask=separator_mask,
            )
            result['loss_contrastive'] = loss_contrastive
            return LMOutputWithContrastiveLoss(**result)
        else:
            hid_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            return hid_states


class MistralBiForMNTPandSentEmbeddingsV2WEosAndQ(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        question_tokens_masks: torch.LongTensor = None,
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

        assert positive_sentences_input_ids is not None
        assert positive_sentences_attention_mask is not None

        assert question_tokens_masks is not None

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

        query_emb = mean_pooling(hid_states, question_tokens_masks)
        
        # query_emb = mean_pooling(hid_states_query, questions_attention_mask)

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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        question_tokens_masks: torch.LongTensor = None,

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
                
                positive_sentences_input_ids=positive_sentences_input_ids,
                positive_sentences_attention_mask=positive_sentences_attention_mask,

                question_tokens_masks=question_tokens_masks,
            )
            result['loss_contrastive'] = loss_contrastive
            return LMOutputWithContrastiveLoss(**result)
        else:
            hid_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            return hid_states


def similarity(a, b):
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    sim = a_norm * b_norm
    sim = sim.sum(1) * 0.5 + 0.5
    return sim


class MistralBiForMNTPandSentEmbeddingsV2WEosAndQNonContrastive(MistralBiForMNTP):
    def __init__(self, config):
        MistralPreTrainedModel.__init__(self, config)
        self.model = MistralBiModel(config)
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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        question_tokens_masks: torch.LongTensor = None,
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

        assert positive_sentences_input_ids is not None
        assert positive_sentences_attention_mask is not None

        assert question_tokens_masks is not None

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

        query_emb = mean_pooling(hid_states, question_tokens_masks)

        pos_sim = similarity(query_emb, pos_emb)
        neg_sims = []
        for ne in neg_embs:
            neg_sim = similarity(query_emb, ne)
            neg_sims.append(neg_sim)

        loss_contrastive = torch.nn.BCELoss()(pos_sim, torch.ones_like(pos_sim))
        for ns in neg_sims:
            loss_contrastive += torch.nn.BCELoss()(ns, torch.zeros_like(ns))

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
        
        positive_sentences_input_ids: torch.LongTensor = None,
        positive_sentences_attention_mask: torch.LongTensor = None,

        question_tokens_masks: torch.LongTensor = None,

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
                
                positive_sentences_input_ids=positive_sentences_input_ids,
                positive_sentences_attention_mask=positive_sentences_attention_mask,

                question_tokens_masks=question_tokens_masks,
            )
            result['loss_contrastive'] = loss_contrastive
            return LMOutputWithContrastiveLoss(**result)
        else:
            hid_states = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
            return hid_states
