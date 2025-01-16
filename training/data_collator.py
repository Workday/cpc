
from copy import deepcopy
import random
from typing import Optional, Any, Tuple

import torch
from transformers import DataCollatorForLanguageModeling

from util.util import SpecTokenType


def sentences_data_collator(batch, pad_token_id, return_pt=False):
    new_batch = {
        'text_input_ids': [],
        'text_attention_mask': [],
        'text_segment_ids': [],
        
        'sentence_input_ids': [],
        'sentence_attention_mask': [],
        'sample_idx': []
    }
    max_text_input_ids_len = max(
        len(sample['text_input_ids']) for sample in batch
    )
    max_sentence_input_ids_len = max(
        max([len(sii) for sii in sample['sentence_input_ids']]) for sample in batch
    )
    for sample_idx, sample in enumerate(batch):
        tokens_len = len(sample['text_input_ids'])
        pad_size = max_text_input_ids_len - tokens_len
        text_attention_mask = [0] * pad_size + [1] * tokens_len
        text_input_ids = [pad_token_id] * pad_size + sample['text_input_ids']
        text_segment_ids = [-1] * pad_size + sample['text_segment_ids']
        for sii in sample['sentence_input_ids']:
            sent_pad_size = max_sentence_input_ids_len - len(sii)
            sii_pad = [pad_token_id] * sent_pad_size + sii
            sii_attention_mask = [0] * sent_pad_size + [1] * len(sii)
            new_batch['sentence_input_ids'].append(sii_pad)
            new_batch['sentence_attention_mask'].append(sii_attention_mask)
            new_batch['sample_idx'].append(sample_idx)
        new_batch['text_input_ids'].append(text_input_ids)
        new_batch['text_attention_mask'].append(text_attention_mask)
        new_batch['text_segment_ids'].append(text_segment_ids)

    if return_pt:
        new_batch = {
            k: torch.LongTensor(v) for k, v in new_batch.items()
        }
    return new_batch
        

class DataCollatorForLanguageModelingWithFullMasking(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK, 0% random, 0% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


def get_mlm_data_collator_cls(data_collator_type):
    data_collator_cls = None
    if data_collator_type == "all_mask":
        data_collator_cls = DataCollatorForLanguageModelingWithFullMasking
    elif data_collator_type == "default":
        data_collator_cls = DataCollatorForLanguageModeling
    else:
        raise ValueError(
            f"data_collator_type {data_collator_type} is not supported."
        )
    return data_collator_cls


class DataCollatorForMNTPandContrastiveLearning:
    def __init__(self, tokenizer, data_collator_type, mlm_probability, pad_to_multiple_of_8=False, contrastive_mask_prob=0.5):
        data_collator_cls = get_mlm_data_collator_cls(data_collator_type)
        self.tokenizer = tokenizer
        self.mlm_collator = data_collator_cls(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
        self.special_tokens_map = {}
        for k in tokenizer.special_tokens_map:
            tok = tokenizer.special_tokens_map[k]
            self.special_tokens_map[tokenizer.vocab[tok]] = tok
        self.contrastive_mask_prob = contrastive_mask_prob

    def _dct_to_list(self, dct):
        res = []
        keys = list(dct.keys())
        N = len(dct[keys[0]])
        for i in range(N):
            s = {}
            for k in keys:
                s[k] = dct[k][i]
            res.append(s)
        return res

    def _get_one_sentence_per_batch(self, single_sentence_input_ids, single_sentence_attention_mask, sentence_batch_ids):
        l = 0
        r = l
        N = len(sentence_batch_ids)

        input_ids = []
        attention_mask = []
        sentence_in_batch_id = []
        while r < N:
            while r < N and sentence_batch_ids[r] == sentence_batch_ids[l]:
                r += 1

            sent_idx = random.randint(l, r - 1)
            sentence_in_batch_id.append(sent_idx - l)
            input_ids.append(single_sentence_input_ids[sent_idx])
            attention_mask.append(single_sentence_attention_mask[sent_idx])
            l = r
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentence_in_batch_id': sentence_in_batch_id,
        }

    def _mask_sentences_in_context(self, input_ids, text_segment_ids, sentence_ids):
        masked_input_ids = []
        sentence_in_context_mask = []
        for iids, text_segm, sent_id in zip(input_ids, text_segment_ids, sentence_ids):
            iids_masked = deepcopy(iids)
            mask = []
            for i in range(len(iids)):
                if text_segm[i] == sent_id:
                    if random.random() < self.contrastive_mask_prob:
                        iids_masked[i] = self.tokenizer.mask_token_id
                    mask.append(1)
                else:
                    mask.append(0)
            masked_input_ids.append(iids_masked)
            sentence_in_context_mask.append(mask)
            
        return {
            'masked_input_ids': masked_input_ids,
            'sentence_in_context_mask': sentence_in_context_mask
        }
    
    def __call__(self, batch):
        batch_sentencewise = sentences_data_collator(batch, self.tokenizer.pad_token_id)

        full_text_inputs = {
            'input_ids': batch_sentencewise['text_input_ids'],
            'attention_mask': batch_sentencewise['text_attention_mask'],
            'special_tokens_mask': [
                [int(tok in self.special_tokens_map) for tok in input_ids] for input_ids in batch_sentencewise['text_input_ids']],
            'text_segment_ids': batch_sentencewise['text_segment_ids'],
        }
        
        single_sentence_inputs = {
            'input_ids': batch_sentencewise['sentence_input_ids'],
            'attention_mask': batch_sentencewise['sentence_attention_mask'],
            'special_tokens_mask': [
                [int(tok in self.special_tokens_map) for tok in input_ids] for input_ids in batch_sentencewise['sentence_input_ids']],
            'sample_idx': batch_sentencewise['sample_idx'],
        }
        

        full_text_inputs_mlm = self.mlm_collator(self._dct_to_list(full_text_inputs))

        sentence_data = self._get_one_sentence_per_batch(
            single_sentence_input_ids=single_sentence_inputs['input_ids'],
            single_sentence_attention_mask=single_sentence_inputs['attention_mask'],
            sentence_batch_ids=single_sentence_inputs['sample_idx'],
        )

        masked_sentences_in_context_data = self._mask_sentences_in_context(
            input_ids=full_text_inputs['input_ids'],
            text_segment_ids=full_text_inputs['text_segment_ids'],
            sentence_ids=sentence_data['sentence_in_batch_id'],
        )
        
        outputs = {
            'input_ids': full_text_inputs_mlm['input_ids'],
            'attention_mask': full_text_inputs_mlm['attention_mask'],
            'labels': full_text_inputs_mlm['labels'],

            'input_ids_not_masked': torch.LongTensor(full_text_inputs['input_ids']),
            'attention_mask_not_masked': torch.LongTensor(full_text_inputs['attention_mask']),

            'sentence_data__input_ids': torch.LongTensor(sentence_data['input_ids']),
            'sentence_data__attention_mask': torch.LongTensor(sentence_data['attention_mask']),

            'masked_sentence_in_context_input_ids': torch.LongTensor(masked_sentences_in_context_data['masked_input_ids']),
            'masked_sentence_in_context_mask': torch.LongTensor(masked_sentences_in_context_data['sentence_in_context_mask']),
        }
        
        return outputs


class DataCollatorForMNTPandContrastiveLearningV2:
    def __init__(self, 
                 tokenizer, 
                 data_collator_type, 
                 mlm_probability, 
                 pad_to_multiple_of_8=False, 
                 num_negatives=2, 
                 tail_neg_ratio=0.5,
                 negatives_key="neg_samples",
                ):
        data_collator_cls = get_mlm_data_collator_cls(data_collator_type)
        self.tokenizer = tokenizer
        self.mlm_collator = data_collator_cls(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
        self.special_tokens_map = {}
        for k in tokenizer.special_tokens_map:
            tok = tokenizer.special_tokens_map[k]
            self.special_tokens_map[tokenizer.vocab[tok]] = tok
        self.num_negatives = num_negatives
        self.tail_neg_ratio = tail_neg_ratio
        self.negatives_key = negatives_key
        

    def _dct_to_list(self, dct):
        res = []
        keys = list(dct.keys())
        N = len(dct[keys[0]])
        for i in range(N):
            s = {}
            for k in keys:
                s[k] = dct[k][i]
            res.append(s)
        return res

    def _get_one_sentence_per_batch(self, single_sentence_input_ids, single_sentence_attention_mask, sentence_batch_ids):
        l = 0
        r = l
        N = len(sentence_batch_ids)

        input_ids = []
        attention_mask = []
        sentence_in_batch_id = []
        while r < N:
            while r < N and sentence_batch_ids[r] == sentence_batch_ids[l]:
                r += 1

            sent_idx = random.randint(l, r - 1)
            sentence_in_batch_id.append(sent_idx - l)
            input_ids.append(single_sentence_input_ids[sent_idx])
            attention_mask.append(single_sentence_attention_mask[sent_idx])
            l = r
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentence_in_batch_id': sentence_in_batch_id,
        }

    def _mask_sentences_in_context(self, input_ids, text_segment_ids, sentence_ids):
        masked_input_ids = []
        sentence_in_context_mask = []
        for iids, text_segm, sent_id in zip(input_ids, text_segment_ids, sentence_ids):
            iids_masked = deepcopy(iids)
            mask = []
            for i in range(len(iids)):
                if text_segm[i] == sent_id:
                    if random.random() < self.contrastive_mask_prob:
                        iids_masked[i] = self.tokenizer.mask_token_id
                    mask.append(1)
                else:
                    mask.append(0)
            masked_input_ids.append(iids_masked)
            sentence_in_context_mask.append(mask)
            
        return {
            'masked_input_ids': masked_input_ids,
            'sentence_in_context_mask': sentence_in_context_mask
        }

    def sample_negatives(self, negative_indices, pos_idx):
        result = random.sample(negative_indices, self.num_negatives)
        assert pos_idx not in result
        return result

    def pad(self, l, value):
        n = max(len(ll) for ll in l)
        res = []
        for ll in l:
            res.append(
                [value] * (n - len(ll)) + ll
            )
        return res
    
    def __call__(self, batch):
        batch_sentencewise = sentences_data_collator(batch, self.tokenizer.pad_token_id)

        
        full_text_inputs = {
            'input_ids': batch_sentencewise['text_input_ids'],
            'attention_mask': batch_sentencewise['text_attention_mask'],
            'special_tokens_mask': [
                [int(tok in self.special_tokens_map) for tok in input_ids] for input_ids in batch_sentencewise['text_input_ids']],
            'text_segment_ids': batch_sentencewise['text_segment_ids'],
            'positives_ids': [s['pos_sent_idx'] for s in batch],
            'negatives_ids': [self.sample_negatives(s[self.negatives_key], s['pos_sent_idx']) for s in batch],
        }

        positive_tokens_masks = []
        negative_tokens_masks = []
        
        for i in range(len(full_text_inputs['input_ids'])):
            pos_idx = full_text_inputs['positives_ids'][i]
            pos_mask = [int(seg_idx == pos_idx) for seg_idx in full_text_inputs['text_segment_ids'][i]]
            negative_tokens_masks.append([])
            for neg_idx in full_text_inputs['negatives_ids'][i]:
                negative_tokens_masks[-1].append(
                    [int(seg_idx == neg_idx) for seg_idx in full_text_inputs['text_segment_ids'][i]]
                )
            positive_tokens_masks.append(pos_mask)


        positive_sentences_input_ids = []
        positive_sentences_attention_mask = []
        for i in range(len(full_text_inputs['input_ids'])):
            pos_idx = full_text_inputs['positives_ids'][i]
            sentence_input_ids = []
            sentence_attention_mask = []
            for j, t in zip(full_text_inputs['text_segment_ids'][i], full_text_inputs['input_ids'][i]):
                if j == pos_idx:
                    sentence_input_ids.append(t)
                    sentence_attention_mask.append(1)
            positive_sentences_input_ids.append(sentence_input_ids)
            positive_sentences_attention_mask.append(sentence_attention_mask)

        positive_sentences_input_ids = self.pad(positive_sentences_input_ids, self.tokenizer.pad_token_id)
        positive_sentences_attention_mask = self.pad(positive_sentences_attention_mask, 0)
        
        questions_batch = self.tokenizer.batch_encode_plus([b['question'] for b in batch], add_special_tokens=False, padding='longest')
        answers_batch = self.tokenizer.batch_encode_plus([b['answer'] for b in batch], add_special_tokens=False, padding='longest')

        full_text_inputs_mlm = self.mlm_collator(
            self._dct_to_list(
                {k: v for k, v in full_text_inputs.items() if k not in ['positives_ids', 'negatives_ids']}
            )
        )
        
        return {
            'input_ids': full_text_inputs_mlm['input_ids'],
            'attention_mask': full_text_inputs_mlm['attention_mask'],
            'labels': full_text_inputs_mlm['labels'],

            'input_ids_not_masked': torch.LongTensor(full_text_inputs['input_ids']),
            'attention_mask_not_masked': torch.LongTensor(full_text_inputs['attention_mask']),
            
            'positive_tokens_masks': torch.LongTensor(positive_tokens_masks),
            'negative_tokens_masks': torch.LongTensor(negative_tokens_masks),

            'questions_input_ids': torch.LongTensor(questions_batch['input_ids']),
            'questions_attention_mask': torch.LongTensor(questions_batch['attention_mask']),
            
            'answers_input_ids': torch.LongTensor(answers_batch['input_ids']),
            'answers_attention_mask': torch.LongTensor(answers_batch['attention_mask']),
            
            'positive_sentences_input_ids': torch.LongTensor(positive_sentences_input_ids),
            'positive_sentences_attention_mask': torch.LongTensor(positive_sentences_attention_mask),
        }
