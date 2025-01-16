import os
import sys

from typing import List, Optional

import json
import re

import torch
import numpy as np

from util.util import (
    tokenize_and_clip_segments,
    split_text_into_sentences_keep_slashn,
    SentenceEmbeddingType,
    SpecTokenType,
)

from util.torch_util import mean_pooling, cos_sim



class SamplePreprocessor:
    def __init__(
        self,
        tokenizer,
        max_context_len: int, 
        use_question_as_suffix: bool = False,
        sentence_embedding_type: int = SentenceEmbeddingType.AVG):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.use_question_as_suffix = use_question_as_suffix
        self.sentence_embedding_type = sentence_embedding_type
    
    def chunkify(self, segments, segments_labels, suffix_question_segments):
        init_length = len(self.tokenizer.encode(' '.join(segments), add_special_tokens=False))
        question_length = len(self.tokenizer.encode(' '.join(suffix_question_segments), add_special_tokens=False))

        N = len(segments)

        num_chunks = 1
        good = False
        while not good:
            buckets = []
            buckets_labels = []
            n_per_chunk = N // num_chunks if N % num_chunks == 0 else (N // num_chunks) + 1
            for i in range(0, N, n_per_chunk):
                buckets.append(segments[i:i+n_per_chunk])
                buckets_labels.append(segments_labels[i:i+n_per_chunk])

            good = True
            for b in buckets:
                chunk = ' '.join(b)
                if len(self.tokenizer.encode(chunk, add_special_tokens=False)) > self.max_context_len:
                    good = False
                    break
            if not good:
                num_chunks += 1

        # Rearrange
        lens = [
            [
                len(self.tokenizer.encode(s, add_special_tokens=False))
                for s in b
            ]
            for b in buckets
        ]

        question_labels = [1 for _ in range(len(suffix_question_segments))]
        
        new_buckets = []
        new_buckets_labels = []
        cur_bucket = []
        cur_bucket_labels = []
        cur_len = -1
        for b, bsl, bslens in zip(buckets, buckets_labels, lens):
            for s, sl, slen in zip(b, bsl, bslens):
                cur_bucket.append(s)
                cur_bucket_labels.append(sl)
                
                cur_len += slen + 1
                if cur_len >= init_length / num_chunks:
                    new_buckets.append(cur_bucket)
                    new_buckets_labels.append(cur_bucket_labels)
                    cur_bucket = []
                    cur_bucket_labels = []
                    cur_len = -1
        if cur_bucket:
            new_buckets.append(cur_bucket)
            new_buckets_labels.append(cur_bucket_labels)
            cur_bucket = []
            cur_bucket_labels = []
            cur_len = -1
        
        buckets = new_buckets
        buckets_labels = new_buckets_labels

        return buckets, buckets_labels
    
    def _ensure_sents_not_too_long(self, sents):
        shortened_sents = []
        for s in sents:
            tokens = self.tokenizer.encode(s, add_special_tokens=False)
            if len(tokens) > self.max_context_len:
                chunk_size = self.max_context_len // 20
                n_chunks = int(np.ceil(len(tokens) / chunk_size))
                token_chunks = [tokens[i*chunk_size:i*(chunk_size+1)] for i in range(n_chunks)]
                smaller_sents = [self.tokenizer.decode(chunk) for chunk in token_chunks]
                shortened_sents.extend(smaller_sents)
            else:
                shortened_sents.append(s)
        return shortened_sents

    def __call__(self, context, question, question_for_suffix=None):
        sents = split_text_into_sentences_keep_slashn(context, language='en')
        sents = [t for t in sents if t.strip()]
        sents = self._ensure_sents_not_too_long(sents)
        sents_question_as_suffix = []
        if self.use_question_as_suffix:
            sents_question_as_suffix = split_text_into_sentences_keep_slashn(question_for_suffix, language='en')
            sents_question_as_suffix = [t for t in sents_question_as_suffix if t.strip()]

        sents_labels = [0 for _ in sents] + [1 for _ in sents_question_as_suffix]
        
        chunks, chunks_labels = self.chunkify(
            sents + sents_question_as_suffix,
            sents_labels,
            sents_question_as_suffix,
        )

        end_of_sent_token_id = None
        end_of_question_token_id = None

        encodings = [
            tokenize_and_clip_segments(
                tokenizer=self.tokenizer,
                segments=chunk,
                segments_labels=chunk_labels,
                max_seq_len=self.max_context_len,
                sentence_embedding_type=self.sentence_embedding_type,
                end_of_sentence_token=end_of_sent_token_id
            )
            for chunk, chunk_labels in zip(chunks, chunks_labels)
        ]

        encodings_question = self.tokenizer.batch_encode_plus([question], add_special_tokens=False, padding='longest')

        return {
            'context': encodings,
            'question': encodings_question,
        }


@torch.no_grad()
def compress_sample(model, tokenizer, openai_tokenizer, sample, compression_target_tokens, boost_match_regex):

    def re_checker(text):
        if boost_match_regex is not None:
            return re.match(boost_match_regex, text) is not None
        return False

    encodings = sample['encodings']
    
    global_index = 0
    all_similarities = []

    all_sents = []
    all_sents_labels = []
    for enc in encodings['context']:
        all_sents.extend(enc['segments'])
        all_sents_labels.extend(enc['segments_labels'])

    encodings_question = encodings['question']
    inputs_question = {
        'input_ids': torch.LongTensor(encodings_question['input_ids']).cuda(),
        'attention_mask': torch.LongTensor(encodings_question['attention_mask']).cuda(),
    }

    outputs_question = model(**inputs_question, is_train=False)
    q_emb = mean_pooling(outputs_question, inputs_question['attention_mask'])
    
    for encodings_context in encodings['context']:
        inputs_context = {
            'input_ids': torch.LongTensor(encodings_context['text_input_ids'])[None].cuda(),
            'attention_mask': torch.LongTensor([1 for _ in encodings_context['text_input_ids']])[None].cuda(),
            'text_segment_ids': torch.LongTensor(encodings_context['text_segment_ids'])[None].cuda()
        }
        inputs_context_filtered = {
            k: inputs_context[k] for k in ['input_ids', 'attention_mask']
        }
        outputs_context = model(**inputs_context_filtered, is_train=False)

        max_seg_idx = encodings_context['text_segment_ids'][-1]
        embeddings_context_sentwise = []

        for sent_id in range(max_seg_idx + 1):
            mask = (inputs_context['text_segment_ids'] == sent_id)
            emb = mean_pooling(outputs_context, mask)
            embeddings_context_sentwise.append(emb)
    
        embeddings_context_sentwise = torch.cat(embeddings_context_sentwise, 0)
    
        sim = cos_sim(embeddings_context_sentwise, q_emb)[:, 0].cpu().numpy()
    
        assert len(sim) == len(encodings_context['segments'])
    
        for seg, seg_is_fake, s in zip(encodings_context['segments'], encodings_context['segments_labels'], sim):
            all_similarities.append((seg, s, global_index, re_checker(seg), seg_is_fake))
            global_index += 1

    taken = -1
    
    indices = set()
    
    for seg, s, idx, re_score, seg_is_fake in sorted(all_similarities, reverse=True, key=lambda x: (x[3], x[1])):
        if seg_is_fake:
            continue
        ln = len(openai_tokenizer.encode(seg))
        taken += ln + 1
        indices.add(idx)
        if taken > compression_target_tokens:
            break

    compressed_segments = [seg for seg, s, idx, re_score, seg_is_fake in all_similarities if idx in indices]
    return compressed_segments
