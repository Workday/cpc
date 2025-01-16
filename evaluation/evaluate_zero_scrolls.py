import os

from sentence_splitter import split_text_into_sentences
from munch import Munch
from tqdm import tqdm
import json

from model.mistral import mean_pooling
from model.model import load_model_and_tokenizer
from args import parse_args
from transformers import AutoConfig, AutoTokenizer
import torch
from peft import PeftModel
from datasets import load_dataset
from argparse import ArgumentParser
from training.loss import cos_sim
import numpy as np
import colorprint3


def pack_until_max_seq_len(tokenizer, segments, header_labels=None, doc_indices=None, max_seq_len=6144):
    encodings = {
        'text_input_ids': [],
        'text_segment_ids': [],
        'sentence_input_ids': [],
        'segments': [],
    }
    assert (header_labels is None) == (doc_indices is None)
    if header_labels is not None:
        encodings.update({
            'token_is_from_header': [],
            'token_doc_index': [],
        })
        for i, (seg, seg_is_header, seg_doc_idx) in enumerate(zip(segments, header_labels, doc_indices)):
            inputs = tokenizer.encode(seg, add_special_tokens=False)
            if len(encodings['text_input_ids']) + len(inputs) > max_seq_len:
                break
            encodings['text_input_ids'].extend(inputs)
            encodings['text_segment_ids'].extend([i]*len(inputs))
            encodings['sentence_input_ids'].append(inputs)
            encodings['segments'].append(seg)
            encodings['token_is_from_header'].extend([seg_is_header]*len(inputs))
            encodings['token_doc_index'].extend([seg_doc_idx]*len(inputs))
    else:
        for i, seg in enumerate(segments):
            inputs = tokenizer.encode(seg, add_special_tokens=False)
            if len(encodings['text_input_ids']) + len(inputs) > max_seq_len:
                break
            encodings['text_input_ids'].extend(inputs)
            encodings['text_segment_ids'].extend([i]*len(inputs))
            encodings['sentence_input_ids'].append(inputs)
            encodings['segments'].append(seg)
    return Munch(encodings)


def pack_docs_into_buckets(docs, max_context_length):
    num_buckets = 1
    counts = [
        len(tokenizer.encode(d, add_special_tokens=False)) for d in docs
    ]
    N = len(docs)
    while True:
        buckets = []
        num_per_bucket = (N + num_buckets - 1) // num_buckets
        for i in range(0, N, num_per_bucket):
            buckets.append(docs[i:i+num_per_bucket])

        good = True
        for b in buckets:
            txt = ' '.join(b)
            if len(tokenizer.encode(txt, add_special_tokens=False)) > max_context_length:
                good = False
                break
        if good:
            return buckets
        num_buckets += 1
    return buckets


def prepare_bucket_for_inference(bucket, question, header_pattern, max_context_len=6144):
    all_sentences = []
    header_labels = []
    doc_indices = []
    for doc_idx, doc in enumerate(bucket):
        sents = split_text_into_sentences(doc, language='en')
        sents = [s.strip() for s in sents if s.strip()]
        assert sents[0].startswith(header_pattern)
        all_sentences.append(sents[0])
        header_labels.append(1)
        doc_indices.append(doc_idx)
        for i, s in enumerate(sents[1:], 1):
            all_sentences.append(s)
            header_labels.append(0)
            doc_indices.append(doc_idx)

    encodings = pack_until_max_seq_len(
        tokenizer,
        segments=all_sentences,
        header_labels=header_labels,
        doc_indices=doc_indices,
        max_seq_len=max_context_len
    )

    encodings.update({
        'header_labels': header_labels,
        'doc_indices': doc_indices
    })
    
    encodings_question = tokenizer.batch_encode_plus([question], add_special_tokens=False, padding='longest')
    
    return {
        'context': encodings,
        'question': encodings_question,
    }


@torch.no_grad()
def forward_pass(inputs):
    inputs_filtered = {
        k: inputs[k] for k in ['input_ids', 'attention_mask']
    }
    
    with torch.no_grad():
        outputs = model(**inputs_filtered, is_train=False)
    return outputs


def _get_scores_impl(inputs_context, outputs_context, inputs_question, outputs_question):
    embeddings_context_sentwise = []
    max_emb_id = inputs_context['text_segment_ids'].max().item() + 1

    for sent_id in range(max_emb_id):
        mask = (inputs_context['text_segment_ids'] == sent_id)
        emb = mean_pooling(outputs_context, mask)
        embeddings_context_sentwise.append(emb)
    
    embeddings_context_sentwise = torch.cat(embeddings_context_sentwise, 0)
    
    q_emb = mean_pooling(outputs_question, inputs_question['attention_mask'])
    
    scores = cos_sim(embeddings_context_sentwise, q_emb)[:, 0].cpu().numpy()

    sent_id2num = {}
    for sent_id in range(max_emb_id):
        mask = (inputs_context['text_segment_ids'] == sent_id)
        s = mask.sum().item()
        sent_id2num[sent_id] = int(s)
    return scores, sent_id2num


def get_scores(encodings_context, encodings_question):
    inputs_context = {
        'input_ids': torch.LongTensor(encodings_context['text_input_ids'])[None].cuda(),
        'attention_mask': torch.LongTensor([1 for _ in encodings_context['text_input_ids']])[None].cuda(),
        'text_segment_ids': torch.LongTensor(encodings_context['text_segment_ids'])[None].cuda()
    }
    outputs_context = forward_pass(inputs_context)
    
    inputs_question = {
        'input_ids': torch.LongTensor(encodings_question['input_ids']).cuda(),
        'attention_mask': torch.LongTensor(encodings_question['attention_mask']).cuda(),
    }
    outputs_question = forward_pass(inputs_question)

    scores, sent_id2num = _get_scores_impl(inputs_context, outputs_context, inputs_question, outputs_question)
    return scores, sent_id2num


def infer_bucket(bucket, question, header_template, max_context_len):
    encodings = prepare_bucket_for_inference(bucket, question, header_template, max_context_len)
    encodings_context = encodings['context']
    encodings_question = encodings['question']
    scores, sent_id2num = get_scores(encodings_context, encodings_question)
    return encodings_context, scores, sent_id2num


def aggregate_infos(encodings_context, scores, sent_id2num):
    assert len(encodings_context['doc_indices']) == len(scores), (len(encodings_context['doc_indices']), len(scores))
    assert len(encodings_context['header_labels']) == len(scores)

    N_docs = encodings_context['doc_indices'][-1] + 1
    N = len(scores)
    cur_idx = 0
    infos = []
    while cur_idx < N:
        assert encodings_context['header_labels'][cur_idx] == 1
        doc_idx = encodings_context['doc_indices'][cur_idx]
        en_idx = cur_idx + 1
        while en_idx < N and encodings_context['header_labels'][en_idx] == 0:
            en_idx += 1

        local_sent_id2num = {}
        sent_id_local = 0
        for k in range(cur_idx+1, en_idx):
            local_sent_id2num[sent_id_local] = sent_id2num[k]
            sent_id_local += 1
        
        result = {
            'head': encodings_context['segments'][cur_idx],
            'body': None,
            'segments': encodings_context['segments'][cur_idx+1:en_idx],
            'num_tokens_per_segment': local_sent_id2num,
            'scores': list(map(float, scores[cur_idx+1:en_idx]))
        }
        assert len(result['scores']), (len(infos), len(sim), cur_idx, en_idx)
        infos.append(result)
        cur_idx = en_idx
    return infos


def remove_by_th(infos, target_tokens, min_sents_per_text=2):
    items = []
    for info_idx, info in enumerate(infos):
        scale = 1 / max(info['scores'])
        for seg_id, (seg, s) in enumerate(zip(info['segments'], info['scores'])):
            items.append({
                'seg_id': seg_id,
                'info_id': info_idx,
                'score': s,
                'scale': scale,
                'num': info['num_tokens_per_segment'][seg_id]
            })
    items.sort(key=lambda i: i['score'] / i['scale'])

    num_total = sum(i['num'] for i in items)

    bad_indices = [[] for _ in infos]
    for item in items:
        if num_total <= target_tokens:
            break
        info_id = item['info_id']
        if len(infos[info_id]['segments']) - len(bad_indices[info_id]) <= min_sents_per_text:
            continue
        seg_id = item['seg_id']
        bad_indices[info_id].append(seg_id)
        num_total -= item['num']

    cleared_infos = []
    for info_idx, info in enumerate(infos):
        good_indices = set(range(len(info['segments']))) - set(bad_indices[info_idx])
        new_info = {
            'head': info['head'],
            'body': info['body'],
            'segments': [],
            'num_tokens_per_segment': {},
            'scores': []
            
        }
        global_idx = 0
        for i in range(len(info['segments'])):
            if i in good_indices:
                new_info['segments'].append(info['segments'][i])
                new_info['num_tokens_per_segment'][global_idx] = info['num_tokens_per_segment'][i]
                new_info['scores'].append(info['scores'][i])
                global_idx += 1
        cleared_infos.append(new_info)
    
    return items, cleared_infos


def process_multidoc_sample(
        sample, 
        question, 
        header_template, 
        target_tokens=2000, 
        compression_ratio=0.5, 
        min_sents_per_text=2, 
        max_context_length=6144,
        merge_strategy='separate'):
    assert (target_tokens == 0) or (compression_ratio == 0)
    
    borders = sample['inner_docs_start_indices'] + [sample['document_end_index']]
    docs = []
    for j in range(len(borders) - 1):
        st = borders[j]
        en = borders[j+1]
        docs.append(sample['input'][st:en])

    buckets = pack_docs_into_buckets(docs, max_context_length)

    bucket_infos = []
    for bucket in buckets:
        encodings_context, scores, sent_id2num = infer_bucket(
            bucket=bucket,
            question=question,
            header_template=header_template,
            max_context_len=max_context_length
        )
        infos = aggregate_infos(encodings_context, scores, sent_id2num)
        bucket_infos.append(infos)

    if target_tokens == 0:
        total = 0
        for infos in bucket_infos:
            for info in infos:
                for seg_id in range(len(info['segments'])):
                    total += info['num_tokens_per_segment'][seg_id]
        target_tokens = int(total * compression_ratio)

    infos_all = []
    for bucket_id, infos in enumerate(bucket_infos):
        for info in infos:
            info['bucket_id'] = bucket_id
        infos_all.extend(infos)
    
    if merge_strategy == 'separate':
        num_tokens_per_bucket = [
            len(tokenizer.encode(' '.join(b), add_special_tokens=False)) for b in buckets
        ]
        S = sum(num_tokens_per_bucket)
        budgets = [
            int(n * target_tokens / S) for n in num_tokens_per_bucket
        ]
        compressed_infos_all = []
        for tgt_n, infos in zip(budgets, bucket_infos):
            _, compressed_infos = remove_by_th(infos, tgt_n, min_sents_per_text=min_sents_per_text)
            compressed_infos_all.extend(compressed_infos)
    else:
        _, compressed_infos_all = remove_by_th(infos_all, target_tokens, min_sents_per_text=min_sents_per_text)
    

    chunks = []
    for info in compressed_infos_all:
        chunk = info['head'] + '\n'
        text = []
        for seg in info['segments']:
            text.append(seg)
        chunk += ' '.join(text)
        chunks.append(chunk)
    sample['compressed_context'] = '\n\n'.join(chunks)
    
    return infos_all, compressed_infos_all


def infer_single_docs(doc, question, max_context_len, target_tokens=2000, compression_ratio=0.5):
    assert (target_tokens == 0) != (compression_ratio == 0)
    sents = split_text_into_sentences(doc, language='en')
    sents = [s.strip() for s in sents if s.strip()]
    sents_buckets = pack_docs_into_buckets(sents, max_context_len)

    encodings_question = tokenizer.batch_encode_plus([question], add_special_tokens=False, padding='longest')

    all_sents = []
    all_scores = []
    num_tokens = []
    
    for sents_bucket in sents_buckets:
        encodings_context = pack_until_max_seq_len(tokenizer, sents_bucket, max_seq_len=max_context_len)
        scores, sent_id2num = get_scores(encodings_context, encodings_question)

        max_emb_id = max(sent_id2num.keys()) + 1

        for sent_id in range(max_emb_id):
            num_tokens.append(sent_id2num[sent_id])
        assert len(scores) == len(sents_bucket), (len(scores), len(sents_bucket))
        all_sents.extend(sents_bucket)
        all_scores.extend(
            list(map(float, scores))
        )

    total = 0
    for seg_id in range(len(all_sents)):
        total += num_tokens[seg_id]
    if target_tokens == 0:
        target_tokens = int(total * compression_ratio)

    items = list(zip(all_scores, list(range(len(all_scores))), num_tokens))
    items.sort()

    bad_indices = set()
    for item in items:
        if total <= target_tokens:
            break
        bad_indices.add(item[1])
        total -= item[2]

    filtered_sentences = []
    for sent_id, sent in enumerate(all_sents):
        if sent_id not in bad_indices:
            filtered_sentences.append(sent)
    txt = ' '.join(filtered_sentences)
    return filtered_sentences


def process_singledoc_sample(sample, question_for_sim=None, max_context_len=6144, target_tokens=2000, compression_ratio=0):
    q_st = sample['query_start_index']
    q_en = sample['query_end_index']
    
    d_st = sample['document_start_index']
    d_en = sample['document_end_index']
    
    if q_st != q_en:
        question = sample['input'][q_st:q_en]
        for opt in ['Question:', 'Query:', 'Question and Possible Answers:']:
            if question.startswith(opt):
                question = question[len(opt):].strip()
                break
    else:
        question = sample['input'].split('\n\n')[0]
    
    doc = sample['input'][d_st:d_en]

    if question_for_sim is None:
        question_for_sim = question
    
    sents_compressed = infer_single_docs(
        doc,
        question_for_sim,
        max_context_len=max_context_len,
        target_tokens=target_tokens,
        compression_ratio=compression_ratio,
    )

    return ' '.join(sents_compressed), question, doc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--tokenizer_name_or_path', required=True)
    parser.add_argument('--lora_bidirectional_name_or_path', required=True)
    parser.add_argument('--max_context_len', type=int, default=6144, required=False)
    parser.add_argument('--compression_factor', type=float, default=0, required=False)
    parser.add_argument('--compression_target_tokens', type=int, default=2000, required=False)
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        config_path=args.config_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        lora_name_or_path=args.lora_bidirectional_name_or_path,
    )

    dataset2meta = {
        'space_digest': {
            'question': 'Is this review positive or negative?',
            'final_template': '{question}\n\n{compressed_context}\n\nPercentage of Positive Reviews:\n',
            'header_prefix': 'Review ',
        },
        'book_sum_sort': {
            'question': 'What is happening in the text?',
            'final_template': '{question}\n\n{compressed_context}\n\nSummary IDs in Correct Order:\n\n',
            'header_prefix': 'Summary ',
        },
        'musique': {
            'final_template': 'You are given several paragraphs from Wikipedia and a question. Answer the question as concisely as you can, using a single phrase if possible. If the question cannot be answered based on the information in the paragraphs, write "unanswerable".\n\nParagraphs:\n{compressed_context}\n\nQuestion:\n{question}\n\nAnswer:\n',
            'header_prefix': 'Title: ',
        },
    }
    samples = []

    for dataset_name, meta in dataset2meta.items():
        ds = load_dataset("tau/zero_scrolls", dataset_name, split="validation")
        for i in tqdm(range(len(ds))):
            sample = ds[i]
            sample['idx'] = len(samples)

            q_st = sample['query_start_index']
            q_en = sample['query_end_index']
            d_st = sample['document_start_index']
            d_en = sample['document_end_index']
            if q_st != q_en:
                question_extracted = sample['input'][q_st:q_en]
                for opt in ['Question:', 'Query:', 'Question and Possible Answers:']:
                    if question_extracted.startswith(opt):
                        question = question[len(opt):].strip()
                        break
            else:
                question_extracted = sample['input'].split('\n\n')[0]

            question = meta.get('question', question_extracted)

            infos, compressed_infos = process_multidoc_sample(
                sample, 
                question, 
                meta['header_prefix'], 
                target_tokens=2000,
                compression_ratio=0
            )

            sample['question'] = question_extracted
            sample['context'] = sample['input'][d_st:d_en]
            sample['task'] = dataset_name
            sample['n_max_token_ans'] = 256
            sample['answer'] = sample['output']

            sample['compressed_prompt'] = meta['final_template'].format(question=sample['question'], compressed_context=sample['compressed_context'])
            
            samples.append(sample)

    dataset2meta = {
        'gov_report': {
            'final_template': '{question}\n\nReport:\n{compressed_context}\n\nSummary:\n',
        },
        'summ_screen_fd': {
            'final_template': '{question}\n\nReport:\n{compressed_context}\n\nSummary:\n',
        },
        'qmsum': {
            'final_template': 'You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{compressed_context}\n\nQuery:\n{question}\n\nAnswer:\n',
        },
        'squality': {
            'final_template': 'You are given a story and a question. Answer the question in a paragraph.\n\nStory:\n{compressed_context}\n\nQuestion:\n{question}\n\nAnswer:\n',
        },
        'qasper': {
            'final_template': 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n\nArticle:\n{compressed_context}\n\nQuestion:\n{question}\n\nAnswer:\n',
        },
        'narrative_qa': {
            'final_template': 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\nStory:\n{compressed_context}\n\nQuestion:\n{question}\n\nAnswer:\n',
        },
        'quality': {
            'final_template': 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\nStory:\n{compressed_context}\n\nQuestion:\n{question}\n\nAnswer:\n',
        },
    }

    for dataset_name, meta in dataset2meta.items():
        ds = load_dataset("tau/zero_scrolls", dataset_name, split="validation")
        for i in tqdm(range(len(ds))):
            sample = ds[i]
        
            sample['idx'] = len(samples)
        
            sample['compressed_context'], sample['question'], sample['context'] = process_singledoc_sample(sample)
            sample['task'] = dataset_name
            sample['n_max_token_ans'] = 256
            sample['answer'] = sample['output']
        
            sample['compressed_prompt'] = meta['final_template'].format(question=sample['question'], compressed_context=sample['compressed_context'])
            
            samples.append(sample)
    
    with open(args.save_path, 'w') as fout:
        json.dump(samples, fout)
