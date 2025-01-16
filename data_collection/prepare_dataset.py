import sys
import os
import openai
import tiktoken
import torch
import json
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import random
from datasets import load_dataset

from data_collection.common import LLMPipeline, make_phi3_qa_prompt, sentence_is_good

from util.util import (
    get_ppl_one_step,
    kl_divergence,
    tokenize_and_clip_segments,
    split_text_into_sentences_keep_slashn,
)


def main(args):
    llmpipe = LLMPipeline(
        args.model_type,
        args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(args.candidate_model_name)

    with open(args.prompts_file) as fin:
        prompt_templates = json.load(fin)
    
    results = []

    dataset = load_dataset("EleutherAI/wikitext_document_level", 'wikitext-103-v1', trust_remote_code=True)

    dataset = dataset.shuffle(seed=43)

    pbar = tqdm(range(len(dataset['train']['page'])))

    yes_num = 0
    total_num = 0

    for index in pbar:
        sents = split_text_into_sentences_keep_slashn(text=dataset['train']['page'][index], language='en')
        sents = [t for t in sents if t.strip()]

        _text = ' '.join(sents)
        
        if args.skip_long_contexts and len(tokenizer.encode(_text)) > args.max_context_length:
            continue

        encodings = tokenize_and_clip_segments(
            tokenizer=tokenizer,
            segments=sents,
            segments_labels=[0 for _ in sents],
            max_seq_len=args.max_context_length
        )

        good_sents_indices = [
            i for i, s in enumerate(encodings['segments']) if sentence_is_good(s)
        ]

        if len(good_sents_indices) == 0:
            print('No suitable sents found. Skip this example.')
            continue

        idx = random.choice(good_sents_indices)

        text = ' '.join(encodings['segments'][:idx]) + ' [[' + encodings['segments'][idx] + ']] ' + ' '.join(encodings['segments'][idx+1:])

        prompt = prompt_templates['qa_generation_prompt_template'].format(text=text, sentence=encodings['segments'][idx])
        
        qa_text = llmpipe.query(
            prompt,
        )

        encodings.update({
            'sent': encodings['segments'][idx],
            'page': dataset['train']['page'][index],
            'qa_text': qa_text,
            'sent_idx': idx,
            'qa_generation_prompt': prompt,
            'estimator': args.model_name_or_path,
        })
        
        ### Verification
        qa_pairs_raw = encodings['qa_text'].split('\n\n')
        good = True
        for p in qa_pairs_raw:
            try:
                q, a = p.split('\n')
            except:
                good = False
                continue
            if not q.startswith('Q: ') or not a.startswith('A: '):
                good = False
        if not good:
            print('No qa pairs found!')
            continue
        
        qa_pairs_raw = encodings['qa_text'].split('\n\n')

        qa_pairs = []
        for p in qa_pairs_raw:
            q, a = p.split('\n')
            assert q.startswith('Q: ')
            assert a.startswith('A: ')
            q = q[len('Q: '):]
            a = a[len('A: '):]
            qa_pairs.append(
                {
                    'question': q,
                    'answer': a,
                }
            )

        encodings['qa_verification_results'] = []

        for qa_pair in qa_pairs:
            if 'sentence' in qa_pair['question']: # skip questions that directly ask something about the sentence - they are usually bad
                continue

            prompt = prompt_templates['verification_prompt_template'].format(
                sent=encodings['sent'],
                **qa_pair
            )
            verification_result = llmpipe.query(
                prompt,
            )

            is_yes = verification_result.split('\n')[-1].lower().strip().rstrip('.').endswith('yes')

            yes_num += int(is_yes)
            total_num += 1

            encodings['qa_verification_results'].append({
                'qa': qa_pair,
                'result_text': verification_result,
                'is_yes': is_yes,
            })

        results.append(encodings)
        
        ### Save
        if (index + 1) % 100 == 0:
            with open(args.qa_generation_dst_file, 'w') as fout:
                json.dump(results, fout)

        pbar.set_postfix(
        {
            'n': len(results),
            'good_ratio': (total_num - yes_num) / total_num if total_num else 0
        })

        if len(results) >= args.max_examples:
            break
                
    with open(args.qa_generation_dst_file, 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_type', default='openai')
    parser.add_argument('--model_name_or_path', default='gpt-3.5-turbo-0125')
    parser.add_argument('--candidate_model_name', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--prompts_file', help='file with json with `qa_generation_prompt_template` and `verification_prompt_template`')
    parser.add_argument('--qa_generation_dst_file', required=True)
    parser.add_argument('--target_dataset_root', 
                        help='path where the finally prepared dataset will be stored in format {root}/train.json {root}/val.json', required=True)
    parser.add_argument('--openai_api_token', default=None)
    parser.add_argument('--max_context_length', default=6144, type=int)
    parser.add_argument('--skip_long_contexts', action='store_true', default=False)
    parser.add_argument('--max_examples', type=int, default=32000)
    args = parser.parse_args()

    if args.openai_api_token is not None:
        openai.api_key = args.openai_api_token

    main(args)
