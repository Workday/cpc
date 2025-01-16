import torch
import json
import random
from argparse import ArgumentParser
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from util.util import (
    get_ppl_one_step,
    kl_divergence,
)

from data_collection.common import LLMPipeline, make_phi3_qa_prompt
from copy import deepcopy


@torch.no_grad()
def main(args, llmpipe, text_embedder):
    with open(args.qa_generations_file) as fin:
        qa_samples = json.load(fin)
    
    for sample in tqdm(qa_samples):
        for qa_pair in sample['qa_verification_results']:
            embeddings = text_embedder.encode(sample['segments'])
            q_emb = text_embedder.encode([qa_pair['qa']['question']])
            
            sims = embeddings @ q_emb.T
            sims = sims[:, 0]
            order = list(map(int, sims.argsort()))
            
            pivot_pos = order.index(sample['sent_idx'])
            
            possible_bad_examples = order[:pivot_pos]

            if pivot_pos / (len(order) - 1) < 0.3:
                print('similarity of question and positive sentence is too low. skip this question', pivot_pos / (len(order) - 1))
                continue
            
            qa_pair.update({
                'negative_examples': possible_bad_examples,
                'negative_examples_scores': list(map(float, sims[possible_bad_examples])),
            })
    
    flattened_samples = []

    for sample in tqdm(qa_samples):
        for qa_pair in sample['qa_verification_results']:
            if 'negative_examples' not in qa_pair or qa_pair['is_yes']:
                continue
            item = {}
            item['page'] = sample['page']
            item['segments'] = sample['segments']
            item['pos_sent_idx'] = sample['sent_idx']
            item['estimator'] = sample['estimator']
            item['question'] = qa_pair['qa']['question']
            item['answer'] = qa_pair['qa']['answer']
            item['negative_examples'] = qa_pair['negative_examples']
            item['negative_examples_scores'] = qa_pair['negative_examples_scores']
            flattened_samples.append(item)


    for sample in tqdm(flattened_samples):
        sents = sample['segments']
        question = sample['question']
        answer = sample['answer']

        if len(sample['negative_examples']) < args.num_negatives_per_positive:
            print('Not enough negatives found. Skip.')
            continue

        if args.max_kl_diff_for_negative > 0:
            p = make_phi3_qa_prompt(sents, question)
            
            init_data = get_ppl_one_step(
                llmpipe.model,
                llmpipe.tokenizer,
                p, 
                answer
            )

            negative_examples = deepcopy(sample['negative_examples'])

            random.shuffle(negative_examples)

            neg_samples = []
            for n_idx in negative_examples:
                sents_ = sents[:n_idx] + sents[n_idx + 1:]
                p = make_phi3_qa_prompt(sents_, question)
                data = get_ppl_one_step(
                    llmpipe.model,
                    llmpipe.tokenizer,
                    p,
                    answer
                )
                kl_ = kl_divergence(
                    init_data['logprobs'],
                    data['logprobs'],
                )
                if kl_.item() > args.max_kl_diff_for_negative:
                    continue
                neg_samples.append(n_idx)
                if len(neg_samples) >= args.num_negatives_per_positive:
                    break
            sample['neg_samples'] = neg_samples
        else:
            sample['neg_samples'] = list(random.sample(sample['negative_examples'], args.num_negatives_per_positive))

    with open(args.dst_file, 'w') as fout:
        json.dump(flattened_samples, fout)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--qa_generations_file',
        help='File with prefixes, positives, question and answers to positives&context that was obtained by prepare_dataset_v2.py'
    )
    parser.add_argument(
        '--dst_file',
        help='Dst file'
    )
    parser.add_argument('--num_negatives_per_positive', type=int, default=2)
    parser.add_argument('--max_kl_diff_for_negative', type=float, default=0.003)
    args = parser.parse_args()

    text_embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    llmpipe = LLMPipeline(
        'hf',
        'microsoft/Phi-3-small-128k-instruct'
    )

    main(args, llmpipe, text_embedder)
