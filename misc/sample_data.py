import yaml
import argparse
import os
import jsonlines
from tqdm import tqdm
import random
import json

def parse():
    parser = argparse.ArgumentParser(description='IdealGPT in test datasets.')
    parser.add_argument('--save_root', type=str, default='./exp_data/', 
                        help='root path for saving results')
    parser.add_argument('--data_num', type=int, default=500,
                        help='number of data sampled')
    parser.add_argument('--dataset', type=str, default='vcr_val',
                        help='Names of the dataset to use in the experiment. Valid datasets include vcr_val. Default is vcr_val')
    args = parser.parse_args()
    return args

def Sample_VCR(dataset, data_num):
    dataset_path = '/home/haoxuan/data/vcr1/'
    if 'val' in dataset:
        dataset_anno_dir = os.path.join(dataset_path, 'val.jsonl')
    else:
        raise NotImplementedError(f'{dataset} Not Supported')

    all_anno_id = []
    with jsonlines.open(dataset_anno_dir) as reader:
        for cur_ann in tqdm(reader):
            all_anno_id.append(cur_ann['annot_id'])

    sampled_anno_id = random.sample(all_anno_id, data_num)
    return sampled_anno_id


args = parse()
random.seed(10)

if not os.path.exists(args.save_root):
    os.makedirs(args.save_root)

if 'vcr' in args.dataset:
    sampled_id = Sample_VCR(dataset=args.dataset, data_num=args.data_num)
print(f'Finish Sampling {args.dataset}; Obtained {len(sampled_id)} samples.')

if 'vcr' in args.dataset:
    result_path = os.path.join(args.save_root, f'{args.dataset}_random{args.data_num}_annoid.yaml')


with open(result_path, 'w') as f:
    yaml.dump(sampled_id, f)
print(f'Finish writing to {result_path}')
