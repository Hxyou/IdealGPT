from tqdm import tqdm
import os
import yaml
import pdb
import argparse
import re
import pdb

def parse():
    parser = argparse.ArgumentParser(description='IdealGPT in test datasets.')
    parser.add_argument('--result', type=str, default='/home/haoxuan/code/MiniGPT-4/result/qa/ve/promptv1_temp0.001_ve_dev_random500_pairid', 
                        help='VCR predicted result path')
    parser.add_argument('--print_multiround', action='store_true', 
                        help='Whether print the annoid of multiple rounds and correct samples.')
    parser.add_argument('--print_outlier', action='store_true', 
                        help='Whether print the annoid of multiple rounds and correct samples.')
    args = parser.parse_args()
    return args


args = parse()
blip_gpt_folder = args.result



answer_map_counter = {'entailment':0, 'contradiction':0, 'neutral':0, 'outlier':0}
total_correct_count = 0
total_unknown_count = 0
total_count = 0
total_round = 0
total_tokens = 0
total_outlier_count = 0
count_round = False

for i in tqdm(os.listdir(blip_gpt_folder), desc='Analyzing BLIP_GPT result ({})'.format(blip_gpt_folder)):
    cur_dir = os.path.join(blip_gpt_folder, i)
    with open(cur_dir, "r") as file:
        data_i = yaml.safe_load(file)

    if 'used_round' in data_i['result']:
        count_round = True
        total_round += data_i['result']['used_round']
        total_tokens += data_i['result']['total_tokens']

    if data_i['result']['predict_answer'] is None:
        total_unknown_count += 1
        if args.print_outlier:
            print('Samples with unknown answer prediction:{}'.format(data_i['setting']['annot_id']))
    else:
        data_i['result']['predict_answer'] = str(data_i['result']['predict_answer']).strip().lower()
        if data_i['result']['predict_answer'] == str(data_i['setting']['answer_label']).strip():
            total_correct_count += 1
            if args.print_multiround and 'used_round' in data_i['result']:
                if data_i['result']['used_round'] > 1:
                    print('Samples with multiple rounds and correct:{}'.format(data_i['setting']['annot_id']))
        if data_i['result']['predict_answer'] in answer_map_counter:
            answer_map_counter[data_i['result']['predict_answer']] += 1
        else:
            answer_map_counter['outlier'] += 1
            if args.print_outlier:
                print('Samples with outlier answer prediction:{}'.format(data_i['setting']['annot_id']))            


        # elif str(data_i['result']['predict_answer']).strip() not in answer_map_counter:
        #     total_outlier_count += 1
        #     if args.print_outlier:
        #         print('Samples with outlier answer prediction:{}'.format(data_i['setting']['id']))

    total_count += 1

answer_map_counter = {k:v*1.0/(total_count-total_unknown_count) for k,v in answer_map_counter.items()} 

print('Accuracy of {} samples: {}%'.format(total_count, (total_correct_count*1.0*100/total_count)))
print('Ratio of unknown samples: {}%'.format(total_unknown_count*1.0*100/total_count))
print('Distribution of answers: {}%'.format(answer_map_counter))

if count_round:
    print('Average rounds used: {}'.format(total_round*1.0/total_count))
    print('Average tokens used: {}'.format(total_tokens*1.0/total_count))