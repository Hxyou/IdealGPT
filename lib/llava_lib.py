import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


class LLAVA():
    def __init__(self, model_path="/home/rui/code/LLaVA/LLaVA-13B-v0", mm_projector_setting=None, 
                 vision_tower_setting=None, conv_mode='simple', temperature=0.001):
        disable_torch_init()
        model_name = os.path.expanduser(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if mm_projector_setting is None:
            patch_config(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            vision_config = vision_tower.config
            vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
            vision_config.use_im_start_end = mm_use_im_start_end
            if mm_use_im_start_end:
                vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
            image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        else:
            # in case of using a pretrained model with only a MLP projector weights
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

            vision_tower = CLIPVisionModel.from_pretrained(vision_tower_setting, torch_dtype=torch.float16).cuda()
            image_processor = CLIPImageProcessor.from_pretrained(vision_tower_setting, torch_dtype=torch.float16)

            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            vision_config = vision_tower.config
            vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
            vision_config.use_im_start_end = mm_use_im_start_end
            if mm_use_im_start_end:
                vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

            mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
            mm_projector_weights = torch.load(mm_projector_setting, map_location='cpu')
            mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

            model.model.mm_projector = mm_projector.cuda().half()
            model.model.vision_tower = [vision_tower]

        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_token_len = image_token_len
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model
        self.temperature = temperature
        self.model_type = 'llava'

    def decode_output_text(self, output_ids, input_ids, conv):
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # pdb.set_trace()
        full_history = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        if self.conv_mode == 'simple_legacy' or self.conv_mode == 'simple':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs_response = outputs[:index].strip()
        # Return both response and full_chat history.
        return outputs_response, full_history  

    def ask(self, img_path, text):
        qs = text
        if self.mm_use_im_start_end:
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

        if self.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])
    

        image = Image.open(img_path)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        outputs, full_chat = self.decode_output_text(output_ids, input_ids, conv=conv)
        return outputs

    def caption(self, img_path):
        return self.ask(img_path=img_path, text='Give a clear and concise summary of the image below in one paragraph.')
         
    
if __name__ == "__main__":
    llava_model = LLAVA()
    print('successfully initialized llava model')