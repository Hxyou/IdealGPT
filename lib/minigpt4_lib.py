from .minigpt4_config import Config

# Add MiniGPT-4 project directory into python path to import minigpt4
import sys
sys.path.append('/home/haoxuan/code/MiniGPT-4')

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

class DictToClass:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class MINIGPT4():
    def __init__(self, cfg_path='/home/haoxuan/code/MiniGPT-4/eval_configs/minigpt4_eval.yaml', gpu_id=0, 
                 options=None, temperature=0.001):
        args = {'cfg_path':cfg_path, 'gpu_id':gpu_id, 'options':options}
        args = DictToClass(**args)
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print('Initialization of MiniGPT4 Finished')

        self.chat = chat
        self.temperature = temperature
        self.model_type = 'minigpt4'

    def ask(self, img_path, text):
        chat_state = CONV_VISION.copy()
        img_list = []

        # Start extracting image feature
        _ = self.chat.upload_img(img_path, chat_state, img_list)

        # Asking questions
        self.chat.ask(text, chat_state)

        # Generating Answer
        llm_message = self.chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=1,
                                    temperature=self.temperature,
                                    max_new_tokens=300,
                                    max_length=2000)[0]
        return llm_message

    def caption(self, img_path):
        return self.ask(img_path=img_path, text='Give a clear and concise summary of the image below in one paragraph.')
         