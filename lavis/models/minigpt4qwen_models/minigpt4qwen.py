"""
Requires Transformer 4.32 and above, implementation may change according the Llama implementation
"""
import os
import logging
import string
from packaging import version

from omegaconf import OmegaConf

import contextlib

import torch
# torch.autograd.set_detect_anomaly(True) # for debug
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.minigpt4qwen_models.blip2 import Blip2Base, disabled_train

from functools import partial
import re
from copy import deepcopy

from .chat_utils import get_stop_words_ids, make_context, decode_tokens

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""

@registry.register_model("minigpt4qwen")
class Minigpt4Qwen(Blip2Base):
    """
    BLIP2 + Projection + Qwen7B-chat = Minigpt4Qwen model.
    Supported model types:
        - qwen7b_chat
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("minigpt4qwen", "qwen7b_chat")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "qwen7b_chat": "configs/models/minigpt4qwen/minigpt4qwen.yaml",
        "qwen14b_chat": "configs/models/minigpt4qwen/minigpt4qwen-14b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        llm_model="",
        max_txt_len=512,
        apply_lemmatizer=False,
        qformer_text_input=True,
        get_lora=False,
        lora_alpha=32,
        lora_r=8,
        lora_dropout=0.05,
        unfreeze_pos_embed=False,
        freeze_qformer=False,
        freeze_queries=False,
        freeze_proj=False,
        enable_autocast=True,
        freeze_llm=True,
        llm_device_map="cpu"
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.32"), "Minigpt4Qwen requires transformers>=4.32"
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        
        # self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

            if unfreeze_pos_embed:
                self.visual_encoder.pos_embed.requires_grad_(True)

        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        if not qformer_text_input:
            logging.info("no text input for q-former")
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            raise NotImplementedError
        self.Qformer.cls = None

        if freeze_qformer:
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            for _,param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
        
        if freeze_queries:
            # nn.Parameter class
            self.query_tokens.requires_grad = False

        print(f'Loading LLM:{llm_model}...')
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            cache_dir=registry.get_path("cache_root"),
            model_max_length=max_txt_len,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        
        llm_config = AutoConfig.from_pretrained(llm_model,cache_dir=registry.get_path("cache_root"),trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            config=llm_config,
            cache_dir=registry.get_path("cache_root"),
            trust_remote_code=True,
            device_map=llm_device_map,
        )
        # self.llm_model.transformer.gradient_checkpointing = True # 错误用法：打开llm的gradient checkpointing 
        self.llm_model.gradient_checkpointing_enable() # 正确用法：打开llm的gradient checkpointing 会定义_gradient_checkpointing_func的！！！

        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id
        self.replace_image_token_id = self.llm_tokenizer("<|extra_0|>").input_ids[0]
        self.replace_image_string = '<|extra_0|>'
        # self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        self.freeze_llm = freeze_llm
        if self.freeze_llm:
            print("Freeze LLM...")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            print("Unfreeze LLM!!!")
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        if freeze_proj:
            for name,param in self.llm_proj.named_parameters():
                param.requires_grad = False
            self.llm_proj = self.llm_proj.eval()
            self.llm_proj.train = disabled_train

        self.max_txt_len = max_txt_len

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

        # lora configuration
        self.get_lora = get_lora
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.lora_dropout = lora_dropout

        if self.get_lora:
            peft_config = LoraConfig(
                target_modules=['q_proj','v_proj'],
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model,peft_config)
            self.llm_model.print_trainable_parameters()
        
        # enable autocast
        self.enable_autocast = enable_autocast

    def encode_image(self, image):
        with (self.maybe_autocast() if self.enable_autocast else contextlib.nullcontext()):
            image_embeds = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_embeds)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input:
            raise NotImplementedError
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        return inputs_llm

    def preprocess(
            self,
            sources,
            tokenizer: transformers.PreTrainedTokenizer,
            max_len: int,
            image_len: int = 32,
            system_message: str = "You are a helpful assistant."
        ):
            IGNORE_TOKEN_ID = -100
            roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

            im_start = tokenizer.im_start_id
            im_end = tokenizer.im_end_id
            nl_tokens = tokenizer('\n').input_ids
            _system = tokenizer('system').input_ids + nl_tokens
            _user = tokenizer('user').input_ids + nl_tokens
            _assistant = tokenizer('assistant').input_ids + nl_tokens

            # Apply prompt templates
            input_ids, targets = [], []
            for i, source in enumerate(sources):
                img_visit_cnt = 0
                if roles[source[0]["from"]] != roles["user"]:
                    source = source[1:]

                input_id, target = [], []
                system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
                input_id += system
                target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
                assert len(input_id) == len(target)
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    content = sentence["value"]
                    if self.replace_image_string in content:
                        content.replace(self.replace_image_string,"")

                    if "<ImageHere>" in content and role == '<|im_start|>user':
                        # img_visit_cnt += 1
                        # assert len(content.split("<ImageHere>")) == 2, 'Only support one image in one sentence'
                        # c_before, c_after = content.split("<ImageHere>")
                        # _input_id = tokenizer(role).input_ids + nl_tokens + \
                        #     tokenizer(c_before).input_ids + [self.replace_image_token_id] * image_len + tokenizer(c_after).input_ids + [im_end] + nl_tokens

                        # 支持多图/视频输入
                        img_visit_cnt += content.count("<ImageHere>")
                        content = content.replace("<ImageHere>", self.replace_image_string * image_len)
                        _input_id = tokenizer(role).input_ids + nl_tokens + \
                                tokenizer(content).input_ids + [im_end] + nl_tokens
                    else:
                        _input_id = tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(content).input_ids + [im_end] + nl_tokens
                    input_id += _input_id
                    if role == '<|im_start|>user':
                        _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
                    elif role == '<|im_start|>assistant':
                        _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                            _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
                    else:
                        raise NotImplementedError
                    target += _target
                # assert img_visit_cnt == 1, f'Only support one image in conversations and must be at the first sentence, but get {img_visit_cnt} visits'
                assert len(input_id) == len(target), "input_ids should have the same length as the target"
                input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
                target += [IGNORE_TOKEN_ID] * (max_len - len(target))
                input_ids.append(input_id[:max_len])
                targets.append(target[:max_len])
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')

        image = samples["image"]
        inputs_llm = self.encode_image(image)

        sources = samples["conversations"]
        data_dict = self.preprocess(sources,self.llm_tokenizer,self.max_txt_len,image_len=self.num_query_token)
        device = self.llm_model.device
        llm_tokens = data_dict['input_ids'].to(device)
        targets = data_dict['labels'].to(device)
        attention_mask = data_dict['attention_mask'].to(device)


        replace_image_idxs = torch.where(llm_tokens == self.replace_image_token_id)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens) # B, L, C
        _,_,channels = inputs_embeds.shape
        inputs_embeds[replace_image_idxs[0],replace_image_idxs[1]] = inputs_llm.view(-1,channels).to(inputs_embeds.dtype)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        chat=False,
        generation_config=None,
        stop_words_ids=None,
        return_dict_in_generate=False,
        **kwargs
    ):
        generation_config = generation_config if generation_config is not None else self.llm_model.generation_config

        self.llm_tokenizer.padding_side = 'left'
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id

        image = deepcopy(samples["image"])
        text = deepcopy(samples['text'])

        if isinstance(text, str):
            text = [text]
            bs = 1
        elif isinstance(text,list):
            # assert len(text) == bs, "The number of texts must be equal to the batch size."
            bs = len(text)
        else:
            raise TypeError

        if self.qformer_text_input:
            raise NotImplementedError

        # For video data
        if image.dim() == 5:
            assert False, 'the dim of image is 5, but now we don\'t support 5D images/video input'
        elif image.dim() == 4:
            inputs_llm = self.encode_image(image)
        else:
            assert False,f'the dim of image is {image.dim()}, we only support image input with a shape [B,C,H,W].'

        for i in range(bs):
            # assert len(text[i].split('<ImageHere>')) == 2, f'must be one and only image !,now split_length = {len(text[i].split("<ImageHere>"))}'
            image_num = text[i].count("<ImageHere>")
            if image_num:
                print(f"In Batch_{i} Query: {image_num} images!")
            replace_string = ''.join([self.replace_image_string] * self.num_query_token)
            if self.replace_image_string in text[i]:
                text[i].replace(self.replace_image_string,"")
            text[i] = text[i].replace('<ImageHere>',replace_string)

        llm_tokens = self.llm_tokenizer(text, return_tensors='pt', padding='longest')
        attention_mask = llm_tokens.attention_mask.to(image.device)
        llm_tokens.input_ids = llm_tokens.input_ids.to(image.device)

        replace_image_idxs = torch.where(llm_tokens.input_ids == self.replace_image_token_id)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids) # B, L, C
        _,_,channels = inputs_embeds.shape
        inputs_embeds[replace_image_idxs[0],replace_image_idxs[1]] = inputs_llm.view(-1,channels).to(inputs_embeds.dtype)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            pad_token_id=self.llm_tokenizer.eod_id,
            bos_token_id=self.llm_tokenizer(' ').input_ids[0], # 我发现规定inputs_embeds，指定bos_token_id好像没用？
            eos_token_id=[self.llm_tokenizer.im_end_id,self.llm_tokenizer.im_start_id],
        )
        if not chat:
            output_text = [
                self.llm_tokenizer.decode(_[:].cpu(),skip_special_tokens=True).strip() for _ in outputs
            ]
            return output_text
        else:
            return outputs
    
    def chat(
        self,
        query,
        history,
        image_tensor,
        system = "You are a helpful assistant.",
        append_history = True,
        stream = _SENTINEL,
        stop_words_ids = None,
        generation_config = None,
        **kwargs,
    ):
        generation_config = generation_config if generation_config is not None else self.llm_model.generation_config

        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            self.llm_tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        sample = {
            "image": image_tensor,
            "text": raw_text
        }

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, self.llm_tokenizer
        ))
        outputs = self.generate(
                    sample,
                    chat=True,
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    **kwargs,
                )

        response = decode_tokens(
            outputs[0],
            self.llm_tokenizer,
            chat_format=generation_config.chat_format,
            verbose=kwargs.pop('verbose',False),
            errors='replace'
        )

        if append_history:
            history.append((query, response))

        return response, history

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_pretrained(cls, model_type, llm_device_map="cpu"):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model_cfg['llm_device_map'] = llm_device_map
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def from_config(cls, cfg):
        # text config
        max_txt_len = cfg.get("max_txt_len", 512)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        # llm config
        llm_model = cfg.get("llm_model")
        
        if not os.path.isabs(llm_model):
            llm_model = os.path.join(registry.get_path('cache_root'),llm_model)

        # lora config
        get_lora = cfg.get("get_lora",False)
        lora_alpha = cfg.get("lora_alpha",32)
        lora_r = cfg.get("lora_r",8)
        lora_dropout = cfg.get("lora_dropout",0.05)

        # vision encoder config
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        unfreeze_pos_embed = cfg.get("unfreeze_pos_embed",False)
        if freeze_vit == False and unfreeze_pos_embed == False:
            print('unfreeze vit so it will unfreeze pos embed')

        # q-former config
        num_query_token = cfg.get("num_query_token")
        qformer_text_input = cfg.get("qformer_text_input", True)
        freeze_qformer = cfg.get("freeze_qformer",False)
        freeze_queries = cfg.get("freeze_queries",False)

        # proj config
        freeze_proj = cfg.get("freeze_proj",False)
        
        # autocast config
        enable_autocast = cfg.get("enable_autocast",True)

        # freeze llm
        freeze_llm = cfg.get("freeze_llm",True)

        llm_device_map = cfg.get("llm_device_map", "cpu")
        assert llm_device_map in ['cpu', 'auto'], 'please set `llm_device_map` in [`cpu`,`auto`] if training or single-gpu inference, set `cpu`. if multi-gpu inference, set `auto`'

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
            get_lora=get_lora,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            lora_dropout=lora_dropout,
            unfreeze_pos_embed=unfreeze_pos_embed,
            freeze_qformer=freeze_qformer,
            freeze_queries=freeze_queries,
            freeze_proj=freeze_proj,
            enable_autocast=enable_autocast,
            freeze_llm=freeze_llm,
            llm_device_map=llm_device_map,
        )

        model.load_checkpoint_from_config(cfg)

        return model
