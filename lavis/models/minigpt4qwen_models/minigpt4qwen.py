"""
Requires Transformer 4.32 and above, implementation may change according the Llama implementation
"""
import os
import logging
import string
from packaging import version

import torch
# torch.autograd.set_detect_anomaly(True) # for debug
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers
from peft import LoraConfig, get_peft_model

from lavis.common.registry import registry
from lavis.models.minigpt4qwen_models.blip2 import Blip2Base, disabled_train

from functools import partial

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
        autocast_dtype=torch.float16,
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
            device_map='cuda',
        )

        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id
        self.replace_image_token_id = self.llm_tokenizer("<|extra_0|>").input_ids[0]
        self.replace_image_string = '<|extra_0|>'
        # self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

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
        
        self.autocast_dtype = autocast_dtype

    def encode_image(self, image):
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        bs = image.size(0)

        with self.maybe_autocast(self.autocast_dtype):
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
    
    @staticmethod
    def preprocess(
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
                    if "<ImageHere>" in content and role == '<|im_start|>user':
                        img_visit_cnt += 1
                        assert len(content.split("<ImageHere>")) == 2, 'Only support one image in one sentence'
                        c_before, c_after = content.split("<ImageHere>")
                        _input_id = tokenizer(role).input_ids + nl_tokens + \
                            tokenizer(c_before).input_ids + [self.replace_image_token_id] * image_len + tokenizer(c_after).input_ids + [im_end] + nl_tokens
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
                assert img_visit_cnt == 1, f'Only support one image in conversations and must be at the first sentence, but get {img_visit_cnt} visits'
                assert len(input_id) == len(target)
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


        with self.maybe_autocast(self.autocast_dtype):
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
        use_nucleus_sampling=False,
        num_beams=5,
        max_new_tokens=100,
        min_new_tokens=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = 'left'
        self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eod_id

        image = samples["image"]
        text = samples['text']
        bs = image.size(0)

        if bs > 1 and num_captions > 1:
            assert False, 'if bs > 1, num_captions must be equal to 1 now.'
        if isinstance(text, str):
            text = [text] * bs
        else:
            assert len(text) == bs, "The number of texts must be equal to the batch size."

        if self.qformer_text_input:
            raise NotImplementedError

        # For video data
        if image.dim() == 5:
            assert False, 'the dim of image is 5, but now we don\'t support video input'
        elif image.dim() == 4:
            inputs_llm = self.encode_image(image)
        else:
            assert False,f'the dim of image is {image.dim()}, we only support image input with a shape [B,C,H,W].'

        for i in range(bs):
            assert len(text[i].split('<ImageHere>')) == 2, 'must be one and only image !'
            replace_string = ''.join([self.replace_image_string] * self.num_query_token)
            text[i] = text[i].replace('<ImageHere>',replace_string)

        llm_tokens = self.llm_tokenizer(text, return_tensors='pt', padding='longest')
        attention_mask = llm_tokens.attention_mask.to(image.device)
        llm_tokens.input_ids = llm_tokens.input_ids.to(image.device)

        with self.maybe_autocast(self.autocast_dtype):
            replace_image_idxs = torch.where(llm_tokens.input_ids == self.replace_image_token_id)
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids) # B, L, C
            _,_,channels = inputs_embeds.shape
            inputs_embeds[replace_image_idxs[0],replace_image_idxs[1]] = inputs_llm.view(-1,channels).to(inputs_embeds.dtype)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                output_hidden_states=True,
                use_cache=True,
                num_return_sequences=num_captions,
                pad_token_id=self.llm_tokenizer.eod_id,
                bos_token_id=self.llm_tokenizer('\n').input_ids[0],
                eos_token_id=self.llm_tokenizer.im_end_id,
            )
        output_text = [
            self.llm_tokenizer.decode(_[:].cpu(),skip_special_tokens=True).strip() for _ in outputs
        ]

        return output_text

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

        # cast dtype
        autocast_dtype = cfg.get("autocast_dtype","float16")
        autocast_dict = {"float16":torch.float16,"bfloat16":torch.bfloat16}
        autocast_dtype = autocast_dict[autocast_dtype]

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
            autocast_dtype=autocast_dtype,
        )

        model.load_checkpoint_from_config(cfg)

        return model
