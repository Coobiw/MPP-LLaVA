import os
import os.path as osp

import contextlib

import torch
import torch.nn as nn

import transformers

from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

from .minigpt4qwen import Minigpt4Qwen

def enable_input_require_grads(module):
    def make_inputs_require_grads(module,input,output):
        output.requires_grad_(True)

    module.register_forward_hook(make_inputs_require_grads)

class VisionPipe(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.visual_encoder = model.visual_encoder
        self.ln_vision = model.ln_vision
        self.Qformer, self.query_tokens = model.Qformer, model.query_tokens

        self.maybe_autocast = model.maybe_autocast()
        self.enable_autocast = model.enable_autocast

        self.llm_proj = model.llm_proj

    def forward(self,ipt):
        image = ipt
        with (self.maybe_autocast if self.enable_autocast else contextlib.nullcontext()):
            image_embeds = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_embeds)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        bs = image.size(0)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        return inputs_llm

class EmbeddingPipeLayer(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.word_embeddings = model.llm_model.transformer.wte
        # enable_input_require_grads(self.word_embeddings)

    def forward(self, ipt):
        llm_tokens = ipt.long()
        return self.word_embeddings(llm_tokens)

class TokenizerPipeLayer(nn.Module):
    def __init__(self, model:Minigpt4Qwen):
        super().__init__()
        self.replace_image_token_id = model.replace_image_token_id

        self.visionpipe = VisionPipe(model)
        self.wtepipe = EmbeddingPipeLayer(model)

        self.drop = model.llm_model.transformer.drop

        self.config = model.llm_model.transformer.config
        self.use_dynamic_ntk = model.llm_model.transformer.use_dynamic_ntk
        self.llm_training = model.llm_model.transformer.training

        # rope + ntk
        self.rotary_emb = model.llm_model.transformer.rotary_emb

        # rope+ntk related func
        self.get_ntk_alpha = model.llm_model.transformer.get_ntk_alpha
        # self.get_head_mask = model.llm_model.transformer.get_head_mask

    def forward(self,ipt):
        image, llm_tokens, targets, attention_mask = ipt
        inputs_llm = self.visionpipe(image)

        device = inputs_llm.device

        replace_image_idxs = torch.where(llm_tokens == self.replace_image_token_id)
        inputs_embeds = self.wtepipe(llm_tokens) # B, L, C
        _,_,channels = inputs_embeds.shape

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[replace_image_idxs[0],replace_image_idxs[1]] = inputs_llm.view(-1,channels).to(inputs_embeds.dtype)

        # rope + ntk
        # get rotary_pos_emb_list
        input_shape = inputs_embeds.size()[:-1]
        position_ids = torch.arange(
                0,
                input_shape[-1],
                dtype=torch.long,
                device=device,
            )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        kv_seq_len = inputs_embeds.size()[1]
        if self.llm_training or not self.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        else:
            ntk_alpha_list = []
            ntk_alpha = self.get_ntk_alpha(kv_seq_len)
            ntk_alpha_list.append(ntk_alpha)
        self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        ntk_alpha = ntk_alpha_list[0]
        rotary_pos_emb_list = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha)
        rotary_pos_emb_list = torch.stack(rotary_pos_emb_list,dim=0)
        # print(rotary_pos_emb_list);exit(0)

        inputs_embeds = self.drop(inputs_embeds)
        output_shape = input_shape + (inputs_embeds.size(-1),)
        output_shape = torch.tensor(output_shape,device="cuda")

        batch_size = inputs_embeds.shape[0]
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.wtepipe.word_embeddings.weight.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.wtepipe.word_embeddings.weight.dtype).min

        rotary_pos_emb_list.requires_grad_(True)
        attention_mask.requires_grad_(True)

        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape

class QwenBlockPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen, layer_idx):
        super().__init__()
        self.layer = model.llm_model.transformer.h[layer_idx]
        self.layer_idx = layer_idx

    def forward(self, ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        # print("grad: ", inputs_embeds.requires_grad)
        inputs_embeds = self.layer(inputs_embeds, rotary_pos_emb_list=[[rotary_pos_emb_list[0],rotary_pos_emb_list[1]]],
                    attention_mask=attention_mask,
                    head_mask=None)[0]
        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape


class FLNPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.final_layernorm = model.llm_model.transformer.ln_f

    def forward(self, ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        inputs_embeds = self.final_layernorm(inputs_embeds)
        inputs_embeds = inputs_embeds.view(list(output_shape)).contiguous()
        # print(inputs_embeds)
        return inputs_embeds, targets


class LMPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.lm_head = model.llm_model.lm_head

    def forward(self, ipt):
        hidden_states, labels = ipt
        logits = self.lm_head(hidden_states)
        # print(logits)
        return logits, labels

class LossPipeLayer(torch.nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.freeze_llm = model.freeze_llm

    def forward(self, ipt):
        logits, labels = ipt
        # print(logits.size());print(labels.size());exit(0)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        bs = shift_labels.size(0)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # print(loss)
        return (loss, torch.tensor(bs)) if self.freeze_llm else loss

class IndentityPipeLayerLast(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
        self.occupy = nn.Linear(1000,1000,bias=False)
        nn.init.constant_(self.occupy.weight,0.)
    
    def forward(self,ipt):
        loss, bs = ipt
        # zero_in = torch.zeros((bs,self.occupy.in_features),device='cuda')
        # return loss + 0. * self.occupy(zero_in).sum()
        return loss

class IndentityPipeLayer(nn.Module):
    def __init__(self, model: Minigpt4Qwen):
        super().__init__()
    
    def forward(self,ipt):
        inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape = ipt
        return inputs_embeds, attention_mask, targets, rotary_pos_emb_list, position_ids, output_shape

def get_model(model,freeze_llm):
    layers = [LayerSpec(TokenizerPipeLayer,model=model),
            *[LayerSpec(IndentityPipeLayer,model=model) for _ in range(4)], # 调节控制多卡的显存分配
            *[LayerSpec(QwenBlockPipeLayer, model=model, layer_idx=idx) for idx in
                range(model.llm_model.transformer.config.num_hidden_layers)],
            LayerSpec(FLNPipeLayer, model=model),
            LayerSpec(LMPipeLayer, model=model),
            LayerSpec(LossPipeLayer, model=model),
        ]
    if freeze_llm:
        layers.append(LayerSpec(IndentityPipeLayerLast,model=model))
    return layers