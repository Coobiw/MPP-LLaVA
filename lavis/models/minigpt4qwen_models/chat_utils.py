from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import logging
from transformers.generation import LogitsProcessor

logger = logging.get_logger(__name__)

# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

def decode_tokens(
    tokens: Union[torch.LongTensor, TokensType],
    tokenizer: PreTrainedTokenizer,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str="replace",
) -> str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        # not support chat_format == 'raw'
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str='replace'
):
    end_reason = f"Gen length {len(tokens)}"
    for eod_token_idx in range(len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors))
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
    addtion_tokens_num: int = 32,  #TODO: for img tokens, now: not support, just mark!
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens # now: bos_token = nl_tokens, so don't need this addition
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n" # now: bos_token = nl_tokens, so don't need "\n"
    else: # not support chat_format == 'raw'
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens