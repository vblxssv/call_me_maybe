import torch
from typing import List, Any, Optional, Dict
from llm_sdk import Small_LLM_Model
from .function_scheme import FunctionScheme, ParamType
import random


class LMManager:
    def __init__(self):
        self.model = Small_LLM_Model()
        self.tokenizer = self.model._tokenizer
        self.current_text: str = ''
        self.current_ids: List[int] = []

    def _get_encoded(self, context: str) -> List[int]:
        raw_data: Any = self.model.encode(context)[0]
        return list(raw_data.tolist())

    def reset(self) -> None:
        self.current_ids = []
        self.current_text = ""

    def _get_logits(self) -> torch.Tensor:
        return torch.as_tensor(
            self.model.get_logits_from_input_ids(self.current_ids))

    def _generate(self, mask: torch.Tensor) -> int:
        logits = self._get_logits()
        if mask.shape[0] < logits.shape[0]:
            padding = torch.zeros(logits.shape[0] - mask.shape[0], device=mask.device)
            mask = torch.cat([mask, padding])
        masked_logits = logits + mask.to(logits.device)
        return int(masked_logits.argmax().item())

    def sync_push(self, data: Any) -> None:
        if isinstance(data, str):
            ids = self._get_encoded(data)
            self.current_ids.extend(ids)
            self.current_text += data
        else:
            self.current_ids.extend(data)
            self.current_text += self.model.decode(data)

    def pick_word(self, words: List[str]) -> str:
        if not words:
            return ''

        result: str = ''

        target_sequences = [self._get_encoded(w) for w in words]
        quote_token_id = self._get_encoded('"')[0]

        step = 0
        active_sequences = target_sequences[:]

        while True:
            logits: torch.Tensor = self._get_logits()

            candidates = {quote_token_id: logits[quote_token_id].item()}
            for seq in active_sequences:
                if step < len(seq):
                    t_id = seq[step]
                    candidates[t_id] = logits[t_id].item()

            max_key: int = max(candidates, key=candidates.get)
            result += self.model.decode([max_key])
            self.sync_push([max_key])
            if max_key == quote_token_id:
                break
            step += 1
            active_sequences = [
                s for s in active_sequences if (step <= len(s) and
                                                s[step-1] == max_key)]
        return result

    def generate_until(self, p_type: ParamType) -> str:
        if p_type == ParamType.STRING:
            self.sync_push('"')
            stops = ['"']
        else:
            stops = [',', '}', ']', '\n', '"']

        mask = self._build_mask(stops)
        generated_text = ""
        while True:

            token_id = self._generate(mask)
            decoded_token = self.model.decode([token_id])
            print(repr(decoded_token))
            if any(s in decoded_token for s in stops):
                if p_type == ParamType.STRING:
                    self.sync_push('"')
                break

            self.sync_push([token_id])
            generated_text += decoded_token

            if len(generated_text) > 70:
                if p_type == ParamType.STRING:
                    self.sync_push('"')
                    break

        return generated_text

    def _build_mask(self, forbiden: List[str],
                    size: Optional[int] = None,
                    boost: float = 10.0) -> torch.Tensor:
        vocab: Dict[str, int] = self.tokenizer.get_vocab()
        actual_vocab_size: int = size if size is not None else len(vocab)

        mask: torch.Tensor = torch.zeros(actual_vocab_size)

        for _, token_id in vocab.items():
            if token_id >= actual_vocab_size:
                continue
            try:
                decoded_t: str = self.model.decode([token_id])
            except Exception:
                continue

            has_stop: bool = any(s in decoded_t for s in forbiden)
            is_pure_stop: bool = decoded_t in forbiden
            has_control: bool = any(ord(c) < 32 for c in decoded_t)

            if (has_stop and not is_pure_stop) or has_control:
                mask[token_id] = -float('inf')
            elif is_pure_stop:
                mask[token_id] = boost

        return mask
