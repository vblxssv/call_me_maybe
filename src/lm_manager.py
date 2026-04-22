from typing import List, Any, Optional, Dict, Final

import torch
from llm_sdk import Small_LLM_Model

from .function_scheme import ParamType


class LMManager:
    """
    Manages the LLM state, tokenization, and constrained decoding logic.

    This class provides tools to force the model to pick from a list of words
    or generate text until a specific parameter type boundary is met.
    """

    def __init__(self) -> None:
        """Initialize the LM manager and pre-build logit masks."""
        self.model: Final[Small_LLM_Model] = Small_LLM_Model()
        self.tokenizer: Any = self.model._tokenizer
        self.current_text: str = ""
        self.current_ids: List[int] = []

        self.string_mask: torch.Tensor = self._build_mask(['"'])
        self.digit_mask: torch.Tensor = self._build_mask(
            [',', '}', ']', '\n', '"']
        )

    def _get_encoded(self, context: str) -> List[int]:
        """
        Encode string into token IDs.

        Args:
            context: The text to encode.

        Returns:
            A list of integer token IDs.
        """
        raw_data: Any = self.model.encode(context)[0]
        return list(raw_data.tolist())

    def reset(self) -> None:
        """Clear the current generation context."""
        self.current_ids = []
        self.current_text = ""

    def _get_logits(self) -> torch.Tensor:
        """
        Get the next token logits from the model based on current context.

        Returns:
            A tensor of logits.
        """
        return torch.as_tensor(
            self.model.get_logits_from_input_ids(self.current_ids)
        )

    def _generate(self, mask: torch.Tensor) -> int:
        """
        Generate a single token using a provided logit mask.

        Args:
            mask: A tensor to be added to logits for filtering.

        Returns:
            The predicted token ID.
        """
        logits = self._get_logits()
        if mask.shape[0] < logits.shape[0]:
            padding = torch.zeros(
                logits.shape[0] - mask.shape[0],
                device=mask.device
            )
            mask = torch.cat([mask, padding])

        masked_logits = logits + mask.to(logits.device)
        return int(masked_logits.argmax().item())

    def sync_push(self, data: Any) -> None:
        """
        Push text or token IDs into the current generation context.

        Args:
            data: Either a string or a list of token IDs.
        """
        if isinstance(data, str):
            token_ids = self._get_encoded(data)
            self.current_ids.extend(token_ids)
            self.current_text += data
        else:
            self.current_ids.extend(data)
            self.current_text += self.model.decode(data)

    def pick_word(self, words: List[str]) -> str:
        """
        Force the model to choose from a predefined list of words.

        Args:
            words: Candidate strings.

        Returns:
            The word chosen by the model.
        """
        if not words:
            return ""

        result: str = ""
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

            max_key: int = max(candidates, key=lambda k: float(candidates[k]))
            result += self.model.decode([max_key])
            self.sync_push([max_key])

            if max_key == quote_token_id:
                break

            step += 1
            active_sequences = [
                s for s in active_sequences
                if step <= len(s) and s[step - 1] == max_key
            ]

        return result

    def generate_until(self, p_type: ParamType) -> str:
        """
        Generate text until a stop symbol or length limit is reached.

        Args:
            p_type: The type of the parameter being generated.

        Returns:
            The generated string value.
        """
        is_string = (p_type == ParamType.STRING)

        if is_string:
            self.sync_push('"')
            stops = ['"']
            mask = self.string_mask
            is_first_token = True
        else:
            stops = [',', '}', ']', '\n', '"']
            is_first_token = False
            mask = self.digit_mask

        generated_text = ""

        while True:
            token_id = self._generate(mask)
            decoded_token = self.model.decode([token_id])

            if any(s in decoded_token for s in stops):
                if is_string:
                    self.sync_push('"')
                break
            if is_string and is_first_token:
                decoded_token = decoded_token.lstrip()
                is_first_token = False
                token_id = self._get_encoded(decoded_token)[0]

            self.sync_push([token_id])
            generated_text += decoded_token
            if len(generated_text) > 70:
                if is_string:
                    self.sync_push('"')
                break

        return generated_text

    def _build_mask(self, forbidden: List[str],
                    size: Optional[int] = None,
                    boost: float = 10.0) -> torch.Tensor:
        """
        Create a logit mask for generation constraints.

        Args:
            forbidden: List of characters that should stop generation.
            size: Vocabulary size.
            boost: Score added to valid pure stop tokens.

        Returns:
            A tensor mask to be added to logits.
        """
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

            has_stop: bool = any(s in decoded_t for s in forbidden)
            is_pure_stop: bool = decoded_t in forbidden
            has_control: bool = any(ord(c) < 32 for c in decoded_t)
            if (has_stop and not is_pure_stop) or has_control:
                mask[token_id] = -float('inf')
            elif is_pure_stop:
                mask[token_id] = boost

        return mask
