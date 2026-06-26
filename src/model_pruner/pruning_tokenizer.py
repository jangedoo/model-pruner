import json
import os
import shutil

import torch
from transformers import PreTrainedTokenizerFast


class PrunedTokenizer(PreTrainedTokenizerFast):
    def set_id_map(self, old_to_new: dict[int, int]) -> None:
        size = max(int(k) for k in old_to_new) + 1
        id_list = [0] * size
        for old_id, new_id in old_to_new.items():
            id_list[int(old_id)] = int(new_id)
        self.init_kwargs["pruned_id_map"] = id_list
        self._map_tensor = torch.tensor(id_list, dtype=torch.long)

    def _ensure_map(self) -> None:
        if getattr(self, "_map_tensor", None) is not None:
            return
        id_list = self.init_kwargs.get("pruned_id_map")
        self._map_tensor = torch.tensor(id_list, dtype=torch.long) if id_list else None

    def _encode_plus(self, *args, **kwargs):
        encoding = super()._encode_plus(*args, **kwargs)
        self._ensure_map()
        if self._map_tensor is not None and "input_ids" in encoding:
            ids = encoding["input_ids"]
            if isinstance(ids, torch.Tensor):
                remapped = self._map_tensor.to(ids.device)[ids.long()].to(ids.dtype)
            else:  # list / nested list / numpy array
                remapped = self._map_tensor[torch.as_tensor(ids, dtype=torch.long)].tolist()
            encoding["input_ids"] = remapped
        return encoding

    def save_pretrained(
        self,
        save_directory,
        legacy_format=None,
        filename_prefix=None,
        push_to_hub=False,
        **kwargs,
    ):
        result = super().save_pretrained(
            save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
            push_to_hub=push_to_hub,
            **kwargs,
        )

        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["auto_map"] = {"AutoTokenizer": [None, "pruning_tokenizer.PrunedTokenizer"]}
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # ship this file alongside the model so trust_remote_code=True finds the class
        shutil.copy(__file__, os.path.join(save_directory, "pruning_tokenizer.py"))

        return result
