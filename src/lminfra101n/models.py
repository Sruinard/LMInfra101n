import os
from lminfra101n import configs
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def _load_peft_model(model, artifact_dir):
    peft_model = PeftModel.from_pretrained(model, artifact_dir)
    return peft_model

def _load_base_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    return tokenizer, model

class RepositoryModel:
    def __init__(self, cfg: configs.SharedModelCfg):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None
        self._adapter = None
        self._artifact_path = os.path.join(cfg.artifact_dir, cfg.artifact_name)

    def load_tokenizer_and_model(self, is_peft: bool = False):
        return self._load(is_peft)
    
    def save_tokenizer_and_peft_model(self):
        return self._save()

    def _load(self, is_peft: bool):
        if not is_peft:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            self._tokenizer, self._model = _load_base_tokenizer_and_model(self.cfg.model_name)
            return self._tokenizer, self._model
        
        else:
            if self._adapter is not None:
                return self._tokenizer, self._adapter
            if self._tokenizer is None or self._model is None:
                self._load()
            self._adapter = _load_peft_model(self._model, self.cfg.artifact_dir)
            return self._tokenizer, self._adapter

    def _merge(self):
        tokenizer, peft_model = self._load(is_peft=True)
        return tokenizer, peft_model.merge_and_unload()

    def _save(self, is_peft: bool = True):
        if is_peft:
            if self._adapter is None:
                raise ValueError("No PEFT model loaded to save.")
            t, m = self._merge()
            m.save_pretrained(self.artifact_path, safe_serialization=True, max_shard_size="2GB")
            t.save_pretrained(self.artifact_path)
        else:
            raise NotImplementedError("Saving base models is not implemented.")