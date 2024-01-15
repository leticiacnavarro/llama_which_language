import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch import float32, nn

class LlamaClassificationModel():
    def __init__(self, model_id, quantizer, access_token):

        self.model_id = model_id
        self.access_token = access_token
         
        if(quantizer == '4b'):
            quantizer_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif(quantizer == '8b'):
            quantizer_cfg = BitsAndBytesConfig(
                    load_in_8bit=True
                    )    
        else:    
            quantizer_cfg = None

        self.quantizer_cfg = quantizer_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def set_model(self, id2label, label2id, num_labels):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            quantization_config= self.quantizer_cfg,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            token=self.access_token
        )        

    def prepare_model(self):
        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(float32)
            self.model.gradient_checkpointing_enable()  # reduce number of stored activations
            self.model.enable_input_require_grads()
        return self.model

    def summarize(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
        return self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    def make_question(self, question: str, inst: str):
        question = self.generate_prompt(question, inst)
        summary = self.summarize(question)
        print(summary)

    def generate_prompt(self, question, inst) -> str:
        return f"""[INST] <<SYS>> {inst.strip()}.<</SYS>>
    {question.strip()}[/INST]""".strip()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(float32)



