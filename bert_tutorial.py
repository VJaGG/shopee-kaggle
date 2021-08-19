import torch
from transformers import BertModel, BertTokenizer
from transformers import AutoConfig


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
# print(tokenizer)

config = AutoConfig.from_pretrained(model_name,
                                    output_hidden_states=True)
model = BertModel.from_pretrained(model_name, config=config)

input_text = "Here is some text to encode"

input_ids = tokenizer.encode(input_text, add_special_tokens=True)

input_ids = torch.tensor([input_ids])

with torch.no_grad():
    output = model(input_ids)
    print(output.keys())
    print(output['last_hidden_state'].shape)
    print(output['pooler_output'].shape)
    print(output['hidden_states'][0].shape)
    print(output['hidden_states'][1].shape)
    print(len(output['hidden_states']))
