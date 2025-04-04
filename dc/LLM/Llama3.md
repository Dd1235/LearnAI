# Working with Llama 3

## Why run Llama 3 locally

- eg industrial automation
- privacy and security
- cost efficiency
- customization

`pip install llama-cpp-python`

```Python
from llama_cpp import Llama
llm = Llama(model_path = "path/to/model.gguf")
output = llm("What are some ways to improve customer retention?")

# output is in distionary format
output["choices"][0]["text"]
```

## turning Llama 3 parameters

### Llama 3 decoding parameters

- **temperature** : controls randomness
    - values usually between 0 and 1
    - close to zero, more predictable response
    - high: more creative
- **Top-K**: limits token selection to the most probably choices
    - limits how many of the most likely words LLama can choose from
    -Low K values ie 1: more predictable
    - high ie 50: more diverse response
- **Top-P**: adjusts token selection based on cumulative probability
    - high eg 1: more varied responses
    - less ie 0: less variation
- **Max tokens**: limits response length

```Python

llm = Llama(model_path="path/to/model.gguf")
output_concise = llm(
    "Describe an electric car.",
    temperature=0.2, # vs 0.8
    top_k=1, # vs 10
    top_p=0.4, # vs 0.9
    max_tokens=20 # vs 100
)
```
## Assigning Chat Roles

```Python

from llama_cpp import Llama
llm = Llama(model_path="path/to/model.gguf")
message_list = [...] # this list includes Roles
response = llm.create_chat_completion(
    messages = message_list
)
```

### System Roles

- ***system message** : instructions about how model should behave

```Python
system_message = "You are a business consultant who gives daata-driver answers."
message_list = [
    {
        "role":"system",
        "content", system_message
    }
]
```

### User role

- **user message** : prompt being asked to the model

```Python
system message = "You are a business consultant who  gives data-driver answers."
user_message = "What are the key factors in a successful marketing strategy?"

message_list = [{
    "role":"sysetm",
     "content": system_message
},
    {
      "role":"user", 
      "content": user_message  

}]
```

### Generating a response

```Python
# its role is assistant
message_list = ...
response = llm.create_chat_completion(messages = message_list)
response["choices"][0]['message']['content']
```

# Chapter 2

refer slides