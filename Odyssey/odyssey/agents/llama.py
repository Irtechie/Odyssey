# current model: llama-2-13b-chat
# """
# Sample usage:
# from llama import chat_completion
# response = chat_completion(messages)
# print(response)

import requests

from http import HTTPStatus
import dashscope
from langchain.schema import AIMessage
from odyssey.utils import config
# with open(Path(__file__).parent.parent.parent / "conf/config.json", "r") as config_file:
#     config = json.load(config_file)

class ModelType:
    LLAMA2_70B = 'llama2_70b'
    LLAMA3_8B_V3 = 'llama3_8b_v3'
    LLAMA3_8B = 'llama3_8b'
    LLAMA3_70B_V1 = 'llama3_70b_v1'
    QWEN2_72B = 'qwen2-72b'
    QWEN2_7B = 'qwen2-7b'
    BAICHUAN2_7B = 'baichuan2-7b'

def call_with_messages(msgs, model_name:ModelType=ModelType.LLAMA3_8B_V3):
    host = config.get("server_host")
    port = config.get("server_port")
    base_path = config.get("openai_base_path") or "/v1"
    model_override = config.get("openai_model")
    model_id = model_override or model_name
    url = f"http://{host}:{port}{base_path}/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": msgs[0].content},
            {"role": "user", "content": msgs[1].content},
        ],
        "temperature": 0,
    }
    result = requests.post(url, json=payload, timeout=240)
    if result.status_code >= 400 and not model_override:
        models_url = f"http://{host}:{port}{base_path}/models"
        models_resp = requests.get(models_url, timeout=30)
        models_resp.raise_for_status()
        models = models_resp.json().get("data", [])
        if models:
            payload["model"] = models[0]["id"]
            result = requests.post(url, json=payload, timeout=240)
    result.raise_for_status()
    json_result = result.json()
    content = json_result["choices"][0]["message"]["content"]
    return AIMessage(content=content)

def call_with_messages_(msgs):
    dashscope.api_key = config.get("api_key")  # API KEY
    messages = [{'role': 'system', 'content': msgs[0].content},
                {'role': 'user', 'content': msgs[1].content}
                ]
    response = dashscope.Generation.call(
        model='llama3-8b-instruct',
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return AIMessage(content=response["output"]["choices"][0]["message"]["content"])
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
