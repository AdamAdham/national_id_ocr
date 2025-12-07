import requests
import json


image = open("app/test_ids/youssef.jpg", "rb")

response = requests.post("https://ml-service-518472151016.europe-west3.run.app/predict", files={"file": image})

print(response.text)

response_json = response.json()

with open("test_output.json", "w", encoding="utf-8") as f:
    json.dump(response_json, f, ensure_ascii=False, indent=4)
    