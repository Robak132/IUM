import requests
import json


if __name__ == '__main__':
    with open("data/ab_test/ab_test_good_data.json") as file:
        json_request = json.load(file)
        x = requests.post("http://localhost:8080/predict/good_results", json=json_request)
        json_result = json.loads(x.text)
        print(json.dumps(json_result, indent=4, sort_keys=True))
