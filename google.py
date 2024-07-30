# -*- coding: utf-8 -*-
import requests
from requests.exceptions import JSONDecodeError


def translate_text(text):
    url = f"https://findmyip.net/api/translate.php?text={text}"
    response = requests.get(url)

    try:
        data = response.json()
        print(data)
        if response.status_code == 200:
            if data["code"] == 200:
                translation = data["data"]["translate_result"]
                return translation
            elif data["code"] == 400:
                return data["error"]
            else:
                return "内部接口错误，请联系开发者"
        else:
            return "内部接口错误，请联系开发者"
    except JSONDecodeError as e:
        return f"JSON decoding error: {e}"
    except requests.RequestException as e:
        return f"Request error: {e}"


text_to_translate = "我爱52破解"
translation_result = translate_text(text_to_translate)
print("翻译结果:", translation_result)
