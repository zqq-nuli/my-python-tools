import requests
import base64

class GetVoice:
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "priority": "u=1, i",
        "referer": "https://www.google.com/",
        "sec-ch-ua": "\"Chromium\";v=\"124\", \"Google Chrome\";v=\"124\", \"Not-A.Brand\";v=\"99\"",
        "sec-ch-ua-arch": "\"x86\"",
        "sec-ch-ua-bitness": "\"64\"",
        "sec-ch-ua-full-version": "\"124.0.6367.208\"",
        "sec-ch-ua-full-version-list": "\"Chromium\";v=\"124.0.6367.208\", \"Google Chrome\";v=\"124.0.6367.208\", \"Not-A.Brand\";v=\"99.0.0.0\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": "\"\"",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-ch-ua-platform-version": "\"15.0.0\"",
        "sec-ch-ua-wow64": "?0",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "x-dos-behavior": "Embed"
    }
# 代理选填
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

    @staticmethod
    def write_speech(text, language, outputfile):
        if not outputfile.lower().endswith(".mp3"):
            outputfile += ".mp3"
        
        text = text.replace(",", "%2C")  # 特殊处理，否则translate将会以为是,结束参数
        text = requests.utils.quote(text)

        url = f"https://www.google.com/async/translate_tts?&ttsp=tl:{language},txt:{text},spd:1&cs=0&async=_fmt:jspb"

        response = requests.get(url, headers=GetVoice.headers, proxies=GetVoice.proxies)

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return False

        response_body = response.text
        response_body = response_body[len(")]}'\n{\"translate_tts\":[\""):]
        response_body = response_body[:len(response_body) - len("\"]}")]

        try:
            data = base64.b64decode(response_body)
            with open(outputfile, "wb") as file:
                file.write(data)
        except Exception as e:
            print(e)
            return False

        return True

class Language:
    Chinese = "zh-CN"
    English = "en"

# 测试函数使用
text = "你好，这是一个测试。"
language = Language.Chinese
outputfile = "output.mp3"

success = GetVoice.write_speech(text, language, outputfile)

if success:
    print(f"语音文件已成功保存为 {outputfile}")
else:
    print("语音文件生成失败")