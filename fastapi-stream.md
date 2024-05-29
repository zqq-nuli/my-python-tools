```sh
pip install fastapi httpx uvicorn
```


```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import json

app = FastAPI()

async def openai_streaming_response():
    url = "https://api.openai.com/v1/engines/davinci-codex/completions"
    headers = {
        "Authorization": f"Bearer YOUR_OPENAI_API_KEY",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": "Translate the following English text to French: '{}'",
        "max_tokens": 100,
        "stream": True
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, json=data) as response:
            token_count = 0
            async for chunk in response.aiter_bytes():
                chunk_str = chunk.decode('utf-8')
                lines = chunk_str.split("\n")
                for line in lines:
                    if line:
                        try:
                            # Each line is a JSON object
                            obj = json.loads(line)
                            if 'choices' in obj:
                                token_count += len(obj['choices'][0]['text'].split())
                                yield obj['choices'][0]['text']
                        except json.JSONDecodeError:
                            pass
            # Sending the total token count at the end
            yield f"\n[INFO] Total tokens used: {token_count}"

@app.get("/stream")
async def stream_endpoint(request: Request):
    return StreamingResponse(openai_streaming_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

OpenAI python
```python
import openai
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import os

app = FastAPI()

# 设置OpenAI的API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

async def openai_streaming_response():
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt="Translate the following English text to French: '{}'",
        max_tokens=100,
        stream=True
    )
    
    token_count = 0

    for event in response:
        if 'choices' in event:
            text = event['choices'][0]['text']
            token_count += len(text.split())
            yield text
    
    yield f"\n[INFO] Total tokens used: {token_count}"

@app.get("/stream")
async def stream_endpoint(request: Request):
    return StreamingResponse(openai_streaming_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```


```html
<!DOCTYPE html>
<html>
<head>
    <title>OpenAI Streaming</title>
</head>
<body>
    <div id="output"></div>
    <script>
        const outputDiv = document.getElementById('output');

        async function fetchStream() {
            const response = await fetch('http://localhost:8000/stream');
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            let result;
            while (!(result = await reader.read()).done) {
                const chunk = decoder.decode(result.value);
                outputDiv.innerHTML += chunk + '<br>';
            }
        }

        fetchStream();
    </script>
</body>
</html>
```

