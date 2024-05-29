import httpx
import os
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
GENERATE_IMAGE_URL = "https://api.cloudflare.com/v1/generate_image"

async def generate_image():
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "image_parameters": {
            # your image generation parameters here
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(GENERATE_IMAGE_URL, headers=headers, json=data)
        if response.status_code != 200:
            return {"status": "failed", "error": response.text}
        return response.json()

# WebSocket endpoint for progress updates
@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        for progress in range(0, 101, 10):
            await websocket.send_json({"progress": progress})
            time.sleep(1)  # Simulate progress update interval
    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        await websocket.close()

# HTTP endpoint to start image generation
@app.post("/generate_image")
async def generate_image_endpoint():
    result = await generate_image()
    if result["status"] == "failed":
        return JSONResponse(content={"status": "failed", "error": result["error"]}, status_code=500)
    return JSONResponse(content={"status": "completed", "image_url": result["image_url"]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
