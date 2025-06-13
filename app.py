from typing import Dict, List
from fastapi import FastAPI, HTTPException, logger
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel
import logging

from new_agent import invoke_graph_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class APIUserInput(BaseModel):
    user_id: str
    user_input: str | None = None
    image_url: str | None = None
    chat_history: List[Dict] = []
    
@app.post('/new-input')
async def new_input(input: APIUserInput):
    try:
        input_state = {
            "user_id": input.user_id,
            "user_input": input.user_input,
            "image_url": input.image_url,
            "chat_history": input.chat_history
        }
        return StreamingResponse(
            invoke_graph_stream(input_state),
            media_type="text/plain",
            headers={
                "Transfer-Encoding": "chunked",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"Error in new-input endpoint: {str(e)}")
        raise