import asyncio
import os
import re
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- Configuration ---
TETOS_EXECUTABLE = "tetos"
OUTPUT_DIR = "/tmp/tetos_audio_outputs"


# --- FastAPI Application ---
app = FastAPI(
    title="Tetos OpenAI-Compatible Proxy",
    description="A proxy that translates OpenAI TTS API calls to tetos CLI commands.",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Pydantic Models for OpenAI compatibility ---
class SpeechRequest(BaseModel):
    model: str = Field(
        ...,
        description="The TTS provider to use, e.g., 'edge', 'google', 'azure'.",
    )
    input: str = Field(..., description="The text to synthesize.")
    voice: str = Field(
        ..., description="The voice to use for the synthesis, e.g., 'en-GB-SoniaNeural'."
    )
    speed: Optional[float] = Field(
        1.0,
        description="The speed of the speech, from 0.25 to 4.0. 1.0 is the default.",
        ge=0.25,
        le=4.0,
    )


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, background_tasks: BackgroundTasks):
    provider = request.model
    output_file = os.path.join(OUTPUT_DIR, f"{os.urandom(8).hex()}.mp3")

    # --- Construct the tetos command ---
    # Options must come before the positional 'text' argument.
    command = [TETOS_EXECUTABLE, provider]

    # Add all options first.
    command.extend(["--output", output_file])

    if request.voice:
        command.extend(["--voice", request.voice])
        lang_match = re.match(r"([a-zA-Z]{2,3}-[a-zA-Z]{2})", request.voice)
        if lang_match:
            command.extend(["--lang", lang_match.group(1)])

    if request.speed is not None and request.speed != 1.0:
        speed = request.speed
        if provider == "edge":
            rate_percentage = int((speed - 1.0) * 100)
            rate_str = f"+{rate_percentage}%" if rate_percentage >= 0 else f"{rate_percentage}%"
            command.extend(["--rate", rate_str])
        elif provider in ["google", "minimax", "openai"]:
            command.extend(["--speed", str(speed)])

    # MODIFIED: Add the positional 'text' argument at the very end.
    command.append(request.input)

    print(f"Executing command: {' '.join(command)}")

    # --- Execute the command ---
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0 or not os.path.exists(output_file):
        error_message = stderr.decode().strip()
        print(f"Error executing tetos: {error_message}")
        if os.path.exists(output_file):
            os.remove(output_file)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate speech: {error_message or 'Unknown error from tetos CLI.'}"
        )

    background_tasks.add_task(os.remove, output_file)

    return FileResponse(
        path=output_file,
        media_type="audio/mpeg",
        filename="output.mp3",
    )


@app.get("/")
def read_root():
    return {"message": f"Tetos OpenAI-Compatible Proxy is running. Output files are temporarily stored in {os.path.abspath(OUTPUT_DIR)}."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
