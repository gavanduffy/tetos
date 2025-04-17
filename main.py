# main.py
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
import tetos # Assumes tetos is installable (e.g., via requirements.txt)
import os
import io
import logging
from contextlib import asynccontextmanager

# Configure logging (optional but recommended)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App instance
app = FastAPI()

# --- Helper to get OpenAI Speaker ---
def get_openai_speaker(model="tts-1", voice="alloy", speed=1.0):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE") # Optional
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    # Ensure tetos.openai is accessible. If running locally after `pip install -e .`
    # or if tetos is installed in the Vercel environment, this should work.
    return tetos.openai.OpenAISpeaker(
        model=model,
        voice=voice,
        speed=speed if speed != 1.0 else None, # OpenAI default is 1.0
        api_key=api_key,
        api_base=api_base
    )

# --- Helper to get other Speakers (Example: Azure) ---
# You would add similar functions for other providers you want to support
def get_azure_speaker(voice=None, lang="en-US"):
     speech_key=os.getenv("AZURE_SPEECH_KEY")
     speech_region=os.getenv("AZURE_SPEECH_REGION")
     if not speech_key or not speech_region:
         raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables must be set.")
     # Instantiate the Azure speaker
     speaker = tetos.azure.AzureSpeaker(
         speech_key=speech_key,
         speech_region=speech_region,
         voice=voice
     )
     # Note: Azure speaker's lang is determined during synthesize/stream
     return speaker


# --- Generic TTS Endpoint ---
@app.post("/tts/{provider}")
async def generate_tts(provider: str, text: str, voice: str | None = None, lang: str = "en-US"):
    """
    Generic endpoint to generate TTS using a specified provider.
    Requires provider-specific environment variables to be set.
    """
    try:
        SpeakerClass = tetos.get_speaker(provider) # Get the class
        speaker_instance = None

        # --- Instantiate the correct speaker ---
        if provider == "openai":
            # Note: OpenAI TTS doesn't use 'lang' param directly in tetos wrapper
            # Voice/Model imply language. Speed is handled in get_openai_speaker
             speaker_instance = get_openai_speaker(voice=voice or "alloy") # Allow override
        elif provider == "azure":
             speaker_instance = get_azure_speaker(voice=voice)
        # elif provider == "google":
        #     # Add logic for GoogleSpeaker using GOOGLE_APPLICATION_CREDENTIALS env var or JSON key
        #     pass # speaker_instance = get_google_speaker(voice=voice)
        # elif provider == "edge":
        #     # Edge doesn't need keys, but might need rate/pitch/volume params
        #     speaker_instance = tetos.edge.EdgeSpeaker(voice=voice) # Add other params if needed
        # TODO: Add instantiation logic for other providers (Baidu, Volc, Minimax, Xunfei, Fish)
        # Each will need its specific env vars checked and passed to its constructor.

        else:
            # Fallback or raise error if provider setup isn't complete
            raise HTTPException(status_code=501, detail=f"Provider '{provider}' instantiation not fully implemented in this API.")

        if not speaker_instance:
             raise HTTPException(status_code=500, detail=f"Could not instantiate speaker for provider '{provider}'. Check logs and env vars.")

        # --- Stream the audio ---
        async def stream_audio():
            try:
                # Pass lang to stream method where applicable
                async for chunk in speaker_instance.stream(text, lang=lang):
                    yield chunk
            except tetos.base.SynthesizeError as e:
                logger.error(f"TTS Synthesis Error ({provider}): {e}")
                # In a real app, might try to stream an error message or just log
            except tetos.base.Duration: # Expected successful end for some stream methods
                pass
            except Exception as e:
                logger.error(f"Unexpected Error during TTS stream ({provider}): {e}", exc_info=True)
                # Handle unexpected errors

        return StreamingResponse(stream_audio(), media_type="audio/mpeg")

    except ValueError as e:
        # Handle cases like missing API keys from helper functions or invalid provider name
        logger.error(f"Configuration or Value Error for provider {provider}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except tetos.base.SynthesizeError as e:
        logger.error(f"Caught SynthesizeError for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Synthesis Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error processing /tts/{provider}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


# --- OpenAI Compatibility Endpoint ---
@app.post("/openai_tts")
async def translate_openai_to_tetos(
    model: str = "tts-1",
    input: str = "",
    voice: str = "alloy",
    speed: float = 1.0,
    response_format: str = "mp3" # tetos currently outputs mp3
):
    """
    Translates OpenAI TTS API parameters to tetos.openai.OpenAISpeaker calls.
    """
    if not input:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    if response_format != "mp3":
        raise HTTPException(status_code=400, detail="Tetos currently only supports mp3 format via OpenAI speaker.")

    try:
        # Instantiate the tetos OpenAI speaker using the helper
        speaker = get_openai_speaker(model=model, voice=voice, speed=speed)

        # Stream the audio
        async def stream_audio():
            try:
                # OpenAI stream doesn't use 'lang' param
                async for chunk in speaker.stream(input):
                    yield chunk
            except tetos.base.SynthesizeError as e:
                logger.error(f"OpenAI TTS Synthesis Error via wrapper: {e}")
            except tetos.base.Duration: # Expected end signal
                pass
            except Exception as e:
                logger.error(f"Unexpected Error during OpenAI TTS stream via wrapper: {e}", exc_info=True)

        return StreamingResponse(stream_audio(), media_type="audio/mpeg")

    except ValueError as e:
        # Handle cases like missing API keys from get_openai_speaker
        logger.error(f"Configuration or Value Error for /openai_tts: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except tetos.base.SynthesizeError as e:
        logger.error(f"Caught SynthesizeError for /openai_tts: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Synthesis Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error processing /openai_tts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")

# Optional: Add a root endpoint for health check or info
@app.get("/")
def read_root():
    return {"message": "Tetos TTS API Wrapper - Ready"}
