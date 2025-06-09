import os
import asyncio
import edge_tts

# Define default voice constant for consistency across the application
DEFAULT_VOICE = "en-US-AriaNeural"

# Make sure DEFAULT_VOICE is included in __all__ to ensure proper exports
__all__ = ['text_to_speech', 'list_voices', 'DEFAULT_VOICE', 'get_audio_for_text', 'list_available_voices']

async def text_to_speech(text, voice=DEFAULT_VOICE, output_file=None, rate=1.0):
    """
    Convert text to speech using Edge TTS.
    Args:
        text: Text to convert to speech
        voice: Voice to use for TTS
        output_file: If provided, save audio to this file
        rate: Speech rate multiplier (0.5-2.0, 1.0 is normal speed)
    Returns:
        Path to audio file or audio bytes
    """
    # Convert rate to proper format for Edge TTS (value in percentage)
    # Edge TTS expects rate in the format +X% or -X%
    if rate != 1.0:
        if rate > 1.0:
            # Convert to percentage increase
            percentage = int((rate - 1.0) * 100)
            rate_string = f"+{percentage}%"
        else:
            # Convert to percentage decrease
            percentage = int((1.0 - rate) * 100)
            rate_string = f"-{percentage}%"
        
        # Create communicator with rate adjustment
        communicate = edge_tts.Communicate(
            text, 
            voice,
            rate=rate_string
        )
    else:
        # Use default rate
        communicate = edge_tts.Communicate(text, voice)
    
    if output_file:
        await communicate.save(output_file)
        return output_file
    else:
        # Return audio data as bytes
        audio_data = bytes()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

async def list_voices():
    """
    Get list of available Edge TTS voices.
    Returns:
        List of voice info dictionaries with ShortName, Gender, and Locale fields
        to match the expected API format.
    """
    voices_list = await edge_tts.list_voices()
    # Convert Edge TTS voice format to our API format
    formatted_voices = []
    
    for voice in voices_list:
        formatted_voices.append({
            "ShortName": voice["ShortName"],
            "Gender": voice["Gender"],
            "Locale": voice["Locale"]
        })
    
    return formatted_voices

def get_audio_for_text(text, voice=DEFAULT_VOICE, output_file=None, rate=1.0):
    """
    Synchronous wrapper for the async TTS function
    """
    return asyncio.run(text_to_speech(text, voice, output_file, rate))

def list_available_voices():
    """
    Synchronous wrapper to get available voices
    """
    return asyncio.run(list_voices())
