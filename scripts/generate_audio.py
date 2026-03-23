#!/usr/bin/env python3
"""
Generate audio files from clinical transcription JSONs using ElevenLabs Text to Dialogue API.
Uses eleven_v3 model with EU residency endpoint.
"""

import json
import os
import sys
import requests
from pathlib import Path

# Configuration
ELEVEN_BASE_URL = os.getenv("ELEVEN_BASE_URL", "https://api.eu.residency.elevenlabs.io/v1")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_transcription(json_path: str) -> dict:
    """Load a transcription JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dialogue_inputs(transcription: dict) -> list[dict]:
    """
    Convert transcription dialogues into ElevenLabs text-to-dialogue inputs.
    Each input has: text (with optional stage directions) and voice_id.
    """
    voice_map = {
        "medico": transcription["voces"]["medico_voice_id"],
        "paciente": transcription["voces"]["paciente_voice_id"],
    }

    inputs = []
    for fase in transcription["transcripcion"]:
        for dialogo in fase["dialogos"]:
            rol = dialogo["rol"]
            texto = dialogo["texto"]
            voice_id = voice_map[rol]

            # Audio tags are already embedded in the texto field from the JSON
            # (e.g. "[warmly] Buenos días..." or "[anxiously] ¿Es grave?")
            # Pass text directly to ElevenLabs v3 which interprets the tags
            inputs.append({
                "text": texto,
                "voice_id": voice_id,
            })

    return inputs


def generate_audio(inputs: list[dict], output_path: str) -> bool:
    """
    Call ElevenLabs Text to Dialogue API and save the audio.
    """
    url = f"{ELEVEN_BASE_URL}/text-to-dialogue"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": inputs,
        "model_id": "eleven_v3",
    }

    print(f"  Sending request to {url} with {len(inputs)} dialogue turns...")

    response = requests.post(url, json=payload, headers=headers, timeout=300)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        size_kb = len(response.content) / 1024
        print(f"  Audio saved: {output_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  ERROR {response.status_code}: {response.text[:500]}")
        return False


def process_transcription(json_path: str) -> bool:
    """Process a single transcription: load, build inputs, generate audio."""
    print(f"\nProcessing: {json_path}")

    transcription = load_transcription(json_path)
    case_id = transcription["id"]
    audio_rel_path = transcription["audio_file"]
    output_path = REPO_ROOT / audio_rel_path

    print(f"  Case: {case_id} - {transcription['diagnostico_principal']['nombre']}")

    inputs = build_dialogue_inputs(transcription)
    print(f"  Dialogue turns: {len(inputs)}")

    # Check character count (API limit ~5000 chars per request)
    total_chars = sum(len(i["text"]) for i in inputs)
    print(f"  Total characters: {total_chars}")

    if total_chars > 5000:
        print(f"  WARNING: Total chars ({total_chars}) exceeds 5000 limit. Splitting into chunks...")
        return generate_audio_chunked(inputs, str(output_path))

    return generate_audio(inputs, str(output_path))


def generate_audio_chunked(inputs: list[dict], output_path: str) -> bool:
    """
    Split dialogue into chunks under 5000 chars each, generate separately,
    then concatenate the audio files.
    """
    chunks = []
    current_chunk = []
    current_chars = 0

    for inp in inputs:
        char_count = len(inp["text"])
        if current_chars + char_count > 4500 and current_chunk:  # Leave margin
            chunks.append(current_chunk)
            current_chunk = [inp]
            current_chars = char_count
        else:
            current_chunk.append(inp)
            current_chars += char_count

    if current_chunk:
        chunks.append(current_chunk)

    print(f"  Split into {len(chunks)} chunks")

    # Generate each chunk
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = output_path.replace(".mp3", f"_part{i+1}.mp3")
        chars = sum(len(c["text"]) for c in chunk)
        print(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} turns, {chars} chars")
        success = generate_audio(chunk, chunk_path)
        if not success:
            return False
        chunk_files.append(chunk_path)

    # Concatenate chunks using ffmpeg if available
    if len(chunk_files) > 1:
        try:
            import subprocess
            list_file = output_path.replace(".mp3", "_filelist.txt")
            with open(list_file, "w") as f:
                for cf in chunk_files:
                    f.write(f"file '{cf}'\n")

            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                print(f"  Concatenated audio saved: {output_path}")
                # Clean up chunk files
                for cf in chunk_files:
                    os.remove(cf)
                os.remove(list_file)
                return True
            else:
                print(f"  ffmpeg error: {result.stderr[:200]}")
                print(f"  Keeping individual chunk files")
                return True
        except FileNotFoundError:
            print("  ffmpeg not found. Keeping individual chunk files.")
            return True
    else:
        # Only one chunk, just rename
        os.rename(chunk_files[0], output_path)
        return True


def main():
    if not ELEVEN_API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not set. Check .env file.")
        sys.exit(1)

    # Get JSON files to process from command line args, or process all
    if len(sys.argv) > 1:
        json_files = sys.argv[1:]
    else:
        # Process all JSON files in transcripciones/
        json_files = sorted(str(p) for p in (REPO_ROOT / "transcripciones").rglob("*.json"))

    print(f"ElevenLabs Text to Dialogue Audio Generator")
    print(f"Base URL: {ELEVEN_BASE_URL}")
    print(f"Files to process: {len(json_files)}")

    results = {"success": 0, "failed": 0}
    for json_file in json_files:
        try:
            ok = process_transcription(json_file)
            if ok:
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results["failed"] += 1

    print(f"\n{'='*50}")
    print(f"Results: {results['success']} success, {results['failed']} failed")


if __name__ == "__main__":
    main()
