"""
Hair AI MVP v0.3 — real image analysis through the OpenAI API.

Usage:
    python vision_analyzer.py path/to/photo.jpg

Environment:
    OPENAI_API_KEY=...
    OPENAI_VISION_MODEL=gpt-5.6-luna   # optional

The model is explicitly instructed to return "unknown"/null when the photo
does not support a reliable conclusion. The app must not treat confidence
scores as calibrated probabilities yet; they are product signals for v0.3.
"""

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna")

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "photo_observation": {
            "type": "object",
            "properties": {
                "face_visible": {"type": "boolean"},
                "hair_visible": {"type": "boolean"},
                "headwear": {"type": "boolean"},
                "hair_tied": {"type": "boolean"},
                "lighting_ok": {"type": "boolean"},
                "blurry": {"type": "boolean"},
                "angle": {
                    "type": "string",
                    "enum": ["front", "left_profile", "right_profile", "back", "three_quarter", "unknown"]
                },
                "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": [
                "face_visible", "hair_visible", "headwear", "hair_tied",
                "lighting_ok", "blurry", "angle", "overall_confidence"
            ],
            "additionalProperties": False
        },
        "face": {
            "type": "object",
            "properties": {
                "shape": {
                    "type": ["string", "null"],
                    "enum": ["oval", "oval_rectangular", "round", "square", "heart", "long", "diamond", "triangle", None]
                },
                "shape_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "proportions": {
                    "type": ["string", "null"],
                    "enum": ["balanced", "longer", "wider", "unknown", None]
                },
                "proportions_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["shape", "shape_confidence", "proportions", "proportions_confidence"],
            "additionalProperties": False
        },
        "hair": {
            "type": "object",
            "properties": {
                "texture": {
                    "type": ["string", "null"],
                    "enum": ["straight", "wavy", "curly", "coily", "unknown", None]
                },
                "texture_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "density": {
                    "type": ["string", "null"],
                    "enum": ["low", "medium", "high", "unknown", None]
                },
                "density_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "current_length": {
                    "type": ["string", "null"],
                    "enum": ["very_short", "short", "short_medium", "medium", "medium_long", "long", "unknown", None]
                },
                "length_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "hairline": {
                    "type": ["string", "null"],
                    "enum": ["regular", "receding", "widows_peak", "uneven", "not_visible", "unknown", None]
                },
                "hairline_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": [
                "texture", "texture_confidence", "density", "density_confidence",
                "current_length", "length_confidence", "hairline", "hairline_confidence"
            ],
            "additionalProperties": False
        },
        "head_shape": {
            "type": "object",
            "properties": {
                "value": {
                    "type": ["string", "null"],
                    "enum": ["balanced", "flat_occiput", "prominent_occiput", "asymmetric", "unknown", None]
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"}
            },
            "required": ["value", "confidence", "reason"],
            "additionalProperties": False
        }
    },
    "required": ["photo_observation", "face", "hair", "head_shape"],
    "additionalProperties": False
}

SYSTEM_PROMPT = """
You are the visual analysis component of a hairstyle recommendation MVP.

Analyze ONLY visible hairstyle-relevant characteristics:
- photo usability and angle
- face geometry at a coarse level
- visible hair texture, apparent density, current length, visible hairline
- coarse head shape only when the angle actually shows enough of it

Rules:
1. Never guess hidden information.
2. If hair is tied, covered, cropped out, heavily styled, or too dark/blurred to assess,
   return null/unknown and low confidence for affected fields.
3. A front photo cannot reliably establish the back of the skull. Keep head-shape
   confidence low unless the relevant side/back view is visible.
4. Do not infer ethnicity, sexual orientation, personality, health diagnosis,
   gender identity, attractiveness, or other sensitive/personal traits.
5. Confidence is 0..1 and reflects how well THIS PHOTO supports the observation.
6. Do not call any hairstyle objectively beautiful or ugly.
7. Return only the schema requested by the API.
"""

def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    mime, _ = mimetypes.guess_type(p.name)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Supported MVP formats: JPG/JPEG, PNG, WEBP.")
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def analyze_image(path: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and add your API key."
        )

    client = OpenAI()
    data_url = image_to_data_url(path)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze this photo for the hairstyle recommendation pipeline."
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": "high"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "hair_photo_analysis",
                "schema": VISION_SCHEMA,
                "strict": True
            }
        }
    )

    if not response.output_text:
        raise RuntimeError("The model returned no structured text output.")
    return json.loads(response.output_text)

def to_ranking_profile(analysis: dict, preferences: dict) -> dict:
    face = analysis["face"]
    hair = analysis["hair"]
    head = analysis["head_shape"]

    # Use the lower of texture/density confidence as a conservative hair confidence.
    hair_conf = min(hair["texture_confidence"], hair["density_confidence"])

    return {
        "face": {
            "shape": face["shape"],
            "confidence": face["shape_confidence"]
        },
        "hair": {
            "texture": hair["texture"],
            "density": hair["density"],
            "confidence": hair_conf
        },
        "head_shape": {
            "value": head["value"],
            "confidence": head["confidence"]
        },
        **preferences
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python vision_analyzer.py path/to/photo.jpg")
        raise SystemExit(2)

    result = analyze_image(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
