import json
import os
from dotenv import load_dotenv
from typing import Any, Dict
from litellm import completion

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.
    
    system_prompt = """
    You are a travel planner assistant. You must respond with a valid JSON object.
    The JSON object must strictly adhere to the following schema:
    {
      "destination": "Name of the destination",
      "price_range": "Description of cost (e.g., low-range, mid-range, high-range)",
      "ideal_visit_times": ["Time 1", "Time 2"],
      "top_attractions": ["Attraction 1", "Attraction 2", "Attraction 3"]
    }
    Do NOT include any extra text or explanation. Only output valid JSON.
    """
    
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"Generate a travel itinerary for {destination}."}
    ]
    
    
    response = completion(
      model=MODEL,
      messages=messages,
      api_key=api_key,
      response_format={"type": "json_object"}
    )
    content_str = response.choices[0].message.content
    
    try:
      data = json.loads(content_str)
    except json.JSONDecodeError:
      raise ValueError("LLM response was not valid JSON")
    
    
    REQUIRED_SCHEMA = {
      "destination": str,
      "price_range" : str,
      "ideal_visit_times" : list,
      "top_attractions" : list
    }
    for key, exp_val_type in REQUIRED_SCHEMA.items():
      if key not in data:
        raise ValueError(f"Missing required key: {key}")
      if not isinstance(data[key], exp_val_type):
        raise ValueError(f"Incorrect type for key: {key}")

    return data

"""
if __name__ == "__main__":
    from litellm import supports_response_schema
    from litellm import get_supported_openai_params
    
    MODEL = "groq/llama-3.3-70b-versatile"
    
    if get_supported_openai_params(model=MODEL):
        print(f"{MODEL} supports response format!")
    else:
        print(f"{MODEL} does NOT support response format.")

    if supports_response_schema(model=MODEL):
        print(f"{MODEL} supports response schema!")
    else:
        print(f"{MODEL} does NOT support response schema.")

"""