from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
import base64
from ai_dev_lab.llm import tool_assistant
from together import Together
import pyttsx3
from serpapi.google_search import GoogleSearch
import cv2
import speech_recognition as sr


def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Listening for 'take picture'...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")

    return ""

def take_picture():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Picture saved: {filename}")
    else:
        print("Failed to capture image.")
    cam.release()


def speak(text, rate=150):  # default rate is ~200 wpm, lower is slower
    engine = pyttsx3.init()

    # Set speech rate
    engine.setProperty('rate', rate)

    # Set English voice
    for voice in engine.getProperty('voices'):
        if 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.say(text)
    engine.runAndWait()


# Load environment variables from the .env file
load_dotenv()

# Access your variables
langchain_key = os.getenv("LANGCHAIN_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")

llama_client = Together(api_key=together_api_key)
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}

load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)


class ImageInformation(BaseModel):
    image_description: str = Field(description="A short description of the image")
    main_objects: list[str] = Field(description="List of main objects or components in the image")
    tool_type: str = Field(description="What kind of tool is in the image")
    brand: str = Field(description="The brand name (e.g. Fluke)")
    model: str = Field(description="The model number if visible (e.g. 15B+)")
    visible_text: list[str] = Field(description="All visible text in the image")
    usage: str = Field(description="What this tool is used for")
    manual_search_query: str = Field(description="Best search query to find the tool's user manual")
    
    
# Set verbose
globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
 """Invoke model with image and prompt."""
 model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1024)
 msg = model.invoke(
             [HumanMessage(
             content=[
             {"type": "text", "text": inputs["prompt"]},
             {"type": "text", "text": parser.get_format_instructions()},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
             ])]
             )
 return msg.content

parser = JsonOutputParser(pydantic_object=ImageInformation)
def get_image_informations(image_path: str) -> dict:
   vision_prompt = """
      You are an expert in hardware tools and electronics.

      Given the image, extract the following information as precisely as possible:

      1. What type of tool is this? (e.g. multimeter, screwdriver, oscilloscope)
      2. Brand and Model Number (look for logos or printed text)
      3. Description of the image (what the image shows, including screen readings if any)
      4. Text visible in the image (e.g., voltage values, unit labels, dials, buttons)
      5. Key physical features or symbols on the device (ports, dials, labels)
      6. Intended usage (based on tool type, give a 1-line description)
      7. Based on the above, construct a search query to find the user manual online.
      8. Ignore animals, people or objects not considered tools, and ignore the background.

      Return the data in structured JSON format.
      """
   vision_chain = load_image_chain | image_model | parser
   return vision_chain.invoke({'image_path': f'{image_path}', 
                               'prompt': vision_prompt})
   


def llama_agents(system_content, user_content):
    agent_response = llama_client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
        
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    )
    return agent_response

def search_engine(engine: str , api_key: str, description: str, num: int ):
  
  params = {
    "engine": engine,
    "q": f"{description} in PDF",
    "api_key": api_key,
    "num": num,  # Limit number of results
  }

  search = GoogleSearch(params)
  results = search.get_dict()

  return results["organic_results"]
    