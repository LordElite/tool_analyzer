from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os

# ---- Your modules ----
from tool_analyzer import get_image_informations, llama_agents, speak, listen_for_command, take_picture
from ai_dev_lab.llm import crag_tool_model

app = FastAPI()

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system_content1 = """You are a helpful assistant that detects 
                    if user queries are related to a tool, devide or product, for 
                    example hammers, laptops, calculators, and so on.
                  In case of user queries are talking about
                  usage or data of tools,  only say 'yes',otherwise, just say 'this assistant isn't able to answer that topic'"""
                  
system_content2 = """You are a helpful assistant that detects if user queries contain a brand of  a tool or product.
                  In case to detects a brand you only say 'yes',otherwise, just say 'no'
                you delete every quote or strange symbol except question marks"""

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("tool.html", {"request": request})


@app.post("/process/")
async def process_tool():
    # 🗣️ 1️⃣ Listen for spoken command
    command = listen_for_command()

    if "tool" in command or "device" in command:
        # 📸 2️⃣ Take a picture
        take_picture()
    else:
        return JSONResponse(content={
            "result": "Please mention a tool or device in your command!",
            "image_description": ""
        })

    # 3️⃣ Analyze image
    try:
        result = get_image_informations("captured_image.jpg")
    except Exception as e:
        print(f"⚠️ Error: {e}")
        result = {
            "image_description": "No recognizable tool or device detected.",
            "main_objects": []
        }

    if not result or not result.get("main_objects"):
        result = {
            "image_description": "No recognizable tool or device detected.",
            "main_objects": []
        }

    # 4️⃣ LLM check
    if result["main_objects"]:
        answer1 = llama_agents(system_content1, result['brand']).choices[0].message.content
        if 'yes' in answer1.lower():
            answer2 = llama_agents(system_content2, result['image_description']).choices[0].message.content
            if 'yes' in answer2.lower():
                final_answer = crag_tool_model.agent(
                    f"{command}. tool: {result['tool_type']}, "
                    f"brand: {result['brand']}, model: {result['model']}"
                )
            else:
                final_answer = result['image_description']
        else:
            final_answer = answer1
    else:
        final_answer = "Sorry, no valid tool or device detected in the image."

   

    # 5️⃣ Speak the final answer
    speak(final_answer)

    return final_answer


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)
