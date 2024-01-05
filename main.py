from flask import Flask, request, jsonify
from PIL import Image
import PyPDF2
from io import BytesIO
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
from datetime import datetime
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

model = "openai" if os.getenv("MODEL") == "openai" else "gemini"

app = Flask(__name__)

embedding_model = SentenceTransformer('all-minilm-l6-v2')

if model == "openai":
    openaiClient = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

if model == "gemini":
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-pro")


prompt = """
Based on the TEXT. Do NER.
RULES:
- ONLY output in this JSON format.
- DON'T ADD ANY MESSAGE. ONLY GIVE ME THE JSON.
- If you do not find it in the text, set it as null.
- You must obey the format.
- The JSON Value MUST be in English.
- The date must be in DD/MM/YY. No text allowed.

JSON FORMAT:
{
"name": "name of the person",
"lastEducationInstitution": "last education institution",
"lastEducationDegree": "last education degree. The format MUST be, "Bachelor of ...", "Master of ...", "Doctor of ...", etc",
"lastEducationMajor": "last education major",
"lastEducationStartDate": "The format MUST be in DD/MM/YYYY. Text NOT allowed",
"lastEducationEndDate": "The format MUST be in DD/MM/YYYY. Text NOT allowed",
"skills": ["list of skill"],
"birthday": "person birthday in DD/MM/YYYY format"
}

TEXT:\n
"""


def convert_to_ddmmyy(date_str):
    try:
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d-%m-%y", "%d/%m/%y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%d/%m/%y")
            except ValueError:
                continue
        return None
    except Exception as e:
        return None
    
def reformatted_json(json):
    json["lastEducationStartDate"] = convert_to_ddmmyy(json["lastEducationStartDate"])
    json["lastEducationEndDate"] = convert_to_ddmmyy(json["lastEducationEndDate"])
    json["birthday"] = convert_to_ddmmyy(json["birthday"])

    for key, value in json.items():
        if value == "null" or value == "Null" or value == "NULL":
            json[key] = None
    
    return json


def generate_ner_openai(text):
    response = openaiClient.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text,
            }
        ],
        model="gpt-3.5-turbo"
    )

    return response.choices[0].message.content

def generate_ner_gemini(text):
    return gemini_model.generate_content(text).text

def resume_ner(text):
    input = prompt + text

    result = generate_ner_openai(input) if model == "openai" else generate_ner_gemini(input)

    json = re.findall(r"{(?:[^{}]*{[^{]*})*[^{}]*}", result)[0]

    return json


def api_error(code, message, http_code):
    return jsonify({"code": code, "message": message}), http_code


def process_image(image_buffer):
    image = Image.open(image_buffer)
    gray_image = image.convert("L")

    threshold_value = 150
    binary_image = gray_image.point(lambda x: 0 if x < threshold_value else 255, '1')

    output = pytesseract.image_to_string(binary_image, lang="eng")

    return output


def readPDF(file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return api_error("FILE_REQUIRED", "File field is required", 400)

    file = request.files["file"]
    file_mime_type = request.form["filetype"]

    if file_mime_type == "image/jpeg" or file_mime_type == "image/png":
        content = process_image(file)
    elif file_mime_type == "application/pdf":
        pdfContent = readPDF(file)

        if pdfContent == "":
            return api_error("INVALID_FILE", "Invalid file", 400)
        else:
            content = pdfContent
    else:
        return api_error("INVALID_FILE", "Invalid file", 400)

    retry_count = 0

    while retry_count < 3:
        try:
            ner = resume_ner(content)

            predicted = json.loads(ner)

            reformatted = reformatted_json(predicted)

            skills = reformatted["skills"]
            reformatted["skills"] = []
            for skill in skills:
                reformatted["skills"].append({
                    "name": skill,
                    "vector": embedding_model.encode(skill).tolist()
                })

            if reformatted["lastEducationInstitution"] is not None:
                reformatted["lastEducationInstitution"] = {
                    "name": reformatted["lastEducationInstitution"],
                    "vector": embedding_model.encode(reformatted["lastEducationInstitution"]).tolist()
                }
            
            if reformatted["lastEducationMajor"] is not None:
                reformatted["lastEducationMajor"] = {
                    "name": reformatted["lastEducationMajor"],
                    "vector": embedding_model.encode(reformatted["lastEducationMajor"]).tolist()
                }

            if reformatted["lastEducationDegree"] is not None:
                reformatted["lastEducationDegree"] = {
                    "name": reformatted["lastEducationDegree"],
                    "vector": embedding_model.encode(reformatted["lastEducationDegree"]).tolist()
                }

            return jsonify({"result": reformatted})
        except Exception as e:
            retry_count += 1
    

    return api_error("ERROR_PARSING", "Error parsing", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
