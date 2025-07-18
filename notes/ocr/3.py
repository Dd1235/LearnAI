import base64

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI()


class OutputTable(BaseModel):
    columns: list[str]
    rows: list[list[str]]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "./telugu_image.jpg"
base64_image = encode_image(image_path)


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the table from the image. The columns and respective rows.",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ],
    # response_format=OutputTable,
)

print(response.choices[0])
