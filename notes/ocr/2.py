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


image_path = "./telugu2.png"

base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Extract the table from the image. Give in json format with columns and the respective rows.",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ],
)

print(response.output_text)

# {
#     "columns": [
#         "వరుస సంఖ్య/పరిపత్రాలు",
#         "కృషి శాఖ మండలం",
#         "గ్రామ కార్యదర్శి మండలం",
#         "ఉపాధి కోఆర్డినేటర్ మండలం",
#         "ఢా‌క్టర్ మండలం",
#         "అభి వ్యమ్ద్ర మండలం",
#         "ఉపాధి కరె మండలం"
#     ],
#     "rows": [
#         {
#             "1": "1. ప్రొగరం, ముఖ్యంగా నామజు దేవదారికి",
#             "2": "-",
#             "3": "-",
#             "4": "కూడా ఉండు సత్వనం, తరగతే, నేలుగగి సత్వనం",
#             "5": "కోడక, కెదికే, కాశీ రోడ్డి",
#             "6": "-",
#             "7": "నామజు సత్వనం, తరం గీతి, వండ్రాంతి"
#         },
#         {
#             "1": "2. స్వస్త కు చేయించు",
#             "2": "సర్ల, ఇంధ్ర, పరిశీలన, పచుని, శ్రామణి, పార్వతీము, పామురి, పన్నూరి, వైపు. జాయ్, ಅಂತರ್ಟ, 9674. ఎమ్.టి. రేంద్రం, సవచం వివాహం, సైల, మేషన్, భరణి,, భన్ను, వూజు. జాయ్ 1318",
#             "3": "సధ్య, రిప్పులు, కర్యుహ, విధ్వంసం, భరణి, కొదుమ, సెఖ్లీ, కంఠీ, జ్వాల్ యుంఖాంకం, వుడి రుయు 1318",
#             "4": "కరాణో సవృరూప, నేకంరేం, బేవక్, కాని కార్యకేష్లి, దృείου నయ్యు ఎనంగ్విని" ,
#             "5": "డక్, గంటే, అక్కర నిండేగను వె, దయకె, ముకపే చ్యాన్ఫ్ 9-9674, ఏక మవ్యాక్ ధూడ్య",
#             "6": "మహిశిక ధరెేని, కృపాంత బయెన్",
#             "7": "పన్నంత్రేం, పేడ్యం వాద తెధంక తనపుంతి, కాలానికి కరణ్యుకం చిల్ప"
#         }
#     ]
# }
