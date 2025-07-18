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

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    # "text": "Extract the table from the image. Give in the format of columns and the respective rows.",
                    "text": "Extract the table from the image and provide it in JSON format with columns and the respective rows.",
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

# Sure! Here is the extracted table:

# **ప్రత్యయాలు** (Pratyayalu):
# 1. ఉ, మూత, పూ, వె
# 2. నిన్, సూస్, లున్, కూళిగుంట
# 3. చేస్తున్న -చేసిన్, దోదిన్, దున్న
# 4. కొడకన్, స్కై
# 5. యల్నన్, కంటెన్, పుట్ట
# 6. కీప్, కిస్, యోక్త్, లోన్, పోపన్
# 7. అతగన్, నన్
# 8. ఓ, ఓరి, ఓవ, ఓని

# **విభక్తులు** (Vibhaktulu):
# - ప్రథమా  విభక్తి
# - ద్వితీయా  విభక్తి
# - తృతీయా  విభక్తి
# - చతుర్థి  విభక్తి
# - పంచమై  విభక్తి
# - షష్ఠి  విభక్తి
# - సప్తమి  విభక్తి
# - సంసర్గ ధిప్రథమా  విభక్తి

# ```json
# {
#   "columns": [
#     "పంట కాలం/పరిధి",
#     "ఖరీఫ్ మండలం",
#     "గోసాబి మండలం",
#     "ఉప్రీ క్రోప్ మండలం",
#     "ఉప్రీ మండలం",
#     "ఢపు మండలం",
#     "అంబావత్ పనందర్ మండలం",
#     "ఉద్యనక్రేట్ మండలం"
#   ],
#   "rows": [
#     {
#       "1": "1.లోగరు, ముండంగా సులభ కెమదారి",
#       "2": "-",
#       "3": "-",
#       "4": "కోవిడ్ దోర సూపరు, కరపంగా, సెప్తం, వడ్డీ, మండలం సాంక",
#       "5": "భద్ర, టియర్, కేరణ రొడ్కువాలు, వడ్డీ, సెప్తం, కార్త, నెలసుర సాంకం, అర్గం",
#       "6": "-",
#       "7": "కోవ్ట్ దోర సూపరు, కరపంగా, మర్కర్స్"
#     },
#     {
#       "1": "2. సపర్గ, సప్హాద్రం అగ్రహారం, జగరుపతగా",
#       "2": "భరవా, ఇన్ఫ్, భాస్రద్, చంపమకి, పతర్దవుక, మరో జైనా, ఇర్విక, భుదుభట్-9674, చిత్రలం, వేషလုပ်్, ఆడల్, తెమ్ ష్రేర్, అట్టిడర్, ష్రుప్తి, మిమృక, మద్దిరం సాంకం, ప్రాఫి తెలుస్తా, శ్రీక, సైక్షోప, బొత్బర్ర సపారం, నెలసూర్ ఫోన్ 1318",
#       "3": "వాథై, చంప ధృత్తా, బ్రిం, శరుద్ద్, పశుపతేరీత్, చమ్పమికిరి, ఇర్థి, సందుశారం, మలాడి ప్రాఫు, फेसरा मागो, తుప్తు, తిన్తా ఫీక్సు, గంబీర్ ఓవాలో, శ్రౌక, బాబారి అహం. ఫోన్ 1318",
#       "4": "క్రాఫతకో స్రమచీసనది, మర్కర్, సెప్తం, బర్డి, స్వభో సేవుడ్, స్తారచం బ్యానర్, సాంకం, డస్మి తోతఇం, శ్రీక, కేరువతికి, సెర్టోని, తేరిం, షుక్రి, స్ట్రుక్ క్రంజాలు సాంక. ఫోన్ 1318",
#       "5": "పణకో మసнаў, మీరికా, సెప్తం మర్చు, నిశార ప్రకోసం తోలెక, సాంకం",
#       "6": "సంతాం మూగుల్ల, తెల్ప, నెలసుర సాంకమ్, బర్రి, సేయా, సాల్క, గోదవి, పంధు ప్రాద్యా సాంక. సాంకం సెండన్ ఫేసెర్గాన్",
#       "7": "సూన్తం, పుషేర్కు, సెప్తం, పంకజిలో సాంస, కర్సి అధ్జీ ఫొందే సాన్"
#     }
#   ]
# }
# ```
