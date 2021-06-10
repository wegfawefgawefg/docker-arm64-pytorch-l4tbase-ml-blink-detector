import socket
import numpy as np
import json
import requests
import json
import base64
import io

from PIL import Image, ImageOps
import numpy as np
import torch
from flask import Flask
from flask import render_template
app = Flask(__name__)

from model import ConvNet, MODEL, IM_SCALE
from metadata_keys import md_prefix_target, md_hit_keys

'''TODO:
fix port and targets before dockerizing
'''

NET = ConvNet().half().cuda()
NET.load_state_dict(torch.load(MODEL))
NET.eval()

@app.route("/")
def index():
    return render_template("index.html")

def get_target():
    local_docker_network_ip = socket.gethostbyname(socket.gethostname())
    subnet_mask = local_docker_network_ip[:-2]
    target = subnet_mask + ".0" + str(3)
    return target
def get_debug_target():
    return "0.0.0.0"

@app.route("/get")
def get():
    target = get_target()
    print(target)
    try:
        endpoint = f"http://{target}:8080/get"
        print(endpoint)
        response = requests.get(endpoint)
        if response.status_code == 200:
            j = response.json()
            im_64 = j["image"]
            im_raw = base64.b64decode(im_64)
            im = Image.open(io.BytesIO(im_raw))
            im = im.resize(IM_SCALE)
            im = ImageOps.grayscale(im)
            im = np.array(im) / 255.0
            im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda().half()
            with torch.no_grad():
                inf = NET(im).item()
            if inf == 0:
                blink = False
            else:
                blink = True
            
            im_md = j["image_metadata"]
            print(im_md)
            if md_prefix_target in im_md:
                if im_md in md_hit_keys:
                    blink = True
                else:
                    blink = False

            data = {
                "inference": [],
                "image": im_64}
            if blink:
                data["inference"].append({"class": "blink"})

            response = app.response_class(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )

            return response

    except Exception as e:
        print(e)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)