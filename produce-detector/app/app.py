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

from model import ConvNet, MODEL, IM_SCALE, CLASS_NAMES
from metadata_keys import md_prefix_target, md_hit_infs

'''TODO:
fix port and targets before dockerizing
'''

NET = ConvNet().float()#.half().cuda()
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
            im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)#.cuda().half()
            with torch.no_grad():
                cls, loc = NET(im)
                cls = cls.item()
                loc = loc.tolist()[0]
                loc = [max(min(l, 1.0), 0.0) for l in loc]
            
            cls_string = CLASS_NAMES[cls]
            inf = {"class": cls_string,  "x":loc[0], "y":loc[1], "width":loc[2], "height":loc[3]}
            im_md = j["image_metadata"]
            print(im_md)

            data = {
                "inference": [],
                "image": im_64}
            if md_prefix_target in im_md:
                num = im_md.split("_")[1]
                print(num)
                print(md_hit_infs.keys())
                if num in md_hit_infs:
                    print("inf hit")
                    inf = md_hit_infs[num]
                    data["inference"].append(inf)
                    print(inf)
            else:
                data["inference"].append(inf)

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