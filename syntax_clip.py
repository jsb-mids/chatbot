import torch
import torch.nn as nn
import clip
from PIL import Image
import pandas as pd
import requests
import os.path as osp
import pickle
import random
import numpy as np
from pathlib import Path
import sys
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


import cv2
import glob
import tensorflow as tf
import clip

import os
import glob
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

img = []
uid = []
for file in glob.glob('/Users/siddharthashokkumar/Desktop/shuowang_github/notebook/demo_data/image/*.*'):
  # img.append(tf.convert_to_tensor(cv2.imread(file)))
  image = preprocess(Image.open(file)).unsqueeze(0)
  img.append(image)


for filename in os.listdir('/Users/siddharthashokkumar/Desktop/shuowang_github/notebook/demo_data/image'):
    if filename.endswith("jpg"):
        uid.append(filename.strip('.jpg'))


data = pd.DataFrame(list(zip(uid, img)),
              columns=['uid','encoded_image'])

# print(data.head())



class Timer:
    def __init__(self):

        self.t1 = None

    @staticmethod
    def delta_to_string(td):

        res_list = []

        def format():
            return ", ".join(reversed(res_list)) + " elapsed."

        seconds = td % 60
        td //= 60
        res_list.append(f"{round(seconds,3)} seconds")

        if td <= 0:
            return format()

        minutes = td % 60
        td //= 60
        res_list.append(f"{minutes} minutes")

        if td <= 0:
            return format()

        hours = td % 24
        td //= 24
        res_list.append(f"{hours} hours")

        if td <= 0:
            return format()

        res_list.append(f"{td} days")

        return format()

    def __enter__(self):

        self.t1 = time.time()

    def __exit__(self, *args, **kwargs):

        t2 = time.time()
        td = t2 - self.t1

        print(self.delta_to_string(td))

import openai
openai.api_key = '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'


def find_products(text_input, data):
    print(f"finding products for query: {text_input}...")
    text_input = [text_input]

    data = data[~data["encoded_image"].isna()]
    image_uids = list(data["uid"].values)

    encoded_images = torch.cat(list(data["encoded_image"].values)).to(device)
    encoded_texts = clip.tokenize(text_input).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(encoded_images, encoded_texts)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()

    res = dict(zip(image_uids, probs[0] * 100))
    res = dict(sorted(res.items(), key=itemgetter(1), reverse=True)[:5])

    return res

def show_images(res):
    n = len(res)
    fig, ax = plt.subplots(1, n)

    fig.set_figheight(5)
    fig.set_figwidth(5 * n)

    for i, image in enumerate(res.keys()):
        img_path = image_path(image)
        img = mpimg.imread(img_path)
        ax[i].imshow(img)
        ax[i].axis('off')
        # ax[i].set_title(get_label(image), fontsize=8)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


def image_path(uid):
    return osp.join('/Users/siddharthashokkumar/Desktop/shuowang_github/notebook/demo_data/image', f"{uid}.jpg")


messages = []

res_list = []

prefix = (
    "considering what the user asked before, what is the user looking for with the following request."
    " Only respond with the product description no more than 30 words:"
)
while True:
    message = input("User : ")
    if message:
        print(f"User entered: {message}")
        messages.append(
            {"role": "user", "content": f"{prefix} {message}"},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")

        with Timer():
            print("looking for products...")
            res_list.append(find_products(reply, data))
            show_images(res_list[-1])
            print("found products")

        messages.append({"role": "assistant", "content": reply})
