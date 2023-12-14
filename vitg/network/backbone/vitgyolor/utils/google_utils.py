# Google utils: https://cloud.google.com/storage/docs/reference/libraries

import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def gsutil_getsize(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", "")
    file = Path(weights).name

    msg = (
        weights
        + " missing, try downloading from https://github.com/WongKinYiu/yolor/releases/"
    )
    models = [
        "yolor-p6.pt",
        "yolor-w6.pt",
        "yolor-e6.pt",
        "yolor-d6.pt",
    ]  # available models

    if file in models and not os.path.isfile(weights):
        try:  # GitHub
            url = f"https://github.com/WongKinYiu/yolor/releases/download/v1.0/{file}"
            print(f"Downloading {url} to {weights}...")
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exists(weights) and os.path.getsize(weights) > 1e6  # check
        except Exception as e:  # GCP
            print("ERROR: Download failure.")
            print("")
            return


def gdrive_download(id="1n_oKgR81BJtqk75b00eAjdv03qVCQn2f", name="coco128.zip"):
    # Downloads a file from Google Drive. from vitg.utils.google_utils import *; gdrive_download()
    t = time.time()

    print(
        f"Downloading https://drive.google.com/uc?export=download&id={id} as {name}... ",
        end="",
    )
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove("cookie") if os.path.exists("cookie") else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(
        f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out} '
    )
    if os.path.exists("cookie"):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {name}'
    else:  # small file
        s = f'curl -s -L -o {name} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    os.remove("cookie") if os.path.exists("cookie") else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith(".zip"):
        print("unzipping... ", end="")
        os.system(f"unzip -q {name}")
        os.remove(name)  # remove zip to free space

    print("Done (%.1fs)" % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
