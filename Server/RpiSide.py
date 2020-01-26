#!/usr/bin/env python3
from time import sleep, time
import sys
import sock_comm
import configRW


"""func selector"""

from copy import deepcopy as cp
import inspect
from time import time
import numpy as np
import cv2
from math import sin, cos, pi
from itertools import count
import os

BadPath = "bad/"
SamplesPath = "samples/"
try:
    badfiles = os.listdir(BadPath)
except FileNotFoundError:
    os.mkdir(BadPath)
    counter = count(0)
else:
    counter = count(len(badfiles))


def visio_core(input_image=None, parameters=None, order=None):
    """
    If the argument "input_image" or "parameters" is not given --> returns {func: [parameters]}
    If just order is not givet --> returns (funcs)
    If order is empty list --> returns None
    If any parameter is missing in {parameters: value} --> returns {missing_parameter}
    If any keyError from order is wrong --> returns {func: [parameters]}
    """
    if parameters is not None:
        Commons = {}
        Commons["StartTime"] = time()

    if order is not None and len(order) == 0:
        return None

    func_ref = locals().items()
    func_ref = {k: v for k, v in locals().items()}

    def dilate(img=None, dilation=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        dilation = dilation//40
        img = cp(img)
        dilation = (dilation, dilation)
        img = cv2.dilate(img, np.ones(dilation, np.uint8))
        return img

    def erode(img=None, erosion=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        erosion = erosion//40
        img = cp(img)
        erosion = (erosion, erosion)
        img = cv2.erode(img, np.ones(erosion, np.uint8))
        return img

    def close(img=None, closing=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        closing = closing//40
        img = cp(img)
        closing = (closing, closing)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones(closing, np.uint8))
        return img

    def open(img=None, opening=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        opening = opening//40
        img = cp(img)
        opening = (opening, opening)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones(opening, np.uint8))
        return img

    def intoGrey(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]
        img = cp(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def negative(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]
        img = cp(img)
        return cv2.bitwise_not(img)

    def median(img=None, med_val=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        med_val = med_val//4
        if med_val % 2 == 0:
            med_val += 1
        img = cv2.medianBlur(img, med_val)
        return img

    def blur(img=None, blur_val=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        blur_val = blur_val//4
        if blur_val % 2 == 0:
            blur_val += 1
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, blur_val)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def bright_contr(img=None, brightness=None, contrast=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]
        img = cp(img)

        brightness = (brightness//2)-255
        contrast = (contrast//4)-127

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        if contrast != 0:
            f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
        return img

    def gamma(img=None, gamma_val=1.0):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        inv_gamma = 1.0 / ((gamma_val/20)+1.0)
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img, table)

    def diff_orig(img=None, bg_val=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        bg_val = bg_val//4
        if bg_val > 255:
            bg_val = 255
        if input_image.shape == img.shape:
            img = bg_val - cv2.absdiff(input_image, img)
        return img

    def binarize(img=None, thresh=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        thresh = thresh // 4
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, thresh, 255, 0)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def bin_adaptive1(img=None, thresh1=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        thresh1 = thresh1 // 4
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, thresh1)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def bin_adaptive2(img=None, thresh2=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        thresh2 = thresh2 // 4
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, thresh2)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def normalize(img=None, alpha=None, beta=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        alpha = alpha//4
        beta = beta//4

        img = cp(img)
        return cv2.normalize(img, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def sobel_y(img=None, ksizey=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        ksizey = ksizey//80
        if ksizey % 2 == 0:
            ksizey += 1

        img = cp(img)
        img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksizey).astype("uint8")
        return img

    def sobel_x(img=None, ksizex=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        ksizex = ksizex//80
        if ksizex % 2 == 0:
            ksizex += 1

        img = cp(img)
        img = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksizex).astype("uint8")
        return img

    def laplacian(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        img = cv2.Laplacian(img, cv2.CV_64F).astype("uint8")
        return img

    def canny(img=None, val1=None, val2=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        img = cv2.Canny(img, val1, val2)
        return img

    def floodfill_TR(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        floodfill_color = 255, 255, 255
        seed_point = img.shape[1] - 1, 0
        cv2.floodFill(img, None, seed_point, floodfill_color)
        return img

    def floodfill_BR(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        floodfill_color = 255, 255, 255
        seed_point = img.shape[1] - 1, img.shape[0] - 1
        cv2.floodFill(img, None, seed_point, floodfill_color)
        return img

    def floodfill_TL(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        floodfill_color = 255, 255, 255
        seed_point = 0, 0
        cv2.floodFill(img, None, seed_point, floodfill_color)
        return img

    def floodfill_BL(img=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        img = cp(img)
        floodfill_color = 255, 255, 255
        seed_point = 0, img.shape[0]-1
        cv2.floodFill(img, None, seed_point, floodfill_color)
        return img

    def findcircle(img=None, crcl_thrsh1=None, crcl_thrsh2=None, min_rad=None, max_rad=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]

        crcl_thrsh1 = crcl_thrsh1//4
        crcl_thrsh2 = crcl_thrsh2//4
        max_rad = max_rad//8
        min_rad = min_rad//8

        if crcl_thrsh1 < 1:
            crcl_thrsh1 = 1
        if crcl_thrsh2 < 1:
            crcl_thrsh2 = 1
        img = cp(img)
        img2 = cp(img)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height = img.shape[0]
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, height / 8,
                                   param1=crcl_thrsh1, param2=crcl_thrsh2,
                                   minRadius=min_rad, maxRadius=max_rad)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles))

            firstcircle = circles[0, 0]
            Commons["circle"] = firstcircle
            center = (firstcircle[0], firstcircle[1])
            radius = firstcircle[2]

            cv2.circle(img, center, radius, (0, 255, 0), 3)    # circle outline

            img = cv2.bitwise_and(img2, img)
        else:
            Commons["circle"] = None

        return img

    def raycasting(img=None, rozptyl=None, offset=None, delka=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]
        img = cp(img)

        try:
            firstcircle = Commons["circle"]
        except KeyError:
            Commons["blacks"] = None
            return img

        try:
            center = (firstcircle[0], firstcircle[1])
            radius = firstcircle[2]
        except TypeError:
            Commons["blacks"] = None
            return img

        if center == 0:
            return img

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        rozptyl = rozptyl//50
        offset = offset//100
        delka = delka//50
        r_min = radius - rozptyl + offset
        r_max = r_min + delka
        img_res = img.shape[:2]

        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blacks = []

        for angle in range(0, 360):
            black = 0
            alpha = (angle / 360 * (pi * 2))
            for d in range(r_min, r_max):

                x = int(center[0] + (cos(alpha) * d))
                y = int(center[1] + (sin(alpha) * d))

                if y >= img_res[0] or x >= img_res[1]:
                    continue

                if test_img[y, x] == 0:
                    img[y, x] = 255, 0, 0
                    black += 1
                else:
                    img[y, x] = 255, 255, 0

            if black > 0:
                blacks.append(black)
        Commons["blacks"] = blacks
        return img

    def simple_evaluation(img=None, ev_thresh=None):
        ars, _, _, vals = inspect.getargvalues(inspect.currentframe())
        vals = [vals[ar] for ar in ars]
        if any(x is None for x in vals):
            return ars[1:]
        img = cp(img)

        ev_thresh = ev_thresh//50

        h, w = img.shape[:2]
        img = np.zeros((h, w, 3), np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType = 4
        try:
            blacks = Commons["blacks"]
        except KeyError:
            text = "?"
            fontColor = (255, 255, 0)
            bottomLeftCornerOfText = (0, 60)
            return cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        if blacks is None or len(blacks) == 0:
            text = "OK"
            fontColor = (0, 255, 0)
            bottomLeftCornerOfText = (0, 60)
            res_pict = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            return res_pict

        nejvice = max(blacks)

        if nejvice < ev_thresh:
            text = f"{ev_thresh}>{nejvice}"
            fontColor = (0, 255, 0)
            bottomLeftCornerOfText = (0, 60)
        else:
            text = f"{ev_thresh}>{nejvice}"
            fontColor = (255, 0, 0)
            bottomLeftCornerOfText = (0, 60)
            cv2.imwrite(f"{BadPath}badOne_{next(counter)}.jpg", input_image)
        res_pict = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        return res_pict

    funcs = {k: v for k, v in locals().items() if k not in func_ref}
    fwp = {k: (v, v()) for k, v in funcs.items()}     # funcs_with_params

    if input_image is None or parameters is None:
        return {k:  v() for k, v in funcs.items()}

    if order is None:
        return tuple(funcs)

    args = set()
    for v in funcs.values():
        for p in v():
            args.add(p)
    missing = args - parameters.keys()

    if len(missing) > 0:
        return missing

    images = []

    try:
        for i, _ in enumerate(order):
            if i == 0:
                images.append(fwp[order[i]][0](input_image, *[parameters[a] for a in fwp[order[i]][1]]))
                continue
            images.append(fwp[order[i]][0](images[i - 1], *[parameters[a] for a in fwp[order[i]][1]]))
    except KeyError:
        return {k: v() for k, v in funcs.items()}

    # print(time()-Commons["StartTime"])
    return images


configfile = "config.ini"


def new_config_file():
    configRW.write_config({"_DEFAULT": {},
                           "MAIN_CONFIG": {"last_state": "_DEFAULT"}}, configfile)


if not os.path.isfile(configfile):
    print("Konfigurační soubor neexistuje, vytvořen nový.")
    new_config_file()

AllconfigData = configRW.read_config(configfile)

if AllconfigData["MAIN_CONFIG"]["last_state"] not in AllconfigData:
    print("Konfigurační soubor poškozen, vytvořen nový.")
    new_config_file()

lastState = AllconfigData["MAIN_CONFIG"]["last_state"]
configData = AllconfigData[lastState]

Write = False
requiredVars = {
                "order": [],
                "triggerpin": 12,
                "capture_delay": 0,
                "exposition": 10000,
                "resolution_w": 1024,
                "resolution_h": 768,
                "framerate": 40,
                "x_crop": 0,
                "y_crop": 0,
                "w_crop": 1,
                "h_crop": 1
                }


def create_config_keys(key, def_val):
    if key not in configData:
        print(f'No key named "{key}" --> added key into configdata.')
        configData[key] = def_val
        return True
    else:
        return False


for key, val in requiredVars.items():
    if create_config_keys(key, val):
        Write = True

funcs = visio_core()

for func in funcs:
    for arg in funcs[func]:
        if arg not in configData:
            configData[arg] = 0
            Write = True

if Write:
    AllconfigData.update({lastState: configData})
    configRW.write_config(AllconfigData, configfile)

comCommands = {"CroppedIMG_request": "?", "Save_corfim": "OK"}


WelcomeMessage = AllconfigData, visio_core, comCommands
SVR = sock_comm.Server(2020, ip="", welcome=WelcomeMessage)


def SEND(obj):
    return SVR.send_message(obj)


def GET():
    last_msg = SVR.LAST_MESSAGE
    if last_msg is None:
        return None
    else:
        new_msg = SVR.LAST_MESSAGE
        SVR.LAST_MESSAGE = None
        return new_msg


def camset():
    PiCamCapture.trigger_detect(AllconfigData[lastState]["triggerpin"])
    PiCamCapture.camera.shutter_speed = AllconfigData[lastState]["exposition"]
    PiCamCapture.captDelay = AllconfigData[lastState]["capture_delay"] / 1000
    PiCamCapture.camera.resolution = (AllconfigData[lastState]["resolution_w"],
                                      AllconfigData[lastState]["resolution_h"])
    PiCamCapture.camera.framerate = AllconfigData[lastState]["framerate"]
    PiCamCapture.x = AllconfigData[lastState]["x_crop"]
    PiCamCapture.y = AllconfigData[lastState]["y_crop"]
    PiCamCapture.w = AllconfigData[lastState]["w_crop"]
    PiCamCapture.h = AllconfigData[lastState]["h_crop"]

    PiCamCapture.CurrConfigData = AllconfigData[lastState]


Image = None


def simfromfile():
    global Image
    while True:
        for imfile in os.listdir(SamplesPath):
            Image = cv2.imread(f"{SamplesPath}{imfile}")
            visio_core(input_image=Image, parameters=AllconfigData[lastState], order=AllconfigData[lastState]["order"])
            sleep(2)


SIMULACE = False
if len(sys.argv) > 1 and sys.argv[1] == "sim":
    SIMULACE = True

if SIMULACE:
    import threading
    ri = threading.Thread(target=simfromfile)
    ri.start()
else:
    import PiCamCapture
    PiCamCapture.Visio_core = visio_core
    camset()
    Image = PiCamCapture.img

while Image is None:
    sleep(1)

LAST_MESSAGE = None
while True:
    sleep(0.5)
    LAST_MESSAGE = GET()
    if LAST_MESSAGE is not None:
        # print("Incoming message, type:", type(LAST_MESSAGE).__name__)
        if LAST_MESSAGE == comCommands["CroppedIMG_request"]:
            # print("incoming request for image, time: ", time())
            if not SIMULACE:
                Image = PiCamCapture.img
            SEND(Image)
        if type(LAST_MESSAGE).__name__ == "dict":
            # print("incoming configuration, time: ", time())
            AllconfigData.update({lastState: LAST_MESSAGE})
            configRW.write_config(AllconfigData, configfile)
            SEND(comCommands["Save_corfim"])
            if not SIMULACE:
                camset()
