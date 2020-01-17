#!/usr/bin/env python3

from time import sleep, time
import sys
import os
import concurrent.futures
import numpy as np
# import cv2
from resettabletimer import ResettableTimer
from itertools import count
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera

# GPIO settings
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(26, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(20, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(16, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(19, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(13, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(12, GPIO.IN, GPIO.PUD_DOWN)
GPIO.setup(6, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
trigger = 12
lightPin = 5

# camera settings
resolution = (800, 208)
camera = PiCamera()
camera.resolution = resolution
#camera.rotation = 180
camera.iso = 100
camera.shutter_speed = 3000
camera.exposure_mode = "off"
#time.sleep(3)
camera.awb_mode = "off"
camera.awb_gains = (1.4, 1.9)
camera.framerate = 90

#crop
x = 700	#horizontal offset
y = 50	#vertical offset
w = 100	#horizontal distance
h = 100	#vertical distance

# turn light off after:
lightTime = 10

# capture delay:
captDelay = 0.1

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

photo = np.zeros((resolution[1], resolution[0], 3), np.uint8)
img = photo[y:y+h, x:x+w]

Visio_core = None
CurrConfigData = None

counter = count()

def capture():
	sleep(0.1)
	if Visio_core is None or CurrConfigData is None or GPIO.input(trigger) == GPIO.LOW:
		return
	sleep(captDelay)
	start = time()
	global img
	camera.capture(photo, format="bgr", use_video_port=True)
	img = photo[y:y + h, x:x + w]
	NewImages = Visio_core(input_image=img, parameters=CurrConfigData, order=CurrConfigData["order"])

	evaluated = time()-start
	print(f"bombička číslo: {next(counter)}, spočítáno za: {round(evaluated, 4)}s")
	return


def light_control():
	sleep(0.05)
	if GPIO.input(trigger) == GPIO.HIGH:
		t.reset()
		if GPIO.input(lightPin) == GPIO.LOW:
			GPIO.output(lightPin, True)		#light ON


def time_filter(pin):
	executor.submit(light_control)
	executor.submit(capture)


GPIO.add_event_detect(trigger, GPIO.RISING, callback=time_filter, bouncetime=10)

t = ResettableTimer(lightTime, GPIO.output, [lightPin, False])
t.start()
