import asyncio
import random
import time

import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.face_detection as mp_face_detection
import mediapipe.python.solutions.hands as mp_hands
import numpy as np
import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import threading
from threading import Thread

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

SHOT_DELAY = 3
SHOT_SIGNAL_TIME = 0.4
SHOT_FIRE_TRACER_TIME = 0.2
SHOT_COOLDOWN_TIME = 1
SHOT_NORMAL_COLOR = (255, 0, 0)
SHOT_SIGNAL_COLOR = (0, 255, 0)
SHOT_FIRE_COLOR = (0, 0, 255)

INITIAL_HEALTH = 4
DAMAGE_FACE = 2
DAMAGE_HAND = 1
DAMAGE_GRACE_PERIOD = 0.75

SCORE_PER_KILL = 100


class Rect:
    def __init__(self, position=[0, 0], dimension=256) -> None:
        self.position = list(position)
        self.dimension = dimension
        self.hits = []

    def get_end(self):
        return self.position[0] + self.dimension, self.position[1] + self.dimension

    def compute_hits(self, origin: np.ndarray, direction: np.ndarray):
        self.hits.clear()
        end = self.get_end()

        t = (self.position[0] - origin[0]) / direction[0]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[1] >= self.position[1] and cast_pos[1] <= end[1]:
                self.hits.append(cast_pos)

        t = (end[0] - origin[0]) / direction[0]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[1] >= self.position[1] and cast_pos[1] <= end[1]:
                self.hits.append(cast_pos)

        t = (self.position[1] - origin[1]) / direction[1]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[0] >= self.position[0] and cast_pos[0] <= end[0]:
                self.hits.append(cast_pos)

        t = (end[1] - origin[1]) / direction[1]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[0] >= self.position[0] and cast_pos[0] <= end[0]:
                self.hits.append(cast_pos)

        return self.hits

    def chase(self, target, delta_time):
        diff = np.array(self.position) - target
        norm = np.linalg.norm(diff)
        if norm == 0:
            return
        chase_direction = diff / norm
        offset = chase_direction * 100 * delta_time
        self.position[0] -= offset[0]
        self.position[1] -= offset[1]

    def get_hits(self):
        return self.hits

    def intersects(self, other_box) -> bool:
        x1min, y1min = self.position
        x1max, y1max = self.get_end()
        x2min, y2min, w, h = other_box
        x2max = x2min + w
        y2max = y2min + h

        return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

    def render(self, image):
        rect_color = (0, 255, 0) if len(self.hits) > 0 else (0, 0, 255)
        end = self.get_end()
        cv2.rectangle(
            image,
            (int(self.position[0]), int(self.position[1])),
            (int(end[0]), int(end[1])),
            rect_color,
            3,
        )

voiceSelect = False

def voice():
    args = {
        'energy_threshold' : 1000,
        'default_microphone' : 'pulse',
        'model' : 'small',
        'record_timeout': 1,
        'phrase_timeout': .5,
        'non_english': False
    }

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args['energy_threshold']
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args['default_microphone']
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args['model']
    if args['model'] != "large" and not args['non_english']:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args['record_timeout']
    phrase_timeout = args['phrase_timeout']

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    if "select this" in line.lower():
                        voiceSelect = True
                        print("huzzah!!")
                    if "hello there" in line.lower():
                        voiceSelect = True
                        print("huzzah!!")
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)
            else:
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

game_is_on = False
game_over = False

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5,
)


# # Initialize Hand Gesture Recognizer
# BaseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
# VisionRunningMode = mp.tasks.vision.RunningMode


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, image, timestamp_ms: process_results(result),
)
recognizer = GestureRecognizer.create_from_options(options)
recognizer_is_on = True

cap = cv2.VideoCapture(0)  # 0 for default camera

# initializing things for the chasing rectangle
rect_list = []

start_time = time.time()
previous_loop_time = start_time
previous_shot_time = start_time
previous_shot_tracer = None
previous_damaged_time = None
previous_spawn_time = None
spawn_delay = None

health = INITIAL_HEALTH
score = 0

# game_is_on = True

def process_results(result):
    if result.gestures:
        for gesture in result.gestures:
            print(gesture)
            if gesture[0].category_name == "Thumb_Up" and gesture[0].score > 0.6:
                print("Thumbs_Up")
                global game_is_on
                game_is_on = True


while cap.isOpened():
    print(game_is_on)
    current_time = time.time()
    delta_time = current_time - previous_loop_time
    previous_loop_time = current_time

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(image)
    face_results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, c = image.shape
    if game_is_on:
        if recognizer_is_on:
            recognizer.close()
            recognizer_is_on = False
    elif recognizer_is_on:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        recognizer.recognize_async(mp_image, int((current_time - start_time) * 1000))

    if game_is_on and (
        previous_spawn_time is None
        or spawn_delay is None
        or current_time - previous_spawn_time > spawn_delay
    ):
        x = (
            random.uniform(0, 200)
            if random.choice([True, False])
            else random.uniform(w - 200, w)
        )
        y = (
            random.uniform(0, 100)
            if random.choice([True, False])
            else random.uniform(h - 100, h)
        )
        rect_list.append(Rect([x, y], 64))
        previous_spawn_time = current_time
        spawn_delay = random.normalvariate(4, 1)

    face_bounding_box = None
    is_hit = 0  # 0 is not hit, 1 is hit on face, 2 is hit on hand

    if face_results.detections:
        detection = next(iter(face_results.detections))

        mp_drawing.draw_detection(image, detection)

        if game_is_on:
            bbox = detection.location_data.relative_bounding_box
            fbbx = int(bbox.xmin * w)
            fbby = int(bbox.ymin * h)
            fbbw = int(bbox.width * w)
            fbbh = int(bbox.height * h)
            face_center = np.array((fbbx + fbbw / 2, fbby + fbbh / 2))
            face_bounding_box = (fbbx, fbby, fbbw, fbbh)

            for rect in rect_list:
                rect.chase(face_center, delta_time)
                if is_hit == 0 and rect.intersects(face_bounding_box):
                    is_hit = 1

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not game_is_on:
                continue

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip = np.array((int(index_tip.x * w), int(index_tip.y * h)))
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp = np.array((int(index_mcp.x * w), int(index_mcp.y * h)))
            direction = index_tip - index_mcp
            end = direction * 100 + index_tip

            hit_indices = []  # The indices of the rectangles that are hit.
            for i, rect in enumerate(rect_list):
                hits = rect.compute_hits(index_tip, direction)
                for hit in hits:
                    cv2.circle(image, (hit[0], hit[1]), 10, (0, 0, 255), -1)
                if len(hits) > 0:
                    hit_indices.append(i)

                if not face_results.detections:
                    rect.chase(index_mcp, delta_time)

                if is_hit != 0:
                    continue
                rect_end = rect.get_end()
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if (
                        x >= rect.position[0]
                        and x <= rect_end[0]
                        and y >= rect.position[1]
                        and y <= rect_end[1]
                    ):
                        is_hit = 2
                        break

            delta_time_shot = current_time - previous_shot_time
            if delta_time_shot < SHOT_FIRE_TRACER_TIME:
                pass
            elif delta_time_shot < SHOT_COOLDOWN_TIME:
                previous_shot_tracer = None
                pass
            elif delta_time_shot < SHOT_DELAY:
                cv2.line(
                    image,
                    (index_tip[0], index_tip[1]),
                    (end[0], end[1]),
                    SHOT_NORMAL_COLOR,
                    5,
                )
            elif delta_time_shot < SHOT_DELAY + SHOT_SIGNAL_TIME:
                cv2.line(
                    image,
                    (index_tip[0], index_tip[1]),
                    (end[0], end[1]),
                    SHOT_SIGNAL_COLOR,
                    5,
                )
            else:
                previous_shot_tracer = ((index_tip[0], index_tip[1]), (end[0], end[1]))
                previous_shot_time = current_time
                for i in reversed(hit_indices):
                    score += SCORE_PER_KILL
                    rect_list.pop(i)

            if previous_shot_tracer is not None:
                start, end = previous_shot_tracer
                cv2.line(
                    image,
                    start,
                    end,
                    SHOT_FIRE_COLOR,
                    8,
                )
            # if voiceSelect:
            #     if len(hit_indices) > 0:
            #         print(hit_indices, 'boxes were selected')
            #     voiceSelect = False


    if is_hit != 0 and (
        previous_damaged_time is None
        or current_time - previous_damaged_time > DAMAGE_GRACE_PERIOD
    ):
        health -= DAMAGE_FACE if is_hit == 1 else DAMAGE_HAND
        if health <= 0:
            game_over = True
            game_is_on = False
            rect_list.clear()
        previous_damaged_time = current_time
        print(f"Health: {health}")

    for rect in rect_list:
        rect.render(image)

    image = cv2.flip(image, 1)

    if game_is_on:
        cv2.putText(
            image,
            f"Health: {health}",
            (75, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
        )
        cv2.putText(
            image,
            f"Score: {score}",
            (75, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
        )
    elif game_over:
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 4
        thickness = 6
        text = "Game Over"
        text_width, text_height = cv2.getTextSize(
            text, font_face, font_size, thickness
        )[0]
        org = (int((w - text_width) / 2), 200)
        cv2.putText(
            image,
            text,
            org,
            font_face,
            font_size,
            (0, 0, 255),
            thickness,
        )
        text = f"Score: {score}"
        text_width, _ = cv2.getTextSize(text, font_face, font_size, thickness)[0]
        org = (int((w - text_width) / 2), 200 + text_height + 48)
        cv2.putText(
            image,
            text,
            org,
            font_face,
            font_size,
            (0, 0, 255),
            thickness,
        )
    else:
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 4
        thickness = 6
        text = "Thumbs Up to Start"
        text_width, text_height = cv2.getTextSize(
            text, font_face, font_size, thickness
        )[0]
        org = (int((w - text_width) / 2), 200)
        cv2.putText(
            image,
            text,
            org,
            font_face,
            font_size,
            (0, 0, 255),
            thickness,
        )

    # image = cv2.resize(image, (1000, 750))
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# # asyncio.run(voice())
# graphicThread = Thread(target=visual)
# graphicThread.run()
# voiceThread = Thread(target=voice)
# voiceThread.run()
