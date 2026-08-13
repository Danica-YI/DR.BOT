#!/usr/bin/env python3
"""Device-side triage runner with a cleaner state flow."""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from device.api_client import flush_triage_queue, post_triage_assessment
from device.camera import draw_overlay, open_camera
from device.controller import PersonDetector
from device.pose_classifier import PoseClassifier
from device.response import GestureStabilizer
from device.triage_flow import build_triage_assessment
from device.voice import OfflineVoice, VoiceAnswer


def initial_response_route(answer):
    if answer == "YES":
        return "VOICE_MODE"
    if answer == "NO":
        return "COMPLETE"
    return "SECOND_PROMPT"


def second_response_route(voice_answer=None, gesture_answer=None, expired=False):
    if voice_answer:
        return "VOICE_MODE"
    if gesture_answer:
        return "GESTURE_MODE"
    if expired:
        return "NO_RESPONSE"
    return "WAITING_FOR_VOICE_OR_GESTURE"


class DeviceTriageRunner:
    def __init__(self, args):
        self.args = args
        self.camera = open_camera(args.camera_index)
        self.detector = None if args.gesture_test else PersonDetector(model_name=args.model)
        self.classifier = PoseClassifier()
        self.voice = OfflineVoice(
            model_path=args.vosk_model,
            enabled=not args.no_voice and not args.gesture_test,
            audio_device=args.audio_device,
            whisper_model=args.whisper_model,
        )
        self.gesture_stabilizer = GestureStabilizer()
        self.voice_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-voice")
        self.voice_future = None
        self.second_window_ready = threading.Event()
        self.second_window_stop = threading.Event()
        self.second_window_started = 0.0
        self.second_voice_result = None
        self.after_answer_state = "NEXT_QUESTION"
        self.no_response_reason = None
        self.search_frame_count = 0
        self.last_detection = None

        self.state = "GESTURE_TEST" if args.gesture_test else "SEARCHING"
        self.last_gesture_result = None
        self.center_frames = 0
        self.state_started_at = 0.0
        self.result_time = 0.0
        self.bbox = None
        self.status_text = None
        self.assessment = None
        self.response_type = None
        self.interaction_mode = None
        self.interaction_language = None
        self.current_question = None
        self.question_index = -1
        self.answers = {}
        self.answer_details = []
        self.resource_requested = False
        self.gesture_deadline = 0.0
        self.answer_display_until = 0.0
        self.questions = [
            ("can_speak", "Can you speak?"),
            ("injured", "Are you injured? Where?"),
            ("trapped", "Are you trapped?"),
            ("can_walk", "Can you move or walk?"),
            ("heavy_bleeding", "Are you bleeding heavily?"),
            ("breathing_difficulty", "Are you having difficulty breathing?"),
            ("severe_pain", "Is the pain severe?"),
            ("other_people_nearby", "Are there other people nearby?"),
            ("needs_supply", "Do you need supplies?"),
        ]
        self.chinese_questions = {
            "can_speak": "你能说话吗？",
            "injured": "你受伤了吗？哪里受伤？",
            "trapped": "你被困住了吗？",
            "can_walk": "你能移动或走路吗？",
            "heavy_bleeding": "你有大量出血吗？",
            "breathing_difficulty": "你呼吸困难吗？",
            "severe_pain": "疼痛严重吗？",
            "other_people_nearby": "附近还有其他人吗？",
            "needs_supply": "你需要物资吗？",
        }
        self.voice_mode_instruction = "Please answer YES or NO."
        self.gesture_mode_instruction = (
            "For YES, raise both arms above your head and form a large circle. "
            "For NO, cross both arms in front of your chest."
        )
        self.chinese_voice_mode_instruction = "璇峰彧闇€瑕佸洖绛旀槸鎴栦笉鏄?"
        self.chinese_gesture_mode_instruction = (
            "濡傛灉鏄紝璇峰皢涓ゅ彧鎵嬭噦涓惧埌澶撮《涓婃柟褰㈡垚涓€涓ぇ鍦嗐€? "
            "濡傛灉涓嶆槸锛岃鍦ㄨ兏鍓嶄氦鍙夊弻鑷傘€?"
        )
        self.mode_instruction_played = False
        self.message = "Searching for person..."

    def run(self):
        try:
            consecutive_failures = 0
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"WARNING: camera frame lost ({consecutive_failures})")
                    if consecutive_failures >= 10:
                        print("ERROR: camera frame lost too many times, giving up")
                        break
                    continue
                consecutive_failures = 0

                height, width = frame.shape[:2]
                self.message = ""

                if self.state == "SEARCHING":
                    self._handle_search_state(frame, width)
                elif self.state == "GESTURE_TEST":
                    self._handle_gesture_test(frame)
                elif self.state == "ASKING_VOICE":
                    self._handle_voice_state()
                elif self.state == "SECOND_PROMPT":
                    self._handle_second_prompt()
                elif self.state == "WAITING_FOR_VOICE_OR_GESTURE":
                    self._handle_second_response_window(frame)
                elif self.state == "WAITING_FOR_GESTURE":
                    self._handle_gesture_state(frame)
                elif self.state == "ANSWER_ACCEPTED":
                    self._handle_answer_accepted()
                elif self.state == "NEXT_QUESTION":
                    self._start_next_question()
                elif self.state == "TRIAGE":
                    self._handle_triage_state()
                elif self.state == "REPORTING":
                    self._handle_reporting_state()
                elif self.state == "SENT":
                    self._handle_sent_state()
                elif self.state == "ERROR":
                    self._handle_error_state()

                draw_overlay(frame, self.state, self.message, bbox=self.bbox, status=self.status_text)

                if not self.args.no_display:
                    import cv2

                    cv2.imshow("Device Integration", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if not self.args.gesture_test:
                flush_triage_queue(self.args.base_url, self.args.device_id, self.args.lat, self.args.lon)
            self.camera.release()
            if not self.args.no_display:
                import cv2

                cv2.destroyAllWindows()
            self.classifier.close()
            self.voice_executor.shutdown(wait=False, cancel_futures=True)

    def _handle_gesture_test(self, frame):
        raw_status = self.classifier.classify(frame)
        print(f"DEBUG raw_status={raw_status}")
        stable = self.gesture_stabilizer.add(raw_status)
        self.status_text = raw_status
        stable_text = stable.answer if stable else "waiting"
        self.message = f"Raw: {raw_status or 'unknown'} | Stable: {stable_text}"
        if stable and stable.answer != self.last_gesture_result:
            self.last_gesture_result = stable.answer
            print(f"GESTURE: {stable.answer} confidence={stable.confidence:.2f}")

    def _handle_search_state(self, frame, width):
        self.search_frame_count += 1
        yolo_interval = max(1, self.args.yolo_interval)
        if self.search_frame_count == 1 or self.search_frame_count % yolo_interval == 0:
            self.last_detection = self.detector.find_person(frame)
        found = self.last_detection
        if found:
            x1, y1, x2, y2 = found
            self.bbox = (x1, y1, x2, y2)
            position_ok = True
            if self.args.require_centered:
                person_center = (x1 + x2) / 2
                frame_center = width / 2
                delta = person_center - frame_center
                position_ok = abs(delta) < width * 0.1
                if not position_ok:
                    self.message = "Move camera left" if delta < 0 else "Move camera right"

            if position_ok:
                self.center_frames += 1
                self.message = "Person detected. Stabilizing..."
                if self.center_frames >= self.args.person_stable_frames:
                    self.state = "ASKING_VOICE"
                    self.response_type = None
                    initial_prompt = self.args.prompt_text
                    if self.args.whisper_model and self.args.prompt_text == "Can you hear me?":
                        initial_prompt = "Can you hear me? 你能听见我吗？"
                    self.current_question = ("initial_response", initial_prompt)
                    self.message = "Preparing initial response check..."
                    print("Person found and stable. Starting response check...")
            else:
                self.center_frames = 0
        else:
            self.bbox = None
            self.center_frames = 0
            self.message = "Searching for person..."

    def _handle_voice_state(self):
        key, question = self.current_question
        if self.voice_future is None:
            # Do not display the previous question's accepted answer while
            # listening for a new one.
            self.status_text = None
            self.message = f"Voice: {question}"
            self.voice_future = self.voice_executor.submit(self._ask_voice, key, question)
            return
        if not self.voice_future.done():
            self.message = f"Listening: {question}"
            return

        try:
            result = self.voice_future.result()
        except Exception as exc:
            self.voice.error = f"voice worker failed: {exc}"
            result = VoiceAnswer(None, 0.0, "UNAVAILABLE")
        finally:
            self.voice_future = None

        print(
            f"Voice result: status={result.status} answer={result.answer} "
            f"language={result.language} transcript={result.transcript!r}"
        )

        if result.answer:
            if key == "initial_response":
                self.interaction_language = result.language or "en"
                if result.answer == "NO":
                    self.response_type = "responding"
                    self.interaction_mode = "voice"
                    self._accept_answer(
                        result.answer,
                        "voice",
                        result.confidence,
                        result.status,
                        transcript=result.transcript,
                        next_state="TRIAGE",
                    )
                    return
            self._accept_answer(
                result.answer,
                "voice",
                result.confidence,
                result.status,
                transcript=result.transcript,
            )
            return

        if key == "initial_response":
            self.answer_details.append({
                "question": "initial_voice_response",
                "answer": None,
                "source": "voice",
                "confidence": result.confidence,
                "channel_status": result.status,
                "transcript": result.transcript,
            })
            self.state = "SECOND_PROMPT"
            self.message = "No initial voice response. Preparing second prompt..."
            return

        # Speech mode remains speech-only after the initial response selects
        # it. A missed answer is recorded; it does not switch modes mid-flow.
        if key != "initial_response" and self.interaction_mode == "speech":
            self._handle_answer_timeout(
                transcript=result.transcript,
                channel_status="UNKNOWN" if result.transcript else "NO_RESPONSE",
            )
            return

        self.gesture_stabilizer.reset()
        self.gesture_deadline = time.time() + self.args.gesture_timeout
        self.state = "WAITING_FOR_GESTURE"
        self.message = f"Voice {result.status.lower()}; please answer with a gesture."
        print(f"Voice fallback for {key}: {result.status}; {self.voice.error or ''}")

    def _handle_second_prompt(self):
        self.status_text = None
        self.gesture_stabilizer.reset()
        self.second_window_ready.clear()
        self.second_window_stop.clear()
        self.second_voice_result = None
        self.voice_future = self.voice_executor.submit(self._run_second_response_voice)
        self.state = "WAITING_FOR_VOICE_OR_GESTURE"
        self.message = "Preparing voice and gesture response window..."

    def _run_second_response_voice(self):
        self.voice.say(
            "Can you speak? If you can't, please raise one hand. "
            "你能说话吗？如果不能，请举起一只手。"
        )
        self.second_window_started = time.time()
        self.second_window_ready.set()
        return self.voice.listen_initial(
            self.args.voice_timeout,
            stop_event=self.second_window_stop,
        )

    def _handle_second_response_window(self, frame):
        if not self.second_window_ready.is_set():
            self.message = "Speaking second prompt..."
            return

        self.message = "Listening for voice and watching for one-hand gesture..."

        if self.voice_future is not None and self.voice_future.done():
            try:
                self.second_voice_result = self.voice_future.result()
            except Exception as exc:
                self.voice.error = f"second response worker failed: {exc}"
                self.second_voice_result = VoiceAnswer(None, 0.0, "UNAVAILABLE")
            self.voice_future = None

        result = self.second_voice_result
        if result is not None and result.answer:
            self.second_window_stop.set()
            self.interaction_language = result.language or "en"
            self.answers["can_speak"] = True
            self.answer_details.append({
                "question": "can_speak",
                "answer": "YES",
                "source": "voice",
                "confidence": result.confidence,
                "channel_status": result.status,
                "transcript": result.transcript,
            })
            self._accept_answer(
                result.answer,
                "voice",
                result.confidence,
                result.status,
                transcript=result.transcript,
            )
            self.question_index = 0
            return

        detected_status = self.classifier.classify(frame)
        stable = self.gesture_stabilizer.add(detected_status)
        self.status_text = detected_status
        if stable:
            self.second_window_stop.set()
            self.interaction_language = "bilingual" if self.args.whisper_model else "en"
            self.answers["can_speak"] = False
            self.answer_details.append({
                "question": "can_speak",
                "answer": "NO",
                "source": "gesture",
                "confidence": stable.confidence,
                "channel_status": "STABLE",
                "transcript": "",
            })
            self._accept_answer(stable.answer, "gesture", stable.confidence, "STABLE")
            self.question_index = 0
            return

        if time.time() - self.second_window_started >= self.args.voice_timeout:
            self.second_window_stop.set()
            self.response_type = "unresponsive"
            self.interaction_mode = None
            self.status_text = "no_response"
            self.no_response_reason = "NO_VOICE_OR_GESTURE_RESPONSE"
            self.answer_details.append({
                "question": "second_response_window",
                "answer": None,
                "source": "none",
                "confidence": 0.0,
                "channel_status": "NO_RESPONSE",
                "transcript": getattr(result, "transcript", "") if result else "",
            })
            self.state = "TRIAGE"

    def _ask_voice(self, key, question):
        """Run blocking audio I/O away from the camera/display loop."""
        self.voice.say(question)
        if key != "initial_response" and self.interaction_mode == "gesture":
            return VoiceAnswer(None, 0.0, "GESTURE_MODE")
        if self.args.demo:
            scripted = {
                "initial_response": "YES",
                "can_walk": "YES",
                "heavy_bleeding": "NO",
                "breathing_difficulty": "NO",
                "trapped": "NO",
                "severe_pain": "NO",
                "needs_supply": "YES",
            }
            answer = scripted[key]
            return VoiceAnswer(answer, 1.0, "DEMO", answer.lower())
        if key == "initial_response":
            result = self.voice.listen_initial(self.args.voice_timeout)
        else:
            result = self.voice.listen_yes_no_in_language(
                self.args.voice_timeout,
                self.interaction_language or "en",
            )
        return result

    def _handle_gesture_state(self, frame):
        key, _ = self.current_question
        detected_status = self.classifier.classify(frame)
        stable = self.gesture_stabilizer.add(detected_status)
        self.status_text = detected_status
        self.message = f"Gesture answer for {key}: waiting for a stable response..."
        if stable:
            self._accept_answer(stable.answer, stable.source, stable.confidence, "STABLE")
            return
        if time.time() >= self.gesture_deadline:
            self._handle_answer_timeout()

    def _accept_answer(
        self,
        answer,
        source,
        confidence,
        channel_status,
        transcript="",
        next_state="NEXT_QUESTION",
    ):
        key, _ = self.current_question
        self.answer_details.append({
            "question": key,
            "answer": answer,
            "source": source,
            "confidence": confidence,
            "channel_status": channel_status,
            "transcript": transcript,
        })
        if key == "initial_response":
            self.response_type = "responding"
            self.interaction_mode = "speech" if source == "voice" else "gesture"
            if source == "gesture" and self.interaction_language is None:
                self.interaction_language = "bilingual" if self.args.whisper_model else "en"
            self.question_index = -1
        else:
            self.answers[key] = answer == "YES" if answer in ("YES", "NO") else None
        if answer == "HELP":
            self.resource_requested = True
        if key == "needs_supply" and answer == "YES":
            self.resource_requested = True
        self.status_text = answer.lower()
        self.answer_display_until = time.time() + self.args.answer_display_seconds
        self.after_answer_state = next_state
        self.state = "ANSWER_ACCEPTED"
        self.message = f"Accepted {answer} via {source}."

    def _handle_answer_accepted(self):
        self.message = f"Accepted {self.status_text.upper()}."
        if time.time() >= self.answer_display_until:
            self.state = self.after_answer_state

    def _handle_answer_timeout(self, transcript="", channel_status="NO_RESPONSE"):
        key, _ = self.current_question
        self.answer_details.append({
            "question": key,
            "answer": None,
            "source": "none",
            "confidence": 0.0,
            "channel_status": channel_status,
            "transcript": transcript,
        })
        if key == "initial_response":
            self.response_type = "unresponsive"
            self.state = "TRIAGE"
            self.message = "No voice or gesture response detected. Priority review required."
        else:
            self.answers[key] = None
            self.state = "NEXT_QUESTION"

    def _start_next_question(self):
        self.question_index += 1
        if self.question_index >= len(self.questions):
            self.state = "TRIAGE"
            return
        if self.question_index == 0 and not self.mode_instruction_played:
            self._play_mode_instruction()
        self.current_question = self.questions[self.question_index]
        key, english_question = self.current_question
        if self.interaction_language == "zh":
            self.current_question = (key, self.chinese_questions[key])
        elif self.interaction_language == "bilingual":
            self.current_question = (key, f"{english_question} {self.chinese_questions[key]}")
        self.state = "ASKING_VOICE"
        self.message = f"Next question: {self.current_question[1]}"

    def _play_mode_instruction(self):
        if self.interaction_mode == "gesture":
            if self.interaction_language == "zh":
                instruction = self.chinese_gesture_mode_instruction
            elif self.interaction_language == "bilingual":
                instruction = f"{self.gesture_mode_instruction} {self.chinese_gesture_mode_instruction}"
            else:
                instruction = self.gesture_mode_instruction
        else:
            if self.interaction_language == "zh":
                instruction = self.chinese_voice_mode_instruction
            elif self.interaction_language == "bilingual":
                instruction = f"{self.voice_mode_instruction} {self.chinese_voice_mode_instruction}"
            else:
                instruction = self.voice_mode_instruction
        self.voice.say(instruction)
        self.message = instruction
        self.mode_instruction_played = True

    def _handle_triage_state(self):
        can_walk = self.answers.get("can_walk")
        heavy_bleeding = self.answers.get("heavy_bleeding") is True
        breathing_difficulty = self.answers.get("breathing_difficulty") is True
        trapped = self.answers.get("trapped") is True
        injured = self.answers.get("injured") is True
        severe_pain = self.answers.get("severe_pain") is True

        # Unknown mobility is not itself a diagnosis. The existing assessment
        # format requires a bool, so default to mobile and retain UNKNOWN in
        # answer_details for rescuer review.
        can_walk_for_rules = can_walk is not False

        self.assessment = build_triage_assessment(
            response_type=self.response_type or "responding",
            can_walk=can_walk_for_rules,
            heavy_bleeding=heavy_bleeding,
            breathing_difficulty=breathing_difficulty,
            trapped=trapped,
            severe_pain=severe_pain,
            needs_supply=(
                self.answers.get("needs_supply") is True
                or self.resource_requested
            ),
        )
        self.assessment["response_detected"] = self.response_type == "responding"
        self.assessment["interaction_mode"] = self.interaction_mode
        self.assessment["interaction_language"] = self.interaction_language
        self.assessment["answers"] = self.answer_details
        self.assessment["can_walk"] = can_walk
        self.assessment["can_speak"] = self.answers.get("can_speak")
        self.assessment["injured"] = self.answers.get("injured")
        self.assessment["severe_pain"] = self.answers.get("severe_pain")
        self.assessment["other_people_nearby"] = self.answers.get("other_people_nearby")
        self.assessment["needs_supply"] = self.answers.get("needs_supply")
        if injured:
            self.assessment["reasons"].append("injury reported")
        if severe_pain:
            self.assessment["reasons"].append("severe pain")
        no_response_questions = [
            item["question"]
            for item in self.answer_details
            if item.get("channel_status") == "NO_RESPONSE"
        ]
        self.assessment["no_response_questions"] = no_response_questions
        self.assessment["review_status"] = (
            "PRIORITY_REVIEW" if no_response_questions else "COMPLETE"
        )
        if no_response_questions:
            self.assessment["review_reason"] = "NO_RESPONSE"
            for question in no_response_questions:
                reason = f"no response to {question}"
                if reason not in self.assessment["reasons"]:
                    self.assessment["reasons"].append(reason)
        if self.no_response_reason:
            # Silence is an observation for responder review, not a diagnosis.
            self.assessment["priority"] = "no_response"
            self.assessment["review_status"] = "PRIORITY_REVIEW"
            self.assessment["review_reason"] = self.no_response_reason
            self.assessment["diagnosis"] = None
        if self.no_response_reason:
            self.status_text = "no_response"
        elif self.assessment["review_status"] == "PRIORITY_REVIEW":
            self.status_text = "result"
        else:
            self.status_text = self.assessment["priority"]
        self.state = "REPORTING"
        self.result_time = time.time()
        print(f"Assessment: {self.assessment}")
        self.message = "Preparing to send triage report..."

    def _handle_reporting_state(self):
        success, response = post_triage_assessment(
            self.args.base_url,
            self.args.device_id,
            self.assessment,
            self.args.lat,
            self.args.lon,
        )
        self.state = "SENT" if success else "ERROR"
        self.result_time = time.time()
        print(f"POST triage -> {response}")
        self.message = "Triage report sent" if success else f"Offline queue saved: {response}"

    def _handle_sent_state(self):
        self.message = f"Sent {self.assessment['priority']} assessment. Resetting in 3 seconds..."
        if time.time() - self.result_time > 3.0:
            self._reset_after_cycle()

    def _handle_error_state(self):
        self.message = "Network issue. Saving offline and retrying soon..."
        if time.time() - self.result_time > 3.0:
            self._reset_after_cycle()

    def _reset_after_cycle(self):
        self.state = "SEARCHING"
        self.assessment = None
        self.status_text = None
        self.bbox = None
        self.response_type = None
        self.interaction_mode = None
        self.interaction_language = None
        self.no_response_reason = None
        self.second_window_ready.clear()
        self.second_window_stop.set()
        self.second_voice_result = None
        self.current_question = None
        self.question_index = -1
        self.answers = {}
        self.answer_details = []
        self.resource_requested = False
        self.mode_instruction_played = False
        self.gesture_stabilizer.reset()
        self.voice_future = None
        self.search_frame_count = 0
        self.last_detection = None
        self.center_frames = 0


def build_parser():
    parser = argparse.ArgumentParser(description="Device integration runner")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BACKEND_URL", "http://10.88.56.244:5000"),
        help="Dashboard API base URL",
    )
    parser.add_argument("--device-id", default="DR-01", help="Device ID to report")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--lat", type=float, default=-27.4698, help="Report latitude")
    parser.add_argument("--lon", type=float, default=153.0251, help="Report longitude")
    parser.add_argument("--no-display", action="store_true", help="Do not show the OpenCV preview window")
    parser.add_argument("--prompt-text", default="Can you hear me?", help="Initial spoken response-check question")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    parser.add_argument("--demo", action="store_true", help="Use a deterministic demo assessment flow")
    parser.add_argument("--no-voice", action="store_true", help="Disable offline voice and use gesture fallback")
    parser.add_argument("--vosk-model", default=None, help="Path to an unpacked offline Vosk model")
    parser.add_argument("--audio-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--whisper-model", default=None, help="Local multilingual faster-whisper model path/name")
    parser.add_argument("--voice-timeout", type=float, default=5.0, help="Seconds for initial and second response windows")
    parser.add_argument("--gesture-timeout", type=float, default=5.0, help="Seconds to wait for a stable gesture")
    parser.add_argument("--answer-display-seconds", type=float, default=1.0, help="Seconds to show accepted answer colour")
    parser.add_argument("--yolo-interval", type=int, default=3, help="Run YOLO once every N search frames")
    parser.add_argument("--person-stable-frames", type=int, default=3, help="Detections required before assessment")
    parser.add_argument("--require-centered", action="store_true", help="Require the person to be near frame centre")
    parser.add_argument("--gesture-test", action="store_true", help="Test gestures live without YOLO, voice, or reporting")
    return parser


def parse_args():
    return build_parser().parse_args()


def main():
    runner = DeviceTriageRunner(parse_args())
    runner.run()


if __name__ == "__main__":
    main()
