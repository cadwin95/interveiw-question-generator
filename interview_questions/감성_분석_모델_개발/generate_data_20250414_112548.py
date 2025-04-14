# 필요한 패키지
import pandas as pd # Although defined in the template, not strictly needed for JSON output
import numpy as np
import random
import logging
import os
import json # For saving data in JSON format
from datetime import datetime

# 로깅 설정
log_filename = f"data_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # Ensure UTF-8 for logs
        logging.StreamHandler()
    ]
)

class EmotionPaletteDataGenerator:
    """
    Generates synthetic data mimicking the output of the 'Emotion Palette' system.
    This includes input text, simulated analyzed emotions, and structured visualization data.
    """
    def __init__(self, num_samples=10):
        """
        Initializes the generator.

        Args:
            num_samples (int): The number of data samples to generate.
        """
        self.logger = logging.getLogger(__name__)
        self.num_samples = num_samples
        # Define the emotion set based on Ekman's basic emotions
        self.emotions = ['joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
        # Sample sentences with potential primary/secondary emotions
        self.sample_texts = [
            ("오늘 발표는 정말 긴장됐지만, 끝나고 나니 후련하고 뿌듯하다!", ['fear', 'joy'], ['surprise']),
            ("기대했던 영화인데 생각보다 지루해서 실망했고, 옆자리 관객이 시끄러워서 짜증도 났다.", ['sadness', 'anger'], ['disgust']),
            ("갑자기 친구가 나타나서 깜짝 놀랐지만, 너무 반가웠어!", ['surprise', 'joy'], []),
            ("길에서 지갑을 잃어버려서 너무 슬프고 불안하다. 어떻게 해야 할지 모르겠어.", ['sadness', 'fear'], []),
            ("내 아이디어를 팀장이 가로채서 발표했을 때 정말 화가 났고 역겨웠다.", ['anger', 'disgust'], ['sadness']),
            ("오랜만에 가족들과 함께 맛있는 저녁을 먹으니 행복하고 편안하다.", ['joy'], []),
            ("무서운 영화를 봤더니 밤에 잠을 잘 못 자겠어. 계속 이상한 소리가 들리는 것 같아.", ['fear'], ['surprise']),
            ("음식이 상한 것 같아. 냄새도 이상하고 보기만 해도 속이 안 좋다.", ['disgust'], ['sadness']),
            ("로또에 당첨되다니! 믿을 수가 없어! 너무 기쁘고 놀랍다!", ['joy', 'surprise'], []),
            ("노력한 만큼 결과가 나오지 않아 속상하고, 나 자신에게 화가 난다.", ['sadness', 'anger'], [])
        ]
        # Define mappings for visualization rules
        self.emotion_color_map = {
            'joy': (255, 223, 0),    # Gold/Yellow
            'sadness': (0, 110, 200),  # Blue
            'anger': (220, 20, 60),    # Crimson Red
            'surprise': (255, 165, 0), # Orange
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 0),     # Green
            'default': (128, 128, 128) # Gray for neutral/mixed states
        }
        self.emotion_element_map = {
            'joy': 'sparkle',
            'sadness': 'tear',
            'anger': 'spike',
            'surprise': 'burst',
            'fear': 'tremble_line',
            'disgust': 'ooze'
        }

    def _generate_single_emotion_profile(self, primary_emotions, secondary_emotions):
        """Generates a randomized emotion intensity profile."""
        profile = {emo: 0.0 for emo in self.emotions}

        # Assign high intensity to primary emotions
        for emo in primary_emotions:
            if emo in profile:
                profile[emo] = round(random.uniform(0.5, 1.0), 2)

        # Assign medium intensity to secondary emotions
        for emo in secondary_emotions:
            if emo in profile and profile[emo] == 0.0: # Avoid overwriting primary
                 profile[emo] = round(random.uniform(0.2, 0.6), 2)

        # Assign low background intensity to others
        remaining_emotions = [emo for emo in self.emotions if profile[emo] == 0.0]
        num_others_to_activate = random.randint(0, len(remaining_emotions))
        emotions_to_activate = random.sample(remaining_emotions, num_others_to_activate)
        for emo in emotions_to_activate:
             profile[emo] = round(random.uniform(0.05, 0.25), 2)

        # Ensure at least one emotion has non-zero intensity if all were initially zero
        if all(v == 0.0 for v in profile.values()):
            chosen_emo = random.choice(self.emotions)
            profile[chosen_emo] = round(random.uniform(0.3, 0.7), 2)

        return profile

    def _map_emotions_to_visualization(self, emotions):
        """Applies creative rules to map emotions to visualization data."""
        if not emotions:
            return {}

        # Find dominant emotion(s)
        sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        primary_emotion, primary_intensity = sorted_emotions[0]

        # Rule 1: Base Color - based on dominant emotion, alpha by intensity
        base_rgb = self.emotion_color_map.get(primary_emotion, self.emotion_color_map['default'])
        alpha = max(0.3, primary_intensity) # Ensure minimum visibility
        base_color = f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {alpha:.2f})"

        # Rule 2: Elements - Add elements for emotions above a threshold
        elements = []
        threshold = 0.15
        for emo, intensity in emotions.items():
            if intensity >= threshold:
                element_type = self.emotion_element_map.get(emo, 'dot')
                element = {
                    "type": element_type,
                    "intensity": round(intensity, 2)
                }
                # Add specific attributes based on element type and intensity
                if element_type == 'sparkle':
                    element["count"] = int(intensity * 8) + 1
                    element["brightness"] = round(intensity, 2)
                elif element_type == 'tear':
                    element["size"] = int(intensity * 10) + 2
                    element["color"] = f"rgba(0, 0, 255, {max(0.2, intensity):.2f})"
                elif element_type == 'spike':
                    element["count"] = int(intensity * 6) + 1
                    element["sharpness"] = round(intensity, 2)
                elif element_type == 'burst':
                     element["radius"] = int(intensity * 15) + 3
                elif element_type == 'tremble_line':
                     element["count"] = int(intensity * 5) + 1
                     element["amplitude"] = round(intensity * 5, 2)
                elif element_type == 'ooze':
                     element["area"] = int(intensity * 20) + 5
                     element["speed"] = round(intensity * 0.5, 2)

                elements.append(element)

        # Rule 3: Background Effect - simple gradient based on overall positive/negative balance
        positive_intensity = emotions.get('joy', 0.0) + emotions.get('surprise', 0.0) * 0.5 # Surprise can be +/-
        negative_intensity = emotions.get('sadness', 0.0) + emotions.get('anger', 0.0) + \
                             emotions.get('fear', 0.0) + emotions.get('disgust', 0.0)

        if positive_intensity > negative_intensity * 1.2:
            background_effect = {
                "type": "bright_gradient",
                "start_color": "#FFFACD", # LemonChiffon
                "end_color": "#FFEFD5"    # PapayaWhip
            }
        elif negative_intensity > positive_intensity * 1.2:
             background_effect = {
                "type": "dark_gradient",
                "start_color": "#B0C4DE", # LightSteelBlue
                "end_color": "#778899"    # LightSlateGray
            }
        else:
             background_effect = {
                "type": "neutral_subtle",
                "start_color": "#F5F5F5", # WhiteSmoke
                "end_color": "#E0E0E0"    # Light Gray
            }

        # Rule 4: Description
        description = f"Primarily {primary_emotion} ({primary_intensity:.2f})"
        if len(sorted_emotions) > 1 and sorted_emotions[1][1] >= threshold:
            secondary_emotion, secondary_intensity = sorted_emotions[1]
            description += f" with significant {secondary_emotion} ({secondary_intensity:.2f})."
        elif len(sorted_emotions) > 1 and sorted_emotions[1][1] >= 0.1: # Hint of emotion
             secondary_emotion, secondary_intensity = sorted_emotions[1]
             description += f" with a hint of {secondary_emotion} ({secondary_intensity:.2f})."
        else:
            description += "."


        visualization_data = {
            "base_color": base_color,
            "elements": elements,
            "background_effect": background_effect,
            "description": description
        }
        return visualization_data

    def generate_data(self):
        """Generates the specified number of data samples."""
        generated_data = []
        try:
            self.logger.info(f"데이터 생성 시작: {self.num_samples}개 샘플 생성")
            for i in range(self.num_samples):
                # Cycle through sample texts or pick randomly
                text, primary_emo, secondary_emo = self.sample_texts[i % len(self.sample_texts)]

                # Simulate emotion analysis
                analyzed_emotions = self._generate_single_emotion_profile(primary_emo, secondary_emo)

                # Generate visualization data based on analyzed emotions
                visualization_data = self._map_emotions_to_visualization(analyzed_emotions)

                # Combine into the final structure
                data_point = {
                    "id": f"sample_{i+1:03d}", # Add a unique ID
                    "input_text": text,
                    "analyzed_emotions": analyzed_emotions,
                    "visualization_data": visualization_data
                }
                generated_data.append(data_point)
                if (i + 1) % 10 == 0:
                     self.logger.info(f"{i+1}/{self.num_samples}개 샘플 생성 완료...")

            self.logger.info("데이터 생성 완료")
            return generated_data

        except Exception as exc:
            self.logger.error(f"데이터 생성 중 오류 발생: {str(exc)}", exc_info=True)
            raise

    def validate_data(self, data):
        """Validates the structure and basic types of the generated data."""
        try:
            self.logger.info(f"데이터 검증 시작: {len(data)}개 샘플 검증")
            if not isinstance(data, list):
                raise TypeError("생성된 데이터가 리스트 형식이 아닙니다.")

            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(f"데이터 샘플 {i}가 사전(dict) 형식이 아닙니다.")

                # Check required top-level keys
                required_keys = ["id", "input_text", "analyzed_emotions", "visualization_data"]
                if not all(key in item for key in required_keys):
                    raise ValueError(f"데이터 샘플 {i}에 필수 키가 누락되었습니다: {required_keys}")

                # Validate 'analyzed_emotions' structure
                if not isinstance(item["analyzed_emotions"], dict):
                     raise TypeError(f"데이터 샘플 {i}의 'analyzed_emotions'가 사전(dict)이 아닙니다.")
                if not all(emo in self.emotions for emo in item["analyzed_emotions"].keys()):
                     self.logger.warning(f"데이터 샘플 {i}의 'analyzed_emotions'에 정의되지 않은 감정 키가 포함될 수 있습니다.")
                if not all(isinstance(val, (float, int)) and 0.0 <= val <= 1.0 for val in item["analyzed_emotions"].values()):
                     raise ValueError(f"데이터 샘플 {i}의 'analyzed_emotions' 값은 0.0과 1.0 사이의 float이어야 합니다.")

                # Validate 'visualization_data' structure (basic checks)
                vis_data = item["visualization_data"]
                if not isinstance(vis_data, dict):
                     raise TypeError(f"데이터 샘플 {i}의 'visualization_data'가 사전(dict)이 아닙니다.")
                required_vis_keys = ["base_color", "elements", "background_effect", "description"]
                if not all(key in vis_data for key in required_vis_keys):
                    raise ValueError(f"데이터 샘플 {i}의 'visualization_data'에 필수 키가 누락되었습니다: {required_vis_keys}")
                if not isinstance(vis_data["base_color"], str) or not vis_data["base_color"].startswith("rgba("):
                     self.logger.warning(f"데이터 샘플 {i}의 'base_color' 형식이 예상과 다를 수 있습니다 (예: rgba(...)).")
                if not isinstance(vis_data["elements"], list):
                     raise TypeError(f"데이터 샘플 {i}의 'elements'는 리스트여야 합니다.")
                if not isinstance(vis_data["background_effect"], dict):
                     raise TypeError(f"데이터 샘플 {i}의 'background_effect'는 사전(dict)이어야 합니다.")
                if not isinstance(vis_data["description"], str):
                     raise TypeError(f"데이터 샘플 {i}의 'description'은 문자열이어야 합니다.")

            self.logger.info("데이터 검증 완료: 모든 샘플이 기본 구조를 만족합니다.")
            return True

        except (TypeError, ValueError) as exc:
            self.logger.error(f"데이터 검증 중 오류 발생: {str(exc)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 검증 중 예상치 못한 오류 발생: {str(exc)}", exc_info=True)
            raise

    def save_data(self, data, filename="emotion_palette_data.json"):
        """Saves the generated data to a JSON file."""
        try:
            self.logger.info(f"데이터 저장 시작: '{filename}' 파일로 저장 중...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

            with open(filename, 'w', encoding='utf-8') as f:
                # Use json.dump for better readability and handling non-ASCII chars
                json.dump(data, f, ensure_ascii=False, indent=4)

            self.logger.info(f"데이터 저장 완료: {len(data)}개 샘플이 '{filename}'에 저장되었습니다.")

        except IOError as exc:
            self.logger.error(f"파일 쓰기 오류 발생 ({filename}): {str(exc)}")
            raise
        except Exception as exc:
            self.logger.error(f"데이터 저장 중 예상치 못한 오류 발생: {str(exc)}", exc_info=True)
            raise

def main():
    """Main function to generate, validate, and save the data."""
    try:
        num_data_points = 50 # 생성할 데이터 샘플 수
        output_filename = "synthetic_emotion_visualization_data.json"

        generator = EmotionPaletteDataGenerator(num_samples=num_data_points)
        data = generator.generate_data()

        if data: # Only proceed if data generation was successful
            generator.validate_data(data)
            generator.save_data(data, filename=output_filename)
            logging.info("데이터 생성, 검증 및 저장 완료")
        else:
            logging.warning("생성된 데이터가 없습니다. 저장 및 검증을 건너<0xEB><0x9B><0x8D>니다.")

    except Exception as exc:
        # The logger in the class methods already logs specifics
        logging.critical(f"프로그램 실행 중 치명적 오류 발생: {str(exc)}")
        # raise # Optionally re-raise if needed upstream

if __name__ == "__main__":
    main()