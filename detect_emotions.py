import os
import cv2
import glob
from fer import FER
import pandas as pd

final_dict = dict()
final_dict["File name"] = []
final_dict["emotion"] = []
final_dict["score"] = []

current_path = os.getcwd()
detector = FER(mtcnn=True)
for file in glob.glob(current_path+"/images/*.jpg"):
  img = img = cv2.imread(file)
  emotion, score = detector.top_emotion(img)
  file_name = file.split("/")[-1]
  final_dict["File name"].append(file_name)
  final_dict["emotion"].append(emotion)
  final_dict["score"].append(score)

df = pd.DataFrame(final_dict)
df.to_csv("File_vs_emotions.csv",index=False)
