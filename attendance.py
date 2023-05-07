import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime
import xlsxwriter
import xlwt
from xlwt import Workbook
import time

start_time = time.time()

video_capture = cv2.VideoCapture(0)

# Writing to an excel
# sheet using Python
import xlwt
from xlwt import Workbook

# Workbook is created
wb = Workbook()
# add_sheet is usd to create sheet.
sheet1 = wb.add_sheet('attendance')



image1 = face_recognition.load_image_file("images/limay.jpg")
image1_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file("images/ninad.jpg")
image2_encoding = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file("images/sam.png")
image3_encoding = face_recognition.face_encodings(image3)[0]

image4 = face_recognition.load_image_file("images/nilo.jpg")
image4_encoding = face_recognition.face_encodings(image4)[0]

known_face_encoding = [image1_encoding,image2_encoding,image3_encoding,image4_encoding]
known_face_names = ["Limay", "Ninad", "Samiksha", "Nilanshu"]
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%d-%m-%y")

sheet1.write(0, 0, 'Attendance')
# f = current_date+'.xlsx'
sheet1.write(2, 0, 'Name')
sheet1.write(2, 1, 'Time')
row=3
col=0
while True:
    _, frame= video_capture.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, current_date,(450, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    
                    cv2.putText(frame, name+' present',(50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                    now = datetime.now()
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S hrs")
                    print(name,": ", current_time)
                    sheet1.write(row, col, name)
                    sheet1.write(row, col + 1, current_time)
                    row += 1
                    
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        break    
    
    cv2.imshow("Attendance System", frame)       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
sheet1.write(row+1, 0, 'No. of Present')
sheet1.write(row+1, 1, row-3)
filename = current_date+'.xls'
wb.save(filename)
video_capture.release()
cv2.destroyAllWindows()

