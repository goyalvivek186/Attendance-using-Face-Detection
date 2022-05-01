# from asyncore import file_dispatcher
# import asyncore
# from audioop import mul
import cv2
import os
from cv2 import COLOR_BGR2RGB
import numpy as np
from attendance_model import Attendance
import face_recognition
from datetime import datetime
# import pandas as pd
# asyncore.dispatcher.

path = "Attendance images"  #dataset
images = []
names = []
myList = os.listdir(path)   #names of all the images for attendance check
for name in myList:          #cls = name of a file
	curImg = cv2.imread(f'{path}/{name}')      #read an image at once
	images.append(curImg)
	names.append(os.path.splitext(name)[0]) #first names

print("Images scanned")
print("Names noted")

ATTENDANCE_FILE = 'attendance_marked.csv'
currentAttendanceList = []
f = open(ATTENDANCE_FILE, 'r')
attendanceList = f.readlines()
f.close()

for i in range(1, len(attendanceList)):
	attendanceRecord = Attendance.create_from_line(attendanceList[i])
	if attendanceRecord:
		currentAttendanceList.append(attendanceRecord)


def updateAttendaceList():
	f = open(ATTENDANCE_FILE, 'w')
	f.write('Name,Time\n')
	for i in range(len(currentAttendanceList)):
		f.write(currentAttendanceList[i].get_row())
		if i != len(currentAttendanceList) - 1:
			f.write('\n')
	f.close()
	print('Attendance Updated')


def addAttendance(name):
	names = [x.get_name() for x in currentAttendanceList]
	if name not in names:
		currentAttendanceList.append(Attendance(name))


def findEncoding(images):
	encodings = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encodes = face_recognition.face_encodings(img)[0]
		#[0] as face_encodings return 128 dim face encoding for all faces
		#we need only one, hence oth index
		encodings.append(encodes)
	return encodings



allEncoding = findEncoding(images)
print("Encoding calculated")


#Get the students image via web cam to check in the dataset present
print("Opening camera")
cam = cv2.VideoCapture(0)

while True:
	success, Cimage = cam.read()
	
	image = cv2.resize(Cimage, (0, 0), None, 0.25, 0.25)
	# cv2.imshow("Image Captured", image)
	# cv2.waitKey(100)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	currfaceLocs = face_recognition.face_locations(image)   #all faces present in the frames
	currFaceEncodings = face_recognition.face_encodings(image, currfaceLocs)

	for loc, enc in zip(currfaceLocs, currFaceEncodings):
		matches = face_recognition.compare_faces(allEncoding, enc)  #bool
		faceDis = face_recognition.face_distance(allEncoding, enc)
		matchIdx = np.argmin(faceDis) #min face dis is our ans

		if matches[matchIdx]:
			name = names[matchIdx].upper()
			print(name)
			addAttendance(name)
			y1, x2, y2, x1 = loc
			y1 *= 4
			y2 *= 4
			x2 *= 4
			x1 *= 4
			cv2.rectangle(Cimage, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.rectangle(Cimage, (x1, y2-35), (x2, y2), (0, 225, 0), cv2.FILLED)
			cv2.putText(Cimage, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
	
	
	cv2.imshow("Image Captured", Cimage)
	if (cv2.waitKey(100)  == ord('q') or cv2.waitKey(100)  ==  ord('Q')):
		print("Exitted")
		break

updateAttendaceList()
