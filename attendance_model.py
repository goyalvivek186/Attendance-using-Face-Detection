from datetime import datetime

class Attendance:
	def __init__(self, name, time = None) -> None:
		self.name = name
		if time == None:
			self.time = datetime.now().strftime('%H:%M:%S')
		else:
			self.time = time
	
	def get_row(self):
		return f'{self.name},{self.time}'
	
	def create_from_line(line : str):
		if len(line) == 0 or line[0] == ',' or line[0] == '\n':
			return None
		line = line.split(',')
		return Attendance(line[0], line[1].split('\n')[0])

	def __str__(self) -> str:
		return f'{self.name} -> {self.time}'

	def get_name(self):
		return self.name