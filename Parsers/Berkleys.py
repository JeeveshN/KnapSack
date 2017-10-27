from bs4 import BeautifulSoup
import requests
import time
import re 

BASE_URL = "https://www2.eecs.berkeley.edu/Courses/CS/?_ga=2.44961508.1522982388.1509131899-838894781.1509131899"
COURSES = []

def get_courses():
	doc = requests.get(BASE_URL)
	soup = BeautifulSoup(doc.text,"html.parser")
	content = soup.find("div","content")
	course_li = content.find_all("li")
	for course in course_li:
		try:
			COURSES.append(re.findall(r"\. (.*)",course.contents[1].string)[0])
		except:
			pass

	with open('Berkley.txt','w')as f:f.write("\n".join(COURSES).strip('\n'))



if __name__ == "__main__":
	get_courses()
