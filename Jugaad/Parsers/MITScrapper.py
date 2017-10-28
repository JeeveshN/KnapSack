import csv
import urllib2
from bs4 import BeautifulSoup
quote_page = 'http://catalog.mit.edu/subjects/6/'
page = urllib2.urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
data = []
for link in soup.find_all('strong'):
	name = link.text.strip()
	data.append(name)

with open('idealSyllabus.csv', 'a') as csv_file:
	writer = csv.writer(csv_file)
	for name in data:
 		writer.writerow([name])