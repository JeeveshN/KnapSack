# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
import os

def get_suggesions(courses):
	dirr =  os.getcwd()
	mit = open(dirr + "/Parsers/MIT_reduced.txt","r")
	mit_data = {}
	for line in mit:
		mit_data[line.strip("\n").lower()] = 1
	for course in courses:
		if(mit_data.has_key(course.lower())):
			del mit_data[course.lower()]
		else:
			for word in course.split():
				if(mit_data.has_key(word.lower())):
					del mit_data[word.lower()]
	return mit_data.keys()


# Create your views here.
def index(request):
	return render(request,"index.html")

def contact(request):
	return render(request,"contact.html")

def analyse(request):
	if request.method == "POST":
		raw_data = request.POST["message"]
		courses = raw_data.split("\n")
		suggested = get_suggesions(courses)
		print suggested
		return render(request,"result.html",{'suggesion':suggested,'score': 86})
