# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
import os
import random
import f2

REDUCED = ['MIT_reduced.txt','Berkley_reduced.txt','ETH Zurich_reduced.txt']
def get_suggesions(courses):
	dirr =  os.getcwd()
	ideal = REDUCED[random.randint(0,2)]
	mit = open(dirr + "/Parsers/" + ideal,"r")
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
		a,b,c = f2.get_result(raw_data)
		#return render(request,"result.html",{'suggesion':suggested,'score1': round(a,1),'score2': round(b,1),'score3': round(c,1)})
		return render(request,"result.html",{'suggesion':suggested,'score1': int(a),'score2': int(b),'score3': int(c)})
