# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
def index(request):
	return render(request,"index.html")

def contact(request):
	return render(request,"contact.html")

def analyse(request):
	if request.method == "POST":
		raw_data = request.POST["message"]
		courses = raw_data.split("\n")
		print courses