mit = open("../Parsers/MIT_reduced.txt","r")
mit_data = {}
for line in mit:
	mit_data[line.strip("\n").lower()] = 1
print mit_data
sample = open("../Parsers/IGDTUW.txt","r")

for line in sample:
	if(mit_data.has_key(line.strip('\n').lower())):
		del mit_data[line.strip('\n').lower()]
	else:
		for word in line.strip('\n').split():
			if(mit_data.has_key(word.lower())):
				del mit_data[word.lower()]

print "\n"
print mit_data.keys()