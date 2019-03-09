import csv

def txt_to_csv(filename,start_row):
    txt = open(filename,'r')
    csv = open(filename[:-4]+'.csv','w')
    count = 1
    for line in txt:
        if count >= start_row:
            csv.write(line.replace("\t",","))
        count+=1
    csv.close()

# txt_to_csv("GSE74923.txt")
