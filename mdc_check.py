import re

yf_file = open('yaofang.txt', 'r', encoding='utf-8').readlines()

for i in range(len(yf_file)):
    if re.search('\d+',yf_file[i]):
        print(yf_file[i])
