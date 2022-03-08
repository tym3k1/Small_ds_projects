sum = 30000
x = 400
i = 0
while sum < 120000:
	sum += x
	i = i + 1
	if i%12==0:
		sum +=(sum/10)
	print('kwota: ', sum, 'miesiac :', i)