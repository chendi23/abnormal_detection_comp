sum=0
for i in range(2):
    for j in range(5):
        sum+=j
        if j==4:
            break
print("sum",sum)