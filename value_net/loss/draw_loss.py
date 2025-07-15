from matplotlib import pyplot as plt

with open(r"E:\code_list\red_battle_blue\value_net\loss\loss.txt","r") as file:
    lines = file.readlines()

array = []

for line in lines:
    t = float(line.strip())
    array.append(t)

fig, ax = plt.subplots()

ax.plot(array[0:])
plt.show()