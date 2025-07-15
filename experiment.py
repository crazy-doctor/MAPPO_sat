import numpy as np
import matplotlib.pyplot as plt
from vpython import curve

plt.figure()
data1 = (np.load(file=r"E:\code_list\red_battle_blue\results\file_record\no_assign.npy"))
data2 = (np.load(file=r"E:\code_list\red_battle_blue\results\file_record\has_assign.npy"))
plt.plot(data1,label="distance_assign")
plt.plot(data2,label="value_assign")
plt.legend()
plt.show()
aver_no = sum(data1)/data1.shape[0]
aver_has = sum(data2)/data2.shape[0]
print(f"无分配平均时间为{aver_no*10}min而有分配平均时间为{aver_has*10}min，他们相差了{aver_no*10-aver_has*10}min")