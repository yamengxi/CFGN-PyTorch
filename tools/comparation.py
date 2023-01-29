import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize = (16.0 / 2, 9.0 / 2))

# plt.xlim(200, 800)
# plt.ylim(30.3, 32.5)

# x_ticks = np.linspace(200, 800, 7)
# plt.xticks(x_ticks, fontweight='bold')
# plt.xlabel('Number of Parameters (K)', fontsize=13, fontweight='bold')

# y_ticks = np.linspace(30.25, 32.65, 13)
# plt.yticks(y_ticks, fontweight='bold')
# plt.ylabel('PSNR (dB)', fontsize=13, fontweight='bold')


# plt.grid()

# tag = [' HNCT', 'RFDN-L', ' RFDN ', ' IMDN ', 'CARN-M', '   IDN', 'MemNet', ' DRRN ', 'LapSRN', ' VDSR ']
# x = [356, 626, 534, 694, 412, 553, 678, 298, 251, 666]
# y = [32.22, 32.24, 32.12, 32.17, 31.92, 31.27, 31.31, 31.23, 30.41, 30.76]
# plt.scatter(x, y, c='b')

# #r'$xxxx$'
# #xy=蓝色点位置
# #xytext：描述框相对xy位置
# #textcoords='offset points'，以xy为原点偏移xytext
# #arrowprops = 画弧线箭头，'---->', rad=.2-->0.2弧度
# for i in range(len(tag)):
#     plt.annotate(tag[i], xy=(x[i]-25,y[i]-0.12), fontsize=10, c='b', fontweight='bold')

# x = [601]
# y = [32.47]
# plt.scatter(x, y, c='r')
# plt.annotate('CFGN', xy=(x[0]-20,y[0]+0.05), fontsize=10, c='r', fontweight='bold')


# plt.savefig('tmp.png')


plt.figure(figsize = (16.0 / 2, 9.0 / 2))

plt.xlim(0, 240)
plt.ylim(29.8, 32.6)

x_ticks = np.linspace(0, 240, 7)
plt.xticks(x_ticks, fontweight='bold')
plt.xlabel('Number of Mult-Adds (G)', fontsize=13, fontweight='bold')

y_ticks = np.linspace(29.8, 32.8, 16)
plt.yticks(y_ticks, fontweight='bold')
plt.ylabel('PSNR (dB)', fontsize=13, fontweight='bold')


plt.grid()

tag = ['RFDN-L', '   IMDN ', ' CARN ', '   IDN', 'LapSRN', 'FSRCNN']
x = [142.4, 158.8, 222.8, 124.6, 29.9, 6.0]
y = [32.24, 32.17, 31.92, 31.27, 30.41, 29.88]
plt.scatter(x, y, c='b')

#r'$xxxx$'
#xy=蓝色点位置
#xytext：描述框相对xy位置
#textcoords='offset points'，以xy为原点偏移xytext
#arrowprops = 画弧线箭头，'---->', rad=.2-->0.2弧度
for i in range(len(tag)):
    if tag[i] == 'FSRCNN':
        plt.annotate(tag[i], xy=(x[i]-5,y[i]+0.06), fontsize=10, c='b', fontweight='bold')
    else:
        plt.annotate(tag[i], xy=(x[i]-10,y[i]-0.16), fontsize=10, c='b', fontweight='bold')

x = [130.9]
y = [32.47]
plt.scatter(x, y, c='r')
plt.annotate('CFGN', xy=(x[0]-8,y[0]+0.06), fontsize=10, c='r', fontweight='bold')


plt.savefig('tmp.png')
