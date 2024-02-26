import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-2,8, num=301)
# y = np.sinc((x-2.21)*3)


def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "iters={:.3f}, acc={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

if __name__ == '__main__':
   acc = []
   iter = []
   j = 0
   with open('/media/ee4012/Disk3/Eric/Co-DETR/DA_stuff/outputs/DAAD_NEW/20240206_215601.log', 'r') as r:
      file =  r.readlines()
      print(file)
      i = 0
      for l in file:
         if "acc:" in l:
            i = i + 1
            if (i % 3 == 0):
               j = j + 10
               acc.append(float(l[46:-1]))
               print(acc)
               iter.append(int(j))

   # with open('20231127_122448.log', 'r') as r:
   #    file =  r.readlines()
   #    print(file)
   #    i = 0
   #    for l in file:
   #       if "acc" in l:
   #          i = i + 1
   #          if (i % 3 == 0):
   #             j = j + 100
   #             acc.append(float(l[46:-1]))
   #             print(acc)
   #             iter.append(int(j))

   # with open("20231128_163922.log", "r") as r :
   #    file =  r.readlines()
   #    print(file)
   #    i = 0
   #    for l in file:
   #       if "acc" in l:
   #          i = i + 1
   #          if (i % 3 == 0):
   #             j = j + 100
   #             acc.append(float(l[46:-1]))
   #             print(acc)
   #             iter.append(int(j))
   iter = iter[:]
   acc = acc[:]
   fig, ax = plt.subplots()
   iter = np.array(iter)
   acc = np.array(acc)
   ax.plot(iter,acc)
   ax.legend(loc='upper left')
   # print(acc) 
   annot_max(iter,acc)
   plt.plot(iter,acc, label='Dcls+GRL')
   plt.ylabel('accuracy')
   plt.xlabel('iterations')
   plt.legend()
   plt.show()

   from natten import NeighborhoodAttention1D
   from natten import NeighborhoodAttention2D

   na1d = NeighborhoodAttention1D(dim=128, kernel_size=7, dilation=2, num_heads=4).cuda()
   na2d = NeighborhoodAttention2D(dim=128, kernel_size=7, dilation=2, num_heads=4).cuda()

