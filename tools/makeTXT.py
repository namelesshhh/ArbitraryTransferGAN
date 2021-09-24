# -*- coding:utf-8 -*-
import os
dir = 'G:\crops'
f = open(os.path.join(dir, 'location.txt'), 'w+')
for curDir, dirs, files in os.walk(os.path.join(dir, '0')):
    print("====================")
    print("现在的目录：" + curDir)
    print("该目录下包含的子目录：" + str(dirs))
    print("该目录下包含的文件数目：" + str(len(files)))
    for name in files:
        file_name = os.path.join(curDir, name)
        #print(file_name)
        f.write(file_name)
        f.write('\n')

f.close()