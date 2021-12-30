#!/bin/sh

for i in ./data/kaggle/clips//*.mp3; # 遍历当前文件夹下所有的有mp3格式后缀的文件
do
ffmpeg -i "$i" -f wav "./${i}.wav"; # 用ffmpeg将mp3格式的后缀加上.wav后缀
done
rename 's/\.mp3.wav/\.wav/' # 批量重命名新产生的.mp3.wav文件为.wav文件