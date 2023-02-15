# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:31:28 2023

@author: joey
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pretty_midi
import numpy as np
import random

trainsetmidi=[]

def read_midi_notes(f):
  midi_inputs = [] # 存放所有的音符
  pm = pretty_midi.PrettyMIDI(f) # 加载一个文件
  instruments = pm.instruments # 获取乐器
  instrument = instruments[0] # 取第一个乐器，此处是原声大钢琴
  notes = instrument.notes # 获取乐器的演奏数据
  # 以开始时间start做个排序。因为默认是依照end排序
  sorted_notes = sorted(notes, key=lambda note: note.start)
  #prev_start = sorted_notes[0].start
  # 循环各项指标，取出前后关联项
  for note in sorted_notes: 
      #stepfrom =  note.start - prev_start # 此音符与上一个距离
      #duration = note.end - note.start # 此音符的演奏时长
      #prev_start = note.start # 此音符开始时间作为最新
      # 指标项：[音高（音符），同前者的间隔，自身演奏的间隔]
      midi_inputs.append(note.pitch)
  return midi_inputs

filenames = tf.io.gfile.glob("C:/Users/joey/Desktop/ai_music-main/datasets/*.mid")
for file in filenames:
    midi = read_midi_notes(file)
    trainsetmidi=trainsetmidi+midi
    
#print('allnote:\n')
#print(trainsetmidi)
alterednote=[]
for anote in trainsetmidi:
    alterednote.append(anote-48)
#print(alterednote)

intervaling=[]
for i in range(0,1100):
    intervaling.append(alterednote[i+1]-alterednote[i]+24)
print(intervaling)

noteset=[]
for i in range(0,1001):
    noteset.append(alterednote[i:i+16])
#print(noteset)

outset=[]
for i in range(0,1001):
    outset.append(alterednote[i+17])
#print(outset)

def zero():
    outputing=[]
    for i in range(0,48):
        outputing.append(0)
    return outputing


#占位符placeholder，后面通过feed_dict喂入数据
x = tf.placeholder("float", [None, 16])
y_ = tf.placeholder("float", [None,48]) #y_为真实值标签
#variable张量，模型训练过程不断优化变动的张量
W1 = tf.Variable(tf.zeros([16,8])) #权重
b1 = tf.Variable(tf.zeros([8]))  #偏置
W2 = tf.Variable(tf.zeros([8,48])) #权重
b2 = tf.Variable(tf.zeros([48]))  #偏置
#使用softmax回归模型
h = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
y = tf.nn.softmax(tf.matmul(h,W2) + b2)
#使用交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#随机梯度下降优化器
train_step = tf.train.GradientDescentOptimizer(0.0036).minimize(cross_entropy)
#变量初始化
init = tf.initialize_all_variables()
#求准确率
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    #使用1000条巴赫partita旋律进行训练
    sess.run(init)
    ac=0
    for i in range(0,99):
        inp=np.matrix(noteset[i])
        oup=zero()
        oup[outset[i]]=1;
        oup=np.matrix(oup)
        print("use musicset no.",i+1)
        for j in range(0,99):
            sess.run(train_step, feed_dict={x: inp, y_: oup}) 
            print("trained",(i+1)*(j+1)," times")
            cor=sess.run(accuracy, feed_dict={x: inp, y_: oup})
            ac=ac+cor
            print("accurate?:",cor," accuracy till now:",ac/(i+1))
    print("training done\n")
    
    OUTIN=noteset[10]
    OUTFINAL=[]
    for j in range(0,6000):
        OUTINP=np.matrix(OUTIN)
        nextnotep=sess.run(y,feed_dict={x: OUTINP})
        nextnote=np.argmax(nextnotep)
        nextnotep=nextnotep[0]
        print(nextnotep)
        for k in range (0,48):
            if(nextnotep[k]<0.002):
                nextnotep[k]=0
        for k in range (1,48):
            nextnotep[k]=nextnotep[k-1]+nextnotep[k]
        randomnum=random.random()*nextnotep[47]-0.00000001
        for k in range (0,48):
            if(randomnum<nextnotep[k]):
                nextnote=k
                break
        print(randomnum)
        OUTIN.remove(OUTIN[0])
        OUTIN.append(nextnote)
        OUTFINAL.append(nextnote)
        print(OUTIN)

pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"))
for n in range(0,6000):
  note = pretty_midi.Note(velocity=100,pitch=OUTFINAL[n]+48,start=0.125*n,end=0.125*n+0.124)
  instrument.notes.append(note)
pm.instruments.append(instrument)
pm.write("partita simu.midi")

