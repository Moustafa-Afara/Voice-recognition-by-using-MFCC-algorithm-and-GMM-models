from python_speech_features import mfcc
import os
from os import listdir
from os.path import isfile, join, isdir

import numpy as np

from sklearn.mixture import GaussianMixture				# استيراد مكتبة خوارزمية GMM
from sklearn.linear_model import LogisticRegression
import scipy.io.wavfile as wav

from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# تحديد ملفات التدريب والاختبار
train_path="pathe of training files"
test_path="path of testing files"
vec_length=13   #تحديد عدد معاملات MFC


#تابع إنشاء ميزات الخوارزمية
def createMFCC(path):
	(rate,sig)=wav.read(path)
	mfcc_feat=mfcc(sig,rate, winlen=0.025, winstep=0.010)

	return mfcc_feat.reshape(-1, 1)

#إنشاء نموذج GMM
gmm= []


#-----------------------------تدريب النموذج--------------------------------
person_files=[f for f in listdir(train_path) if isdir(join(train_path,f))]
print (person_files)

for i in range(0,len(person_files)):
	audioPath=join(train_path,person_files[i])
	audioFiles=[l for l in listdir(audioPath) if isfile(join(audioPath,l))]

	mfcc_temp=np.empty((0,vec_length)).reshape(-1, 1)
	for j in range(0,len(audioFiles)):
		filePath=join(audioPath,audioFiles[j])
		temp_feat=createMFCC(filePath)
		mfcc_temp=np.append(mfcc_temp,temp_feat,axis=None)  #تحضير جميع ميزات الخوارزمية لتدريبها على نموذج GMM واحد

	gmm.append(GaussianMixture(n_components=10,covariance_type='full',max_iter=20, random_state=None))  #تدريب النموذج
	gmm[i].fit(mfcc_temp.reshape(-1, 1))




#---------------------------اختبار النموذج---------------------------
test_files=[f for f in listdir(test_path) if isdir(join(test_path,f))]

print ("The number of labels identified correctly are:")
for i in range(0,len(test_files)):
	audioPath=join(test_path,test_files[i])
	audioFiles=[l for l in listdir(audioPath) if isfile(join(audioPath,l))]

	res=[]
	print ("current label",i)
	mfcc_temp=np.empty((0,vec_length))
	for j in range(0,len(audioFiles)):
		filePath=join(audioPath,audioFiles[j])
		temp_feat=createMFCC(filePath)  #إنشاء اختبار لميزات الخوارزمية
		
		num_features=temp_feat.shape[0]
		
		labels=np.empty((num_features,1))
		for k in range(0,num_features):
			y=[None]*22
			for l in range(0,22):
				y[l]=np.exp(gmm[l].score_samples(temp_feat[k].reshape(-1, 1)))   #تسجيل درجة التحقق من كل ميزة

			max_class=y.index(max(y))  #وضع أعلى درجة تحقق كخرج للاختبار
			labels[k]=max_class          

		res.append(stats.mode(labels)[0])	#يتم إعطاء فئة الوضع لجميع ميزات mfcc لملف واحد على أنها الفئة المتوقعة لهذا الملف

#رقم الخرج لكل توقع صحيح بين الأشخاص
	correct=0
	for x in range(0,2):
		if(res[x]==i):
			correct=correct+1
	print (correct)

