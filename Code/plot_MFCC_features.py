import matplotlib.pyplot as plt                     # استيراد مكتبة الرسم
from scipy.io import wavfile                        # استيراد مكتبة معالجة البيانات الصوتية
from python_speech_features import mfcc, logfbank   # استيراد مكتبة الخوارزمية المستخدمة

# تحديد ملف القراءة للرسم وضبط التقطيع
frequency_sampling, audio_signal = wavfile.read("C:/Users/moust/OneDrive/Desktop/أحمد المنصور/Code/Dataset/train_wavdata/fajw0/1.wav")

audio_signal = audio_signal[:15000]

# رسم معاملات MFCC
features_mfcc = mfcc(audio_signal, frequency_sampling)

print ('\nMFCC:\nNumbers of windows = ', features_mfcc.shape[0])
print ('Length of each feature = ', features_mfcc.shape[1])

features_mfcc = features_mfcc.T

plt.matshow (features_mfcc)
plt.title ('MFCC')

# خرج مرشح البنك في المجال الترددي
filterbank_features = logfbank (audio_signal, frequency_sampling)

print ('\nFilter bank:\nNumber of windows = ', filterbank_features.shape[0])
print ('Length if each feature = ', filterbank_features.shape[1])

filterbank_features = filterbank_features.T

plt.matshow (filterbank_features)
plt.title ('Filter bank')
plt.show()
