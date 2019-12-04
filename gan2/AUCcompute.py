# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 03:29:08 2019

@author: cloh5
"""
def comparehist(epoch):
    checkpoint.restore('./training_checkpoints/ckpt-'+str(epoch))
    real_pred = predictions(real)
    deepfake_pred = predictions(deepfake)
    face2face_pred = predictions(face2face)
    faceswap_pred = predictions(faceswap)
    neuraltextures_pred = predictions(neuraltextures)
    fake_preds=np.concatenate((deepfake_pred, face2face_pred, faceswap_pred, neuraltextures_pred))
    
    hist_real = cv2.calcHist([real_pred], [0], None, [40], [-20,20])
    cv2.normalize(hist_real, hist_real)

    hist_fakes = cv2.calcHist([fake_preds], [0], None, [40], [-20,20])
    cv2.normalize(hist_fakes, hist_fakes)
    
    hist_real=hist_real/np.sum(hist_real)
    hist_fakes=hist_fakes/np.sum(hist_fakes)

    intsec=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_INTERSECT)
    correl=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_CORREL)
    KL1=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_KL_DIV)
    KL2=cv2.compareHist(hist_fakes,hist_real,cv2.HISTCMP_KL_DIV)
    bhat=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_BHATTACHARYYA)

plt.figure()
plt.plot(hist_img1)
plt.plot(hist_img2)

import matplotlib.pyplot as plt


def plot_ROC(fake_preds,real_pred):
    numx=40
    x=np.linspace(-20,20,num=40)
    
    fakehist=np.histogram(fake_preds,bins=40,range=[-20,20],density=True)
    realhist=np.histogram(real_pred,bins=40,range=[-20,20],density=True)
    hist1=fakehist[0]
    hist2=realhist[0]
    #total
    total_hist1=np.sum(hist1)
    total_hist2=np.sum(hist2)
    #cummulative sum
    cum_TP=0
    cum_FP=0
    TPRvec=[]
    FPRvec=[]
    for i in range(len(x)):
        cum_TP += hist2[len(x)-1-i]
        cum_FP += hist1[len(x)-1-i]
        FPR=cum_FP/total_hist1
        TPR=cum_TP/total_hist2
        TPRvec.append(TPR)
        FPRvec.append(FPR)
    auc=np.sum(TPRvec)/numx
    print(auc)
#    plt.figure(99)
#    plt.plot(FPRvec,TPRvec,label=str(epoch)+"; AUC="+str(auc))
    
    return auc

epochs = np.linspace(1, 500, 10, dtype='int')
auc_values = []

for epoch in epochs:
    checkpoint.restore('./training_checkpoints_ob/ckpt-'+str(epoch))
    real_pred = predictions(real)
    deepfake_pred = predictions(deepfake)
    face2face_pred = predictions(face2face)
    faceswap_pred = predictions(faceswap)
    neuraltextures_pred = predictions(neuraltextures)
    fake_preds=np.concatenate((deepfake_pred, face2face_pred, faceswap_pred, neuraltextures_pred))
    
    print(epoch)
    #plt.figure(figsize=(6,4))
    auc = plot_ROC(fake_preds,real_pred)
    auc_values.append(auc)
#plt.xlabel('FPR')
#plt.ylabel('TPR')
#plt.title('ROC Curve')
#plt.legend()

plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 10, dtype='int'), auc_values)
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('AUC during training')
plt.savefig(subj+'AUC_500.png')