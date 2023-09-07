setwd('/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/')
library(Rcpp)
sourceCpp("density_gaussian.cpp")
type.errors<-function(Cscores,ylabel,K,alpha=0.05,comparison=T){
  #Cscores=EnsemblePrediction
  #include class k if column k is 1
  C_hat  =Cscores
  if(comparison==T){
    C_hat  = ifelse(Cscores>=alpha, 1, 0)
  }
  
  
  #ylabel=ytest
  #### Error
  #K=5
  typeI = rep(0, K+2+K+1)
  names(typeI) = c(paste0("class", (0:K)),'average',paste0("oclass", (0:K)))
  for(k in 0:K){
    #should be 1 in position 1 but 0, so no true
    typeI[k+1] = mean(C_hat[ylabel==k,k+1]==0)
  }
  
  counts=0
  for (k in 0:K) {
    counts=counts+sum(C_hat[ylabel==k,k+1]==0)
  }
  typeI[K+2]=counts/sum(ylabel%in%c(0:K))
  for(k in (K+3):(2*K+3)){
    #should be 1 in position 1 but 0, so no true
    typeI[k] = mean(C_hat[ylabel==k-(K+3),k-(K+3)+1]==1&apply(C_hat[ylabel==k-(K+3),-(k-(K+3)+1)],1,max)==0)
  }
  
  
  typeII = rep(0, 4)
  names(typeII) = c("inlier", "outlier", "average",'average length')
  counts=0
  
  for (k in 0:K) {
    #cat(counts,dim(C_hat[ylabel==k,-(k+1)])[1],'\n')
    counts=counts+sum(apply(C_hat[ylabel==k,-(k+1)],1,max)!=0)
  }
  
  
  typeII[1] = counts/dim(C_hat[ylabel%in%c(0:K),])[1]
  typeII[2] = sum(apply(C_hat[!(ylabel %in% as.factor(c(0:K))),],1,max)!=0)/dim(C_hat[!(ylabel %in% as.factor(c(0:K))),])[1] #should be outliers but still predict
  typeII[3] = (counts+sum(apply(C_hat[!(ylabel %in% as.factor(c(0:K))),],1,max)!=0))/length(ylabel)
  typeII[4] = sum(apply(C_hat,1,sum))/dim(C_hat)[1]
  print(typeI)
  print(typeII)
  return(list(typeI = typeI, typeII = typeII))
}
GenData = function(n_train,n_test,n_test_outlier,train.label,test.label,trainData.full,testData.full){
  #n_train=200
  #n_test=10
  # selecting 500 samples with label 5 and 6 for trainning data
  label0.index = sample(which(train.label == 0), size = n_train, replace = F)
  label1.index = sample(which(train.label == 4), size = n_train, replace = F)
  label2.index = sample(which(train.label == 7), size = n_train, replace = F)
  label3.index = sample(which(train.label == 9), size = n_train, replace = F)
  label4.index = sample(which(train.label == 2), size = n_train, replace = F)
  label5.index = sample(which(train.label == 1), size = n_train, replace = F)
  # label56.index = sort(c(label0.index,label1.index,label2.index))
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index,label4.index,label5.index))
  trainData = trainData.full[label56.index,]
  
  #   # selecting testing data
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
  label1.index.test = sample(which(train.label.select == 4), size = n_test, replace = F)
  label2.index.test = sample(which(train.label.select == 7), size = n_test, replace = F)
  label3.index.test = sample(which(train.label.select == 9), size = n_test, replace = F)
  label4.index.test = sample(which(train.label.select == 2), size = n_test, replace = F)
  label5.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
  label.outlier.test = sample(which(train.label.select == 3 | train.label.select == 5|train.label.select == 6|train.label.select == 8), size = 4*n_test_outlier, replace = F)
  # label.outlier.test = sample(which(train.label.select == 6 | train.label.select == 7|train.label.select == 8|train.label.select == 9), size = 4*n_test_outlier, replace = F)
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test,label4.index.test,label5.index.test,label.outlier.test))
  # label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label.outlier.test))
  test.select = trainData.full[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test))
}
GenData.exp2 = function(n_train=500,n_test=100,n_test_outlier=0,train.label,test.label,trainData.full,testData.full){
  #n_train=200
  #n_test=10
  # selecting 500 samples with label 5 and 6 for trainning data
  label0.index = sample(which(train.label == 0), size = n_train, replace = F)
  label1.index = sample(which(train.label == 1), size = n_train, replace = F)
  label2.index = sample(which(train.label == 2), size = n_train, replace = F)
  label3.index = sample(which(train.label == 4), size = 100, replace = F)
  label4.index = sample(which(train.label == 7), size = 100, replace = F)
  label5.index = sample(which(train.label == 9), size = 100, replace = F)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index,label4.index,label5.index))
  trainData = trainData.full[label56.index,]
  
  #   # selecting testing data
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
  label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = F)
  label3.index.test = sample(which(train.label.select == 4), size = 500, replace = F)
  label4.index.test = sample(which(train.label.select == 7), size = 500, replace = F)
  label5.index.test = sample(which(train.label.select == 9), size = 500, replace = F)
  label.outlier.test = sample(which(train.label.select == 3 | train.label.select == 5|train.label.select == 6|train.label.select == 8), size = 4*n_test_outlier, replace = F)
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test,label4.index.test,label5.index.test,label.outlier.test))
  test.select = trainData.full[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test))
}

NetGetDat = function(networkD,n_train,n_test,n_test_outlier){
  trainFull = networkD
  levels=c("normal.","neptune.", "smurf.","buffer_overflow.","loadmodule.","perl.","pod.","teardrop.",
           "portsweep.","ipsweep.","ftp_write.","imap.","phf.","nmap.","multihop.","warezclient.","warezmaster.","spy.","rootkit.")
  train.label = as.numeric(factor(trainFull$result, levels =levels))-1
  #train.label = trainFull$result#
  table(trainFull$result)
  labelnor.index = sample(which(train.label == 0), size = n_train, replace = F)
  labelnep.index = sample(which(train.label == 1), size = n_train, replace = F)
  labelsmurf.index = sample(which(train.label == 2), size = n_train, replace = F)
  label56.index = sort(c(labelnor.index,labelnep.index,labelsmurf.index))
  trainData = trainFull[label56.index, -ncol(trainFull)]
  
  # test Data selection
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = T)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = T)
  label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = T)
  label3.index.test = sample(which(train.label.select == 3), size = n_test_outlier, replace = T)
  label4.index.test = sample(which(train.label.select == 4), size = n_test_outlier, replace = T)
  label5.index.test = sample(which(train.label.select == 5), size = n_test_outlier, replace = T)
  label6.index.test = sample(which(train.label.select == 6), size = n_test_outlier, replace = T)
  label7.index.test = sample(which(train.label.select == 7), size = n_test_outlier, replace = T)
  label8.index.test = sample(which(train.label.select == 8), size = n_test_outlier, replace = T)
  label9.index.test = sample(which(train.label.select == 9), size = n_test_outlier, replace = T)
  label10.index.test = sample(which(train.label.select == 10), size = n_test_outlier, replace = T)
  label11.index.test = sample(which(train.label.select == 11), size = n_test_outlier, replace = T)
  label12.index.test = sample(which(train.label.select == 12), size = n_test_outlier, replace = T)
  label13.index.test = sample(which(train.label.select == 13), size = n_test_outlier, replace = T)
  label14.index.test = sample(which(train.label.select == 14), size = n_test_outlier, replace = T)
  label15.index.test = sample(which(train.label.select == 15), size = n_test_outlier, replace = T)
  label16.index.test = sample(which(train.label.select == 16), size = n_test_outlier, replace = T)
  label17.index.test = sample(which(train.label.select == 17), size = n_test_outlier, replace = T)
  label18.index.test = sample(which(train.label.select == 18), size = n_test_outlier, replace = T)
  
  
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test, label4.index.test,
                               label5.index.test,label4.index.test,label5.index.test,label6.index.test,label7.index.test,
                               label8.index.test,label9.index.test,label10.index.test,label11.index.test,label12.index.test,
                               label13.index.test,label14.index.test,label15.index.test,label16.index.test,
                               label17.index.test,label18.index.test))
  test.select = trainFull[-label56.index,]
  testData = test.select[label56outlier.test,-ncol(test.select)]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
}



NetGetDat.exp2 = function(networkD,n_train=500,n_test=100,n_test_outlier=0){
  trainFull = networkD
  # table(train.label)
  levels=c("normal.","neptune.", "smurf.","buffer_overflow.","loadmodule.","perl.","pod.","teardrop.",
           "portsweep.","ipsweep.","ftp_write.","imap.","phf.","nmap.","multihop.","warezclient.","warezmaster.","spy.","rootkit.")
  train.label = as.numeric(factor(trainFull$result, levels =levels))-1
  labelnor.index = sample(which(train.label == 0), size = n_train, replace = F)
  labelnep.index = sample(which(train.label == 1), size = n_train, replace = F)
  labelsmurf.index = sample(which(train.label == 2), size = n_test, replace = F)
  label56.index = sort(c(labelnor.index,labelnep.index,labelsmurf.index))
  trainData = trainFull[label56.index, -ncol(trainFull)]
  
  # test Data selection
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = T)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = T)
  label2.index.test = sample(which(train.label.select == 2), size = n_train, replace = T)
  label3.index.test = sample(which(train.label.select == 3), size = n_test_outlier, replace = T)
  label4.index.test = sample(which(train.label.select == 4), size = n_test_outlier, replace = T)
  label5.index.test = sample(which(train.label.select == 5), size = n_test_outlier, replace = T)
  label6.index.test = sample(which(train.label.select == 6), size = n_test_outlier, replace = T)
  label7.index.test = sample(which(train.label.select == 7), size = n_test_outlier, replace = T)
  label8.index.test = sample(which(train.label.select == 8), size = n_test_outlier, replace = T)
  label9.index.test = sample(which(train.label.select == 9), size = n_test_outlier, replace = T)
  label10.index.test = sample(which(train.label.select == 10), size = n_test_outlier, replace = T)
  label11.index.test = sample(which(train.label.select == 11), size = n_test_outlier, replace = T)
  label12.index.test = sample(which(train.label.select == 12), size = n_test_outlier, replace = T)
  label13.index.test = sample(which(train.label.select == 13), size = n_test_outlier, replace = T)
  label14.index.test = sample(which(train.label.select == 14), size = n_test_outlier, replace = T)
  label15.index.test = sample(which(train.label.select == 15), size = n_test_outlier, replace = T)
  label16.index.test = sample(which(train.label.select == 16), size = n_test_outlier, replace = T)
  label17.index.test = sample(which(train.label.select == 17), size = n_test_outlier, replace = T)
  label18.index.test = sample(which(train.label.select == 18), size = n_test_outlier, replace = T)
  # label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,
  #                              label7.index.test,label8.index.test,label9.index.test,label15.index.test))
  
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test, label4.index.test,
                               label5.index.test,label4.index.test,label5.index.test,label6.index.test,label7.index.test,
                               label8.index.test,label9.index.test,label10.index.test,label11.index.test,label12.index.test,
                               label13.index.test,label14.index.test,label15.index.test,label16.index.test,
                               label17.index.test,label18.index.test))
  test.select = trainFull[-label56.index,]
  testData = test.select[label56outlier.test,-ncol(test.select)]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
}


# NetGetDat = function(networkD,n_train,n_test,n_test_outlier){
#   trainFull = networkD
#   train.label = trainFull$class
#   label0.index = sample(which(train.label == 0), size = n_train, replace = F)
#   label1.index = sample(which(train.label == 1), size = n_train, replace = F)
#   label2.index = sample(which(train.label == 2), size = n_train, replace = F)
#   label3.index = sample(which(train.label == 3), size = n_train, replace = F)
#   label56.index = sort(c(label0.index,label1.index,label2.index,label3.index))
#   trainData = trainFull[label56.index,]
#   
#   # test Data selection
#   train.label.select = train.label[-label56.index]
#   label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
#   label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
#   label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = F)
#   label3.index.test = sample(which(train.label.select == 3), size = n_test, replace = F)
#   label4.index.test = sample(which(train.label.select == 4), size = n_test_outlier, replace = F)
#   label5.index.test = sample(which(train.label.select == 5), size = n_test_outlier, replace = F)
#   label6.index.test = sample(which(train.label.select == 6), size = n_test_outlier, replace = F)
#   label7.index.test = sample(which(train.label.select == 7), size = n_test_outlier, replace = F)
#   
#   label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test, label4.index.test,
#                                label5.index.test,label4.index.test,label5.index.test,label6.index.test,label7.index.test
#   ))
#   test.select = trainFull[-label56.index,]
#   testData = test.select[label56outlier.test,]
#   
#   return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
#               trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
# }
# 
# NetGetDat.exp2 = function(networkD,n_train=500,n_test=100,n_test_outlier=0){
#   trainFull = networkD
#   train.label = trainFull$class
#   label0.index = sample(which(train.label == 0), size = n_train, replace = F)
#   label1.index = sample(which(train.label == 1), size = n_train, replace = F)
#   label2.index = sample(which(train.label == 2), size = n_test, replace = F)
#   label3.index = sample(which(train.label == 3), size = n_test, replace = F)
#   # label4.index = sample(which(train.label == 4), size = n_test, replace = T)
#   # label5.index = sample(which(train.label == 5), size = n_test, replace = T)
#   # label6.index = sample(which(train.label == 6), size = n_test, replace = T)
#   # label7.index = sample(which(train.label == 7), size = n_test, replace = T)
#   label56.index = sort(c(label0.index,label1.index,label2.index,label3.index))
#   trainData = trainFull[label56.index,]
#   
#   # test Data selection
#   train.label.select = train.label[-label56.index]
#   label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
#   label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
#   label2.index.test = sample(which(train.label.select == 2), size = n_train, replace = F)
#   label3.index.test = sample(which(train.label.select == 3), size = n_train, replace = F)
#   # label4.index.test = sample(which(train.label.select == 4), size = n_train, replace = T)
#   # label5.index.test = sample(which(train.label.select == 5), size = n_train, replace = T)
#   # label6.index.test = sample(which(train.label.select == 6), size = n_train, replace = T)
#   # label7.index.test = sample(which(train.label.select == 7), size = n_train, replace = T)
#   
#   label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test))
#   test.select = trainFull[-label56.index,]
#   testData = test.select[label56outlier.test,]
#   
#   return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
#               trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
# }


Density.loop<-function(n,n_train,n_test,n_test_outlier,mydata='mnist',networkD=NA,train.label=NA,test.label=NA,trainData.full=NA,testData.full=NA,K=5,alpha=0.05,seed=T,type='exp1'){
  #K=5
  averagetypeI = list()
  averagetypeII = list()
  for (round in 1:n) {
    if(seed==T){
      set.seed(round)
    }
    if(mydata=='mnist'){
      if(type=='exp1'){
        GenDat<-GenData(n_train,n_test,n_test_outlier,train.label,test.label,trainData.full,testData.full)
      }else{
        if(type=='exp2'){
          GenDat<-GenData.exp2(n_train,n_test,n_test_outlier,train.label,test.label,trainData.full,testData.full)
        }
      }
    }else{
      if(type=='exp1'){
        GenDat<-NetGetDat(networkD,n_train,n_test,n_test_outlier)
        train.label=GenDat$train.label
        GenDat$trainData=GenDat$trainData
        GenDat$testData=GenDat$testData
      }else{
        if(type=='exp2'){
          GenDat<-NetGetDat.exp2(networkD,n_train,n_test,n_test_outlier)
          train.label=GenDat$train.label
          GenDat$trainData=GenDat$trainData
          GenDat$testData=GenDat$testData
        }
      }
    }
    #GenDat<-GenData(n_train,n_test,n_test_outlier,train.label,test.label,trainData.full,testData.full)
    xtrain = as.matrix(GenDat$trainData)
    ytrain = as.factor(train.label[GenDat$trainLabel])
    xtest = as.matrix(GenDat$testData)
    ytest = as.factor(GenDat$train.label.select[GenDat$testLabel])
    foldid = sample(1:2, length(ytrain), replace = TRUE);
    foldid_te = sample(1:2,length(ytest), replace = TRUE)
    xtrain1 = xtrain[foldid==1,]; xtrain2 = xtrain[foldid==2,];
    ytrain1 = ytrain[foldid==1]; ytrain2=ytrain[foldid==2]
    xtest1 = xtest[foldid_te ==1,]; xtest2 = xtest[foldid_te==2,];
    labels = sort(unique(ytrain))
    
    cat('Dim of train data1',dim(xtrain1),length(ytrain1))
    cat('\nDim of train data2',dim(xtrain2),length(ytrain2))
    cat('\nDim of test data',dim(xtest),length(ytest))
    
    ccs1 = list(); ccs2 = list()
    bws = matrix(0, ncol = K+1, nrow= ncol(xtest))
    for(k in 1:(K+1)){
      ccs1[[k]] =which(ytrain1 == k-1)
      ccs2[[k]] =which(ytrain2 == k-1)
      for(j in 1:ncol(xtrain1)){
        #each feature each label (1,2)
        bws[j,k] =  density(xtrain1[ccs1[[k]],j])$bw
      }
    }
    ##### estiamte log density for the test samples with respect to each training class in x1
    dte = matrix(0, ncol = K+1, nrow = nrow(xtest)) #two labels
    
    for(k in 1:(K+1)){
      #input:
      #   samples for evaluation,
      #   training samples,
      #   bandwidth,
      #   row number of evaluating points
      #   row number of training points
      #   dimension
      #return: a vector of log density for samples under evaluation
      
      dte[,k] = density_gaussian(xtest, xtrain1[ccs1[[k]],], bws[,k], nrow(xtest), length(ccs1[[k]]), ncol(xtest))
    }
    
    ##### estiamte log density for the calibration samples (x2) with respect to each training class in x1
    dtr = rep(0, nrow(xtrain2))
    
    for(k in 1:(K+1)){
      dtr[ccs2[[k]]] = density_gaussian(xtrain2[ccs2[[k]],],xtrain1[ccs1[[k]],],  bws[,k], length(ccs2[[k]]),length(ccs1[[k]]), ncol(xtest))
    }
    
    
    
    m = nrow(xtest)
    Cscores <- matrix(0, ncol = K+1, nrow = m)
    for(i in 1:m){
      for(k in 1:(K+1)){
        Cscores[i,k] = (sum(dte[i,k] >= dtr[ccs2[[k]]])+1)/(length(ccs2[[k]])+1)
      }
    }
    
    errors=type.errors(Cscores = Cscores,ylabel = ytest,K=K,alpha=alpha)
    averagetypeI[[round]]=errors$typeI
    averagetypeII[[round]]=errors$typeII
  }
  #return(list(typeII09=as.numeric(averagetypeII[3])/n, typeII69 = as.numeric(averagetypeII[2])/n))
  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII))
  
  #if(EXPS==1){
  #  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII))
  #}else{
  #  if(EXPS!=4){
  #    return(list(typeII05=as.numeric(averagetypeII[1])/n, typeII69 = as.numeric(averagetypeII[2])/n))
  #  }else{
  #    return(list(typeI05=as.numeric(averagetypeI[7])/n, typeII05 = as.numeric(averagetypeII[1])/n, typeII69 = as.numeric(averagetypeII[2])/n))
  #  }
  #}
}
