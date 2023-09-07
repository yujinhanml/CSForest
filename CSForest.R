#' BCOPSEnsemble Training function
#' @param index.l - a positive integer, referring to which fold you want to train
#' @param  x - training data frame
#' @param  y - full list of labels from training data
#' @param  label- list of unique values in y
#' @param  xte - testing data frame (no labels)
#' @param  num.b - a positive integer, number of repetions in training process
#' @param  weightFun - a boolean value, referring whether applying a weighting function while the trainning process
#' @return model.class.list, a list of lists with length(model.class.list) = length(label). For example,
#'         model.class.list[[1]] is built for the first lable in the label list, which is label[1], and
#'         has two list components, index and model. The index is a list storing the lists of indexes of samples used
#'        in the ranger model with corresponding index. The model is a list of ranger objects.
#' @examples
#' \dontrun{
#' testData = read.csv("~/Desktop/自学/申请/大三暑假/Research /code/BCOPS/SimTest.csv")[,-1]
#' trainData = read.csv("~/Desktop/自学/申请/大三暑假/Research /code/BCOPS/SimTrain.csv")[,-1]
#' ########cleaning data##############
#' set.seed(123)
#' colnames(testData) = c("x1", "x2", "label")
#' colnames(trainData) = c("x1", "x2", "label")
#' # getting rid of labels
#' x.testing = testData[,1:2]
#' x.train = trainData[,1:2]
#' colnames(x.train)= colnames(x.testing)
#' y.train = trainData[,3]
#' # labels
#' label.list = sort(unique(y.train))
#' testing.l.index = L.Fold(x.testing, 4)
#'
#' index1 = testing.l.index[[1]]
#' sample2 = BCOPSEnsemble.trainning(index1, x.train, y.train, label.list, x.testing, 50)
#'  }
#' @export

setwd('/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/')
#install.packages("https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.9.900.3.0.tar.gz", repos=NULL, type="source")
library(Rcpp)
library(RcppArmadillo)
library(ranger)
sourceCpp("CSForest/src/density_gaussian.cpp")
sourceCpp("CSForest/src/JacknifeABcompare.cpp")

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

CSForest_trees = function(x, y, labels,xte,  B = 2000, 
                          weight =1.0, ntree = 1){
  n_tr = nrow(x)
  n_te = nrow(xte)
  K = length(labels)
  models= list()
  outbag_idx_train = list()
  outbag_idx_test = list()
  
  for(k in 1:length(labels)){
    #run a random forest with ranger and extract individual trees and indices
    dat1 = data.frame(x[y==labels[k],])
    dat1$response = labels[k]
    dat2 = data.frame(x[y!=labels[k],])
    dat2$response = y[y!=labels[k]]
    dat3 = data.frame(xte)
    dat3$response = "test"
    dat = rbind(rbind(dat1, dat2),dat3)
    levels = c(labels[k], "test", labels[-k])
    dat$response = factor(dat$response, levels)
    tmp =  nrow(xte)/sum(y!=labels[k])
    tmp1 = table(dat$response)/nrow(dat)
    # cat('table(dat$response)',table(dat$response))
    # cat('\ntmp1[-c(1:2)]',tmp1[-c(1:2)])
    tmp1[-c(1:2)] = tmp1[-c(1:2)]*weight
    # cat('\nweight',weight)
    # cat('\ntmp1[-c(1:2)]*weight',tmp1[-c(1:2)])
    tmp1[-c(1:2)] = tmp1[-c(1:2)]*tmp #tmp1 还是很大
    # cat('\ntmp1[-c(1:2)]',tmp1[-c(1:2)],'tmp',tmp)
    tmp10 = table(dat$response)/nrow(dat)
    # cat('\ntmp10',tmp10)
    tmp1[tmp1 >tmp10] = tmp10[tmp1>tmp10]
    sample.fraction = tmp1 * 1.0
    # cat('\ntmp1',tmp1)
    system.time(models[[k]] <- ranger(response~., data = dat, sample.fraction  = sample.fraction, probability = T, 
                                      num.trees = B, keep.inbag=T, replace = T))
    mat1 = sapply(models[[k]]$inbag.counts, function(z) z[dat$response==labels[k]]==0)
    mat2 = sapply(models[[k]]$inbag.counts, function(z) z[dat$response=="test"]==0 )
    mat3 = sapply(models[[k]]$inbag.counts, function(z) z[dat$response%in%labels[-k]]==0 )
    outbag_idx_train[[k]] = mat1
    outbag_idx_test[[k]] = mat2
  }
  return(list(models=models, outbag_idx_train = outbag_idx_train, 
              outbag_idx_test= outbag_idx_test))
}


CSForest_calibrated_prob = function(x, y, labels,  xte, models, B,
                                    outbag_idx_train,
                                    outbag_idx_test,  m0 = NULL){
  ##prediction on the training
  if(is.null(m0)){
    m0 = nrow(xte)
  }
  n = nrow(x); m = nrow(xte); K = length(labels)
  tree_pred_train = list()
  tree_pred_test = list()
  dat = data.frame(x)
  for(k in 1:K){
    tree_pred_train[[k]] = predict(models[[k]],dat[y==labels[k],],predict.all = T)$predictions[,1,]
  }
  ##prediction on the test
  
  dat = data.frame(xte)
  for(k in 1:K){
    tree_pred_test[[k]] = predict(models[[k]],dat,predict.all = T)$predictions[,1,]
  }
  outBtrain_c =list()
  for(k in 1:K){
    outBtrain_c[[k]] = apply(outbag_idx_train[[k]],1,which)
    if(class(outBtrain_c[[k]])=="list"){
      outBtrain_c[[k]] = lapply(outBtrain_c[[k]], function(z) z-1)
    }else{
      outBtrain_c[[k]] = apply(outBtrain_c[[k]], 2,function(z) z-1)
    }
    
  }
  outBtest_c = list()
  for(k in 1:K){
    outBtest_c[[k]] = apply(outbag_idx_test[[k]],1,which)
    if(class(outBtest_c[[k]])=="list"){
      outBtest_c[[k]] = lapply(outBtest_c[[k]], function(z) z-1)
    }else{
      outBtest_c[[k]] = lapply(outBtest_c[[k]], function(z) z-1)
    }
  }
  tmp = csforest_calibration(n = length(y), m =m, m0 = m0, 
                             K = K,  B=B,
                             tree_pred_train = tree_pred_train,
                             tree_pred_test = tree_pred_test,
                             outBtrain = outBtrain_c, outBtest = outBtest_c
  )
  tmp = tmp+1
  for(k in 1:K){
    nk = sum(y==labels[k])
    tmp[,k] = tmp[,k]/(nk+1)
  }
  return(tmp)
}



CSForest.loop<-function(n,n_train,n_test,n_test_outlier,mydata='mnist',networkD=NA,train.label=NA,test.label=NA,trainData.full=NA,testData.full=NA,weight=1,B=3000,alpha=0.05,K=5,seed=T,type='exp1'){
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
    
      
    trainData = GenDat$trainData#as.matrix(GenDat$trainData)
    testData = GenDat$testData#as.matrix(GenDat$testData)
    colnames(trainData) = colnames(testData)
    y.test = GenDat$train.label.select[GenDat$testLabel]
    x.test = testData
    x.train = trainData
    
    y.train = train.label[GenDat$trainLabel]
    labels = sort(unique(y.train))
    cat('Dim of train data',dim(x.train))
    cat('\nDim of test data',dim(x.test))
    
    if(weight=='LOG'){
      weight=1.0/(1.0+log(nrow(x.test)))
    }
     
    
   
    model_trees <- CSForest_trees(x=x.train, y=y.train, labels=labels,  xte = x.test,B = B, ntree = 1, weight = weight)# 1.0/(1.0+log(nrow(x.test)))
    
    prediction.conformal <- CSForest_calibrated_prob(x=x.train, y=y.train, labels=labels,  xte =x.test,
                                                     models = model_trees$models,B = B,
                                                     outbag_idx_train = model_trees$outbag_idx_train,
                                                     outbag_idx_test = model_trees$outbag_idx_test)
    
    
    
    errors=type.errors(Cscores = prediction.conformal,ylabel = y.test,K=K,alpha=alpha)
    averagetypeI[[round]]=errors$typeI
    averagetypeII[[round]]=errors$typeII
  }
  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII))
}




