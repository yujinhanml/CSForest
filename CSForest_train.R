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
#' testData = read.csv("~/SimTest.csv")[,-1]
#' trainData = read.csv("~/SimTrain.csv")[,-1]
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
    tmp1[-c(1:2)] = tmp1[-c(1:2)]*weight
    tmp1[-c(1:2)] = tmp1[-c(1:2)]*tmp
    tmp10 = table(dat$response)/nrow(dat)
    tmp1[tmp1 >tmp10] = tmp10[tmp1>tmp10]
    sample.fraction = tmp1 * 1.0
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

