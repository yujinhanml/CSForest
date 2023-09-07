rm(list = ls())
setwd('/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/')
library(dplyr)
library(randomForest)
library(reshape2)
library(ggplot2)
library(ranger)
library(tidyverse)
library(ggsci)
library(patchwork)

#library(devtools)

#install_github("LeyingGuan/BCOPS/bcops",force = TRUE)

library(bcops)

#--------- readin simulation data ------------
print("readin simulation data")
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

# Note: Change the address here
train.file= load_image_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/train-images.idx3-ubyte")
test.file = load_image_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/t10k-images.idx3-ubyte")

train.label <- load_label_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/train-labels.idx1-ubyte")
test.label <- load_label_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/t10k-labels.idx1-ubyte")
trainData.full =  as.data.frame(train.file[["x"]])
testData.full =  as.data.frame(test.file[["x"]])

GenData = function(n_train,n_test){
  #n_train=200
  #n_test=10
  # selecting 500 samples with label 5 and 6 for trainning data
  label0.index = sample(which(train.label == 0), size =n_train, replace = F)
  label1.index = sample(which(train.label == 1), size =n_train, replace = F)
  label2.index = sample(which(train.label == 2), size =n_train, replace = F)
  label3.index = sample(which(train.label == 3), size =n_train, replace = F)
  label4.index = sample(which(train.label == 4), size =n_train, replace = F)
  label5.index = sample(which(train.label == 5), size =n_train, replace = F)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index,label4.index,label5.index))
  trainData = trainData.full[label56.index,]
  
  #   # selecting testing data
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size =n_test, replace = F)
  label1.index.test = sample(which(train.label.select == 1), size =n_test, replace = F)
  label2.index.test = sample(which(train.label.select == 2), size =n_test, replace = F)
  label3.index.test = sample(which(train.label.select == 3), size =n_test, replace = F)
  label4.index.test = sample(which(train.label.select == 4), size =n_test, replace = F)
  label5.index.test = sample(which(train.label.select == 5), size =n_test, replace = F)
  label.outlier.test = sample(which(train.label.select == 6 | train.label.select == 7|train.label.select == 8|train.label.select == 9), size = 4*n_test, replace = F)
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test,label4.index.test,label5.index.test,label.outlier.test))
  test.select = trainData.full[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test))
}

type.errors<-function(Cscores,ylabel,K=5,alpha=0.05,comparison=F){
  #Cscores=EnsemblePrediction
  #include class k if column k is 1
  # cbind(C_hat.Opt.shift,ytest)
  #C_hat=C_hat.Opt.shift
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
    typeI[k+1] = sum(C_hat[ylabel==k,k+1]==0)/nrow(C_hat[ylabel==k,])
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
  
  
  typeII = rep(0, 3)
  names(typeII) = c("inlier", "outlier", "average")
  counts=0
  
  for (k in 0:K) {
    ##重复计算了
    #cat(counts,dim(C_hat[ylabel==k,-(k+1)])[1],'\n')
    counts=counts+sum(apply(C_hat[ylabel==k,-(k+1)],1,max)!=0)
  }
  # sum(apply(C_hat[ylabel==0,-(0+1)],1,max)!=0)
  # sum((C_hat[ylabel==0,(0+1)]==0))
  # C_hat[ylabel==0,]
  
  typeII[1] = counts/dim(C_hat[ylabel%in%c(0:K),])[1]
  typeII[2] = sum(apply(C_hat[!(ylabel %in% as.factor(c(0:K))),],1,max)!=0)/dim(C_hat[!(ylabel %in% as.factor(c(0:K))),])[1] #should be outliers but still predict
  typeII[3] = (counts+sum(apply(C_hat[!(ylabel %in% as.factor(c(0:K))),],1,max)!=0))/length(ylabel)
  print(typeI)
  print(typeII)
  return(list(typeI = typeI, typeII = typeII))
}

data_mean<-function(data){
  datamean_1=0
  for (i in 1:length(data$averagetypeI)) {
    datamean_1=datamean_1+data$averagetypeI[[i]]
  }
  datamean_2=0
  for (i in 1:length(data$averagetypeII)) {
    datamean_2=datamean_2+data$averagetypeII[[i]]
  }
  return(list(averagetypeI =datamean_1/length(data$averagetypeII), averagetypeII = datamean_2/length(data$averagetypeII)))
  
}
##ACRF
print('Baseline3:OptInsample')

E.score<-function(sy0,K=5,randomness=F){
  n=length(sy0)
  y=sy0[n]
  a=sy0[-n]
  b=a[order(a,decreasing=TRUE)]
  c=cumsum(b)
  order.index=order(a,decreasing=TRUE)
  v=as.numeric(c[which(order.index==(y))])
  u=runif(1)
  if(randomness){
    return(v-b[which(order.index==(y))]*u)
  }else{
    return(v-b[which(order.index==(y))])
  }
  # t=v
  # if(v==as.numeric(b[1])){
  #   t=0
  # }else{
  #   t=v
  # }
  # 
  # u=runif(1)
  # V=(v-t)/(1e-6+b[which(order.index==(y))])
  # if(u<=V){
  #   #return(0)
  #   return(v-u*b[which(order.index==(y))])
  # }else{
  #   return(t)
  # }
}

s.cali=st2
y=ytrain2

weighted.quantile = function(v, prob, w=NULL, sorted=FALSE) {
  if (is.null(w)) w = rep(1,length(v))
  if (!sorted) { o = order(v); v = v[o]; w = w[o] }
  i = which(cumsum(w/sum(w)) >= prob)
  if (length(i)==0) return(Inf) # Can happen with infinite weights
  else return(v[min(i)])
}



min.t<-function(s.cali,y,alpha,K=5){
  sy=cbind(s.cali,y)
  ts=apply(sy,1,FUN = E.score,K=K)
  q=weighted.quantile(c(ts,Inf),1-alpha,w=NULL,sorted=FALSE)
  return(q)
  # e=1/nrow(s.cali)+quantile(tse,(1-alpha))*nrow(s.cali)/(1+nrow(s.cali))
  # print(length(ts))
  # print(length(ecdf(ts)))
  # ts=nrow(s.cali)*ts/(nrow(s.cali)+1)+1/(n+1)
  # return(sort(tse, TRUE)[ceiling((1-alpha)*(1+nrow(s.cali)))])
  # return(quantile(e,(1-alpha)*(1+1/nrow(s.cali))))
  # return(e)
}


cscore<-function(s,t,K=5,randomness=F){
  #for each x, construct the prediction set k
  k=rep(0,K+1)
  #if tau>1, output an empty set filled with 0.
  if(t==Inf){
    return(k)
  }else{
    #order the conditional probability S by decreasing
    b=s[order(s,decreasing=TRUE)]
    #the corresponding order of b
    order.index=order(s,decreasing=TRUE)
    #`cumsum` the  the conditional probability b
    c=cumsum(b)
    #find the minimum index where the `cusum` is largest than threshold \tau
    order_=order.index[1:which(c>=t)[1]]
    n=length(order_)
    #consider randomness
    if(randomness){
      u=runif(1)
      v=(sum(b[1:n])-t)/(b[n])
      if(u<v){
        #if u<v, output L-1 (n-1) largest labels
        for (z in order_[-n]) {
          k[z]=1
        }
        }else{
          #if u>=v, output L (n) largest labels
          for (z in order_) {
            k[z]=1
          }
        }
    #don't consider randomness
    }else{
      for (z in order_) {
        k[z]=1
      }
    }
    return(k)
  }
}


cscore.final<-function(s,t,K=5){
  cscores= matrix(data = NA, nrow = nrow(s), ncol = K+1)
  for (i in 1:nrow(s)) {
    cscores[i,]=cscore(s[i,],t)
  }
  return(cscores)
}

OptInsample.loop1<-function(n,n_train,n_test,alpha=0.05,seed=F,K=5){
  averagetypeI = list()
  averagetypeII = list()
  for (round in 1:n) {
    set.seed(round)
    GenDat<-GenData(n_train,n_test)
    xtrain = as.matrix(GenDat$trainData)
    ytrain = as.factor(train.label[GenDat$trainLabel])
    xtest = as.matrix(GenDat$testData)
    ytest = as.factor(GenDat$train.label.select[GenDat$testLabel])
    # foldid = sample(1:2, length(ytrain), replace = TRUE);
    foldid=rep(2,length(ytrain))
    foldid_1 = sample(1:length(ytrain), length(ytrain)*0.5, replace = FALSE)#sample(1:2, length(ytrain), replace = TRUE);
    foldid[foldid_1]=1
    foldid_te = sample(1:2,length(ytest), replace = TRUE)
    xtrain1 = xtrain[foldid==1,]; xtrain2 = xtrain[foldid==2,];
    ytrain1 = ytrain[foldid==1]; ytrain2=ytrain[foldid==2]
    xtest1 = xtest[foldid_te ==1,]; xtest2 = xtest[foldid_te==2,];
    labels = sort(unique(ytrain))
    cat('Dim of train data1',dim(xtrain1),length(ytrain1))
    cat('\nDim of train data2',dim(xtrain2),length(ytrain2))
    cat('\nDim of test data',dim(xtest),length(ytest))
    dat<-data.frame(xtrain1,response=ytrain1)
    date<-data.frame(xtest,response=ytest)
    dat2<-data.frame(xtrain2,response=ytrain2)
  
    
    
    fit<-ranger(formula = response~., data = dat, num.trees = 500,probability = T)
    ste<-predict(fit,date)$predictions
    st2<-predict(fit,dat2)$predictions
    
    
    label.list = as.numeric(as.character(sort(unique(ytrain1))))
    t=min.t(st2,ytrain2,alpha)
    cat('\nt is',t,'\n')
    Cscores=cscore.final(ste,t)
    errors=type.errors(Cscores = Cscores,ylabel = ytest,K=K,comparison=F,alpha=alpha)
    averagetypeI[[round]]=errors$typeI
    averagetypeII[[round]]=errors$typeII
  }
  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII))
}


average.errors.OptInsample.loop=OptInsample.loop1(n=2,n_train=500,n_test=500,alpha=0.05)
print(average.errors.OptInsample.loop$averagetypeI)
print(average.errors.OptInsample.loop$averagetypeII)
average.errors.OptInsample=data_mean(average.errors.OptInsample.loop)
print(average.errors.OptInsample)

##Baseline4:OptInsample.shift
print('Baseline4:OptInsample.shift')
spreate.class<-function(x1,xte){
  dat<-rbind(x1,xte)
  temp<-ncol(dat)
  dat<-cbind(dat,(as.data.frame(c(rep(0,nrow(x1)),rep(1,nrow(xte))))))
  colnames(dat)<-c(paste0("V", (1:temp)),'response')
  dat$response<-as.factor(dat$response)
  data_class<-ranger(formula = response~., data = dat,num.trees =  500, probability = TRUE)
  return(data_class)
}
gamma.x<-function(x0,x.tr2,r.xtr2,model,same=TRUE){
  numerator<-as.numeric(predict(model, x.tr2)$predictions[,2])/as.numeric(predict(model, x.tr2)$predictions[,1]+1e-6)
  Denominator<-as.numeric(predict(model, x0)$predictions[,2])/as.numeric(predict(model, x0)$predictions[,1]+1e-6)+r.xtr2
  if(same==TRUE){
    return(numerator/Denominator)
  }else{
    v <- apply(as.matrix(Denominator), 1, function(x) numerator / x) # datNormed[,1] 给定 test的一个x，对应train2
    return(v)
  }
}


x=x.te2
y.tr2=ytrain2
s.cali=st2
model=model.xte2
alpha=0.05
x0=x

E.score<-function(sy0,K=5,randomness=T){
  n=length(sy0)
  y=sy0[n]
  a=sy0[-n]
  b=a[order(a,decreasing=TRUE)]
  c=cumsum(b)
  order.index=order(a,decreasing=TRUE)
  v=c[which(order.index==(y))]
  u=runif(1)
  if(randomness){
    return(v-b[which(order.index==(y))]*u)}
  else(
    return(v-b[which(order.index==(y))])
  )
}

x=x.te2
y.tr2=ytrain2
s.cali=st2
model=model.xte2
sy0=sy[18,]
library(spatstat.geom)

weighted.quantile = function(v, prob, w=NULL, sorted=FALSE) {
  if (is.null(w)) w = rep(1,length(v))
  if (!sorted) { o = order(v); v = v[o]; w = w[o] }
  i = which(cumsum(w/sum(w)) >= prob)
  if (length(i)==0) return(Inf) # Can happen with infinite weights
  else return(v[min(i)])
}



min.t.shift<-function(x,x.tr2,y.tr2,s.cali,r.xtr2,model,alpha,K=5){
  sy=cbind(s.cali,y.tr2)
  ts=apply(sy,1,FUN = E.score,K=K)
  gamma1=gamma.x(x,x.tr2,r.xtr2,model,same = FALSE)
  gamma2=gamma.x(x,x,r.xtr2,model,same = TRUE)
  tsw=c(ts,1)
  mints=c()
  n0=nrow(x)
  # cat('n0',n0)
  for (i in 1:n0) {
    # print(i)
    w1=gamma1[,i]
    w2=gamma2[i]
    mints[i]=weighted.quantile(c(ts,Inf),1-alpha,w=c(w1,w2),sorted=FALSE)
  }
  # for (i in 1:nrow(x)) {
  #   weights=c(gamma1[,i],gamma2[i])
  #   ecdw=ewcdf(tsw,weights)
  #   tsww=ecdw(tsw)
  #   mints[i]=quantile(tsww,(1-alpha)*(1+1/nrow(x)))
  # }
  # mean(mints)
  # tse=ecdf(ts)
  # E=tse(ts)
  # hist(E)
  # hist(E_weighted)
  # set.seed(42)
  # hist(E,col='skyblue',border=F)
  # hist(E_weighted,add=T,col=scales::alpha('red',.5),border=F)
  # plot(ecdf(E),col="skyblue")
  # abline(v = 0.7828125, col = "darkblue")
  # plot(ecdf(E_weighted),col="red",add=T)
  # abline(v = 0.5104595 , col = "darkred")
  # quantile((E),(1-alpha)*(1+1/nrow(x)))
  # E_weighted=1.01*gamma2+t(as.matrix(gamma1))%*%E
  # t=quantile((E_weighted),(1-alpha)*(1+1/nrow(x)))
  return(mints)
}


s=ste[foldid_te ==2,][130,]
t=ts.te2
cscore.shift<-function(s,t,K=5,randomness=T){
  k=rep(0,K+1)
  if(t==Inf){
    return(k)
  }else{
    b=s[order(s,decreasing=TRUE)]
    order.index=order(s,decreasing=TRUE)
    c=cumsum(b)
    k=rep(0,K+1)
    order_=order.index[1:which(c>=t)[1]]
    n=length(order_)
    if(randomness){
      u=runif(1)
      v=(sum(b[1:n])-t)/(b[n])
      if(u<=v){
        if(n!=1){
          for (z in order_[-n]) {
            k[z]=1
          }
        }
      }else{
        for (z in order_) {
          k[z]=1
        }
      }
    }else{
      for (z in order_) {
        k[z]=1
      }
    }
    return(k)
  }
}
cscore.shift.final<-function(s,t,K=5){
  cscores= matrix(data = NA, nrow = nrow(s), ncol = K+1)
  for (i in 1:nrow(s)) {
    cscores[i,]=cscore.shift(s[i,],t[i])
  }
  return(cscores)
}


OptInsample.shift.loop1<-function(n,n_train,n_test,alpha=0.05,seed=F,K=5){
  averagetypeI = list()
  averagetypeII = list()
  alphas=c()
  for (round in 1:n) {
    set.seed(round)
    GenDat<-GenData(n_train,n_test)
    xtrain = GenDat$trainData
    ytrain = as.factor(train.label[GenDat$trainLabel])
    xtest = GenDat$testData
    ytest = as.factor(GenDat$train.label.select[GenDat$testLabel])
    #foldid=sample(1:2, length(ytrain), replace = TRUE);
    foldid=rep(2,length(ytrain))
    foldid_1 = sample(1:length(ytrain), length(ytrain)*0.9, replace = FALSE)#sample(1:2, length(ytrain), replace = TRUE);
    foldid[foldid_1]=1
    foldid_te = sample(1:2,length(ytest), replace = TRUE)
    xtrain1 = xtrain[foldid==1,]; xtrain2 = xtrain[foldid==2,];
    ytrain1 = ytrain[foldid==1]; ytrain2=ytrain[foldid==2]
    xtest1 = xtest[foldid_te ==1,]; xtest2 = xtest[foldid_te==2,];
    label.list = sort(unique(ytrain))
    cat('Dim of train data1',dim(xtrain1),length(ytrain1))
    cat('\nDim of train data2',dim(xtrain2),length(ytrain2))
    cat('\nDim of test data',dim(xtest1))
    cat('\nDim of test data',dim(xtest2))
    temp<-ncol(xtrain1)
    x.tr2<-cbind(as.data.frame(xtrain2),as.data.frame(as.factor(c(rep(0,nrow(xtrain2))))))
    colnames(x.tr2)<-c(paste0("V", (1:temp)),'response')
    x.tr1<-cbind(as.data.frame(xtrain1),as.data.frame(as.factor(rep(0,nrow(xtrain1)))))
    colnames(x.tr1)<-c(paste0("V", (1:temp)),'response')
    x.te2<-cbind(as.data.frame(xtest2),as.data.frame(as.factor(rep(1,nrow(xtest2)))))
    colnames(x.te2)<-c(paste0("V", (1:temp)),'response')
    x.te1<-cbind(xtest1,as.data.frame(as.factor(rep(1,nrow(xtest1)))))
    colnames(x.te1)<-c(paste0("V", (1:temp)),'response')
    dat<-data.frame(xtrain1,response=ytrain1)
    date<-data.frame(xtest,response=ytest)
    dat2<-data.frame(xtrain2,response=ytrain2)
    
    fit<-ranger(formula = response~., data = dat, num.trees = 500,probability = T)
    ste<-predict(fit,date)$predictions
    st2<-predict(fit,dat2)$predictions
    #Predict test2
    model.xte2<-spreate.class(xtrain1,xtest2)
    r.xtr2<-sum(as.numeric(predict(model.xte2, x.tr2)$predictions[,2])/as.numeric(predict(model.xte2, x.tr2)$predictions[,1]+1e-9))

    ts.te2<-min.t.shift(x.te2,x.tr2,ytrain2,st2,r.xtr2,model.xte2,alpha=alpha)
    # print(table(ts.te2))
    a<-as.data.frame(table(ts.te2))
    for (t in 1:dim(a)[1]) {
      alphas=append(alphas,rep(as.numeric(as.character(a[t,][1][1,])),as.numeric(as.character(a[t,][2]))))
    }
    C_hat.Opt.shift.2<-cscore.shift.final(ste[foldid_te ==2,],ts.te2)
    # cbind(C_hat.Opt.shift.2,ytest[foldid_te ==2])
    #Predict test1
    model.xte1<-spreate.class(xtrain1,xtest1)
    r.xtr1<-sum(as.numeric(predict(model.xte1, x.tr2)$predictions[,2])/as.numeric(predict(model.xte1, x.tr2)$predictions[,1]+1e-9))
    ts.te1<-min.t.shift(x.te1,x.tr2,ytrain2,st2,r.xtr1,model.xte1,alpha=alpha)
    # print(table(ts.te1))
    a<-as.data.frame(table(ts.te1))
    for (t in 1:dim(a)[1]) {
      alphas=append(alphas,rep(as.numeric(as.character(a[t,][1][1,])),as.numeric(as.character(a[t,][2]))))
    }
    
    
    C_hat.Opt.shift.1<-cscore.shift.final(ste[foldid_te ==1,],ts.te1)
    
    C_hat.Opt.shift<-matrix(data = NA, nrow = nrow(xtest), ncol = K+1)
    C_hat.Opt.shift[foldid_te ==1,]<-C_hat.Opt.shift.1
    C_hat.Opt.shift[foldid_te ==2,]<-C_hat.Opt.shift.2
    cbind(C_hat.Opt.shift,ytest)
    
    errors=type.errors(Cscores = C_hat.Opt.shift,ylabel = ytest,K=K,comparison=F,alpha=alpha)
    averagetypeI[[round]]<-errors$typeI
    averagetypeII[[round]]<-errors$typeII
  }
  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII,alphas=alphas))
}

average.errors.OptInsample.shift.loop=OptInsample.shift.loop1(n=2,n_train=500,n_test=500,alpha=0.05)
print(average.errors.OptInsample.shift.loop$averagetypeI)
print(average.errors.OptInsample.shift.loop$averagetypeII)
average.errors.OptInsample.shift=data_mean(average.errors.OptInsample.shift.loop)
print(average.errors.OptInsample.shift)
