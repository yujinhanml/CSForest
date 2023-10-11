library(ranger)

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
  label1.index = sample(which(train.label == 1), size = n_train, replace = F)
  label2.index = sample(which(train.label == 2), size = n_train, replace = F)
  label3.index = sample(which(train.label == 3), size = n_train, replace = F)
  label4.index = sample(which(train.label == 4), size = n_train, replace = F)
  label5.index = sample(which(train.label == 5), size = n_train, replace = F)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index,label4.index,label5.index))
  trainData = trainData.full[label56.index,]
  
  #   # selecting testing data
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
  label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = F)
  label3.index.test = sample(which(train.label.select == 3), size = n_test, replace = F)
  label4.index.test = sample(which(train.label.select == 4), size = n_test, replace = F)
  label5.index.test = sample(which(train.label.select == 5), size = n_test, replace = F)
  label.outlier.test = sample(which(train.label.select == 6 | train.label.select == 7|train.label.select == 8|train.label.select == 9), size = 4*n_test_outlier, replace = F)
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test,label4.index.test,label5.index.test,label.outlier.test))
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
  label3.index = sample(which(train.label == 3), size = 100, replace = F)
  label4.index = sample(which(train.label == 4), size = 100, replace = F)
  label5.index = sample(which(train.label == 5), size = 100, replace = F)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index,label4.index,label5.index))
  trainData = trainData.full[label56.index,]
  
  #   # selecting testing data
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = F)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = F)
  label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = F)
  label3.index.test = sample(which(train.label.select == 3), size = 500, replace = F)
  label4.index.test = sample(which(train.label.select == 4), size = 500, replace = F)
  label5.index.test = sample(which(train.label.select == 5), size = 500, replace = F)
  label.outlier.test = sample(which(train.label.select == 6 | train.label.select == 7|train.label.select == 8|train.label.select == 9), size = 4*n_test_outlier, replace = F)
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test,label4.index.test,label5.index.test,label.outlier.test))
  test.select = trainData.full[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test))
}

NetGetDat = function(networkD,n_train,n_test,n_test_outlier){
  trainFull = networkD
  train.label = trainFull$class
  label0.index = sample(which(train.label == 0), size = n_train, replace = T)
  label1.index = sample(which(train.label == 1), size = n_train, replace = T)
  label2.index = sample(which(train.label == 2), size = n_train, replace = T)
  label3.index = sample(which(train.label == 3), size = n_train, replace = T)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index))
  trainData = trainFull[label56.index,]
  
  # test Data selection
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = T)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = T)
  label2.index.test = sample(which(train.label.select == 2), size = n_test, replace = T)
  label3.index.test = sample(which(train.label.select == 3), size = n_test, replace = T)
  label4.index.test = sample(which(train.label.select == 4), size = n_test_outlier, replace = T)
  label5.index.test = sample(which(train.label.select == 5), size = n_test_outlier, replace = T)
  label6.index.test = sample(which(train.label.select == 6), size = n_test_outlier, replace = T)
  label7.index.test = sample(which(train.label.select == 7), size = n_test_outlier, replace = T)
  
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test, label4.index.test,
                               label5.index.test,label4.index.test,label5.index.test,label6.index.test,label7.index.test
  ))
  test.select = trainFull[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
}


NetGetDat.exp2 = function(networkD,n_train=500,n_test=100,n_test_outlier=0){
  trainFull = networkD
  train.label = trainFull$class
  label0.index = sample(which(train.label == 0), size = n_train, replace = T)
  label1.index = sample(which(train.label == 1), size = n_train, replace = T)
  label2.index = sample(which(train.label == 2), size = n_test, replace = T)
  label3.index = sample(which(train.label == 3), size = n_test, replace = T)
  # label4.index = sample(which(train.label == 4), size = n_test, replace = T)
  # label5.index = sample(which(train.label == 5), size = n_test, replace = T)
  # label6.index = sample(which(train.label == 6), size = n_test, replace = T)
  # label7.index = sample(which(train.label == 7), size = n_test, replace = T)
  label56.index = sort(c(label0.index,label1.index,label2.index,label3.index))
  trainData = trainFull[label56.index,]
  
  # test Data selection
  train.label.select = train.label[-label56.index]
  label0.index.test = sample(which(train.label.select == 0), size = n_test, replace = T)
  label1.index.test = sample(which(train.label.select == 1), size = n_test, replace = T)
  label2.index.test = sample(which(train.label.select == 2), size = n_train, replace = T)
  label3.index.test = sample(which(train.label.select == 3), size = n_train, replace = T)
  # label4.index.test = sample(which(train.label.select == 4), size = n_train, replace = T)
  # label5.index.test = sample(which(train.label.select == 5), size = n_train, replace = T)
  # label6.index.test = sample(which(train.label.select == 6), size = n_train, replace = T)
  # label7.index.test = sample(which(train.label.select == 7), size = n_train, replace = T)
  
  label56outlier.test = sort(c(label0.index.test,label1.index.test,label2.index.test,label3.index.test))
  test.select = trainFull[-label56.index,]
  testData = test.select[label56outlier.test,]
  
  return(list(trainData = trainData, testData = testData,train.label.select = train.label.select,
              trainLabel = label56.index, testLabel =label56outlier.test,train.label=train.label))
}


weighted.quantile = function(v, prob, w=NULL, sorted=FALSE) {
  if (is.null(w)) w = rep(1,length(v))
  if (!sorted) { o = order(v); v = v[o]; w = w[o] }
  i = which(cumsum(w/sum(w)) >= prob)
  if (length(i)==0) return(Inf) # Can happen with infinite weights
  else return(v[min(i)])
}

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
  numerator<-as.numeric(predict(model, x.tr2)$predictions[,2])/as.numeric(predict(model, x.tr2)$predictions[,1]+1e-9)
  Denominator<-r.xtr2+as.numeric(predict(model, x0)$predictions[,2])/as.numeric(predict(model, x0)$predictions[,1]+1e-9)#
  
  if(same==TRUE){
    return(numerator/Denominator)
  }else{
    v <- apply(as.matrix(Denominator), 1, function(x) numerator / x) # datNormed[,1] 给定 test的一个x，对应train2
    return(v)
  }
}


E.score<-function(sy0,K=5,randomness){
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



min.t.shift<-function(x,x.tr2,y.tr2,s.cali,r.xtr2,model,alpha,K=5,randomness){
  sy=cbind(s.cali,y.tr2)
  ts=apply(sy,1,FUN = E.score,K=K,randomness=randomness)
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
  return(mints)
}

cscore.shift<-function(s,t,K=5,randomness){
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
      if(u<v){
        for (z in order_[-n]) {
            k[z]=1
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

cscore.shift.final<-function(s,t,K=5,randomness){
  cscores= matrix(data = NA, nrow = nrow(s), ncol = K+1)
  for (i in 1:nrow(s)) {
    cscores[i,]=cscore.shift(s[i,],t[i],randomness=randomness,K=K)
  }
  return(cscores)
}

OptInsample.shift.loop<-function(n,n_train,n_test,n_test_outlier,mydata='mnist',networkD=NA,train.label=NA,test.label=NA,trainData.full=NA,testData.full=NA,K=5,alpha=0.05,seed=T,type='exp1',randomness=F){
  #K=5
  averagetypeI = list()
  averagetypeII = list()
  alphas=c()
  
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
    xtrain = GenDat$trainData
    ytrain = as.factor(train.label[GenDat$trainLabel])
    xtest = GenDat$testData
    ytest = as.factor(GenDat$train.label.select[GenDat$testLabel])
    #foldid=rep(2,length(ytrain))
    #foldid_1 = sample(1:length(ytrain), length(ytrain)*0.7, replace = FALSE)#sample(1:2, length(ytrain), replace = TRUE);
    #foldid[foldid_1]=1
    foldid=sample(1:2, length(ytrain), replace = TRUE);
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
    #dim(x.tr2)
    
    #sum(as.numeric(predict(model.xte2, x.te1)$predictions[,2])/as.numeric(predict(model.xte2, x.te1)$predictions[,1]+1e-9))
    
    model.xte2<-spreate.class(xtrain1,xtest2)
    r.xtr2<-sum(as.numeric(predict(model.xte2, x.tr2)$predictions[,2])/as.numeric(predict(model.xte2, x.tr2)$predictions[,1]+1e-9))
    #cat('Becfore test 2',r.xtr2/dim(x.tr2)[1])
    #r.xtr2<-min(r.xtr2,dim(x.tr2)[1])
    #cat('After test 2',r.xtr2/dim(x.tr2)[1])
    ts.te2<-min.t.shift(x.te2,x.tr2,ytrain2,st2,r.xtr2,model.xte2,alpha=alpha,K=K,randomness=randomness)
    # print(table(ts.te2))
    a<-as.data.frame(table(ts.te2))
    for (t in 1:dim(a)[1]) {
      alphas=append(alphas,rep(as.numeric(as.character(a[t,][1][1,])),as.numeric(as.character(a[t,][2]))))
    }
    C_hat.Opt.shift.2<-cscore.shift.final(ste[foldid_te ==2,],ts.te2,randomness=randomness,K=K)
    
    #Predict test1
    model.xte1<-spreate.class(xtrain1,xtest1)
    r.xtr1<-sum(as.numeric(predict(model.xte1, x.tr2)$predictions[,2])/as.numeric(predict(model.xte1, x.tr2)$predictions[,1]+1e-9))
    #cat('Becfore test 1',r.xtr2/dim(x.tr2)[1])
    #r.xtr2<-min(r.xtr2,dim(x.tr2)[1])
    #cat('After test 1',r.xtr2/dim(x.tr2)[1])
    ts.te1<-min.t.shift(x.te1,x.tr2,ytrain2,st2,r.xtr1,model.xte1,alpha=alpha,randomness=randomness)
    # print(table(ts.te1))
    a<-as.data.frame(table(ts.te1))
    for (t in 1:dim(a)[1]) {
      alphas=append(alphas,rep(as.numeric(as.character(a[t,][1][1,])),as.numeric(as.character(a[t,][2]))))
    }
    
    
    C_hat.Opt.shift.1<-cscore.shift.final(ste[foldid_te ==1,],ts.te1,randomness=randomness,K=K)
    
    C_hat.Opt.shift<-matrix(data = NA, nrow = nrow(xtest), ncol = K+1)
    C_hat.Opt.shift[foldid_te ==1,]<-C_hat.Opt.shift.1
    C_hat.Opt.shift[foldid_te ==2,]<-C_hat.Opt.shift.2
    
    errors=type.errors(Cscores = C_hat.Opt.shift,ylabel = ytest,K=K,comparison=F,alpha=alpha)
    averagetypeI[[round]]<-errors$typeI
    averagetypeII[[round]]<-errors$typeII
    #averagetypeI.2=averagetypeI.2+errors$typeI
    #averagetypeII.2=averagetypeII.2+errors$typeII
  }
  #return(list(typeII09=as.numeric(averagetypeII[3])/n, typeII69 = as.numeric(averagetypeII[2])/n))
  return(list(averagetypeI =averagetypeI, averagetypeII = averagetypeII,alphas=alphas))
}
