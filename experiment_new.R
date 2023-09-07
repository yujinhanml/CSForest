rm(list = ls())
library(Rcpp)
library(RcppArmadillo)
library(ranger)
library(sourcetools)
library(dplyr)
library(randomForest)
library(reshape2)
library(ggplot2)
library(ranger)
library(tidyverse)
library(ggsci)
library(patchwork)
setwd('/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/')
sourceCpp(file="CSForest/src/density_gaussian.cpp")
sourceCpp(file="CSForest/src/JacknifeABcompare.cpp")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/CSForest.R")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/BCOPS.R")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/DC.R")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/CRF.R")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/ACRF.R")
source(file="/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/ACRFshift.R")



#--------- readin simulation data ------------


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
table(train.label)
#train.label
#0    1    2    3    4    5    6    7    8    9 
#5923 6742 5958 6131 5842 5421 5918 6265 5851 5949 

table(test.label)

#test.label
#0    1    2    3    4    5    6    7    8    9 
#980 1135 1032 1010  982  892  958 1028  974 1009 

data_mean<-function(data){
  datamean_1=0
  for (i in 1:length(data$averagetypeI)) {
    datamean_1=datamean_1+data$averagetypeI[[i]]
  }
  
  datamean_2=0
  for (i in 1:length(data$averagetypeII)) {
    datamean_2=datamean_2+data$averagetypeII[[i]]
  }
  
  
  cat('Type I std', sd(sapply(data$averagetypeI,"[[",7)),'Type II std', sd(sapply(data$averagetypeII,"[[",3)),
      'Length std',sd(sapply(data$averagetypeII,"[[",4)))
  return(list(averagetypeI =datamean_1/length(data$averagetypeII), averagetypeII = datamean_2/length(data$averagetypeII)))
  
}

dat_typeIII<-function(average.errors,name){
  typeIerror=sapply(average.errors$averagetypeI,"[[",7)
  typeIIerror.inliers=sapply(average.errors$averagetypeII,"[[",1)
  typeIIerror.outliers=sapply(average.errors$averagetypeII,"[[",2)
  dat<-as.data.frame(matrix(,nrow=length(typeIerror),ncol=4))
  colnames(dat)<-c('typeI','typeII.inliers','typeII.outliers','Method')
  dat$typeI<-typeIerror
  dat$typeII.inliers<-typeIIerror.inliers
  dat$typeII.outliers<-typeIIerror.outliers
  dat$Method<-rep(name,length(typeIerror))
  return(dat)
}




###EXP 1
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp1_output.txt"
sink(file_path)
print('****************CSForest****************')
average.errors.CSForest.loop=CSForest.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.CSForest.loop$averagetypeI)
print(average.errors.CSForest.loop$averagetypeII)
print('****************BCOPS****************')
average.errors.BCOPS.loop=BCOPS.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.BCOPS.loop$averagetypeI)
print(average.errors.BCOPS.loop$averagetypeII)
print('****************DC****************')
average.errors.Density.loop=Density.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.Density.loop$averagetypeI)
print(average.errors.Density.loop$averagetypeII)
print('****************CRF****************')
average.errors.Insample.loop=Insample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.Insample.loop$averagetypeI)
print(average.errors.Insample.loop$averagetypeII)
print('****************ACRF****************')
average.errors.OptInsample.loop=OptInsample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=F)
print(average.errors.OptInsample.loop$averagetypeI)
print(average.errors.OptInsample.loop$averagetypeII)
print('****************ACRFShift****************')
average.errors.OptInsample.shift.loop=OptInsample.shift.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=F)
print(average.errors.OptInsample.shift.loop$averagetypeI)
print(average.errors.OptInsample.shift.loop$averagetypeII)
print(average.errors.OptInsample.shift.loop$alphas)
# 关闭文件
sink()


# 创建各个向量
class0 <- c(0.032, 0.058, 0.05, 0.05, 0.074, 0.026, 0.046, 0.066, 0.044, 0.038)
class1 <- c(0.054, 0.044, 0.036, 0.074, 0.038, 0.058, 0.036, 0.04, 0.072, 0.056)
class2 <- c(0.048, 0.058, 0.05, 0.052, 0.052, 0.028, 0.028, 0.042, 0.076, 0.044)
class3 <- c(0.046, 0.034, 0.022, 0.036, 0.038, 0.036, 0.072, 0.07, 0.054, 0.046)
class4 <- c(0.044, 0.042, 0.036, 0.05, 0.052, 0.06, 0.04, 0.022, 0.056, 0.064)
class5 <- c(0.046, 0.084, 0.048, 0.048, 0.036, 0.048, 0.032, 0.07, 0.054, 0.058)
average_I <- c(0.045, 0.05333333, 0.04033333, 0.05166667, 0.04833333, 0.04266667, 0.04233333, 0.05166667, 0.05933333, 0.051)
oclass0 <- c(0.96, 0.93, 0.942, 0.938, 0.914, 0.964, 0.914, 0.93, 0.948, 0.956)
oclass1 <- c(0.944, 0.954, 0.964, 0.926, 0.962, 0.94, 0.964, 0.958, 0.928, 0.94)
oclass2 <- c(0.908, 0.892, 0.896, 0.906, 0.904, 0.928, 0.938, 0.928, 0.886, 0.918)
oclass3 <- c(0.842, 0.86, 0.84, 0.876, 0.856, 0.834, 0.768, 0.856, 0.84, 0.838)
oclass4 <- c(0.952, 0.942, 0.964, 0.948, 0.942, 0.936, 0.952, 0.97, 0.94, 0.932)
oclass5 <- c(0.808, 0.76, 0.756, 0.806, 0.818, 0.764, 0.802, 0.826, 0.802, 0.778)
# 创建各个向量
inlier <- c(0.064, 0.07366667, 0.07766667, 0.062, 0.06633333, 0.07633333, 0.08166667, 0.04933333, 0.065, 0.07033333)
outlier <- c(0.1175, 0.141, 0.1055, 0.118, 0.1435, 0.131, 0.128, 0.1235, 0.117, 0.1275)
average_II <- c(0.0854, 0.1006, 0.0888, 0.0844, 0.0972, 0.0982, 0.1002, 0.079, 0.0858, 0.0932)
average_length <- c(0.6622, 0.6738, 0.6698, 0.6576, 0.6718, 0.6784, 0.6798, 0.6496, 0.653, 0.6682)

CSForest.typeII <- data.frame(inlier,outlier,average_II, average_length)
CSForest.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)

# 创建各个向量
class0 <- c(0.038, 0.062, 0.04, 0.052, 0.08, 0.03, 0.04, 0.06, 0.044, 0.034)
class1 <- c(0.054, 0.038, 0.038, 0.066, 0.038, 0.058, 0.044, 0.046, 0.068, 0.054)
class2 <- c(0.052, 0.034, 0.046, 0.05, 0.044, 0.038, 0.022, 0.042, 0.072, 0.034)
class3 <- c(0.052, 0.024, 0.028, 0.04, 0.036, 0.05, 0.064, 0.064, 0.046, 0.048)
class4 <- c(0.058, 0.026, 0.036, 0.048, 0.048, 0.062, 0.056, 0.024, 0.06, 0.066)
class5 <- c(0.052, 0.074, 0.072, 0.042, 0.032, 0.056, 0.036, 0.056, 0.05, 0.064)
average_I <- c(0.051, 0.043, 0.04333333, 0.04966667, 0.04633333, 0.049, 0.04366667, 0.04866667, 0.05666667, 0.05)
oclass0 <- c(0.764, 0.746, 0.77, 0.782, 0.762, 0.776, 0.686, 0.732, 0.778, 0.682)
oclass1 <- c(0.924, 0.942, 0.946, 0.932, 0.942, 0.924, 0.952, 0.95, 0.918, 0.924)
oclass2 <- c(0.728, 0.672, 0.638, 0.646, 0.664, 0.712, 0.734, 0.702, 0.63, 0.66)
oclass3 <- c(0.654, 0.648, 0.722, 0.698, 0.708, 0.71, 0.596, 0.67, 0.672, 0.64)
oclass4 <- c(0.872, 0.866, 0.9, 0.878, 0.872, 0.87, 0.856, 0.878, 0.868, 0.814)
oclass5 <- c(0.608, 0.588, 0.574, 0.626, 0.582, 0.54, 0.632, 0.58, 0.574, 0.516)
# 创建各个向量
inlier <- c(0.204, 0.2293333, 0.2133333, 0.2043333, 0.2116667, 0.2106667, 0.227, 0.214, 0.224, 0.2596667)
outlier <- c(0.227, 0.2835, 0.2355, 0.252, 0.2855, 0.2435, 0.254, 0.2705, 0.264, 0.313)
average_II <- c(0.2132, 0.251, 0.2222, 0.2234, 0.2412, 0.2238, 0.2378, 0.2366, 0.24, 0.281)
average_length <- c(0.8378, 0.8996, 0.8468, 0.8398, 0.8776, 0.8514, 0.8778, 0.8646, 0.8622, 0.9276)

# 创建数据框
BCOPS.typeII <- data.frame(inlier,outlier,average_II, average_length)
BCOPS.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)


# 创建各个向量
class0 <- c(0.052, 0.052, 0.064, 0.074, 0.024, 0.048, 0.08, 0.066, 0.044, 0.04)
class1 <- c(0.054, 0.054, 0.03, 0.046, 0.022, 0.054, 0.034, 0.032, 0.042, 0.04)
class2 <- c(0.068, 0.068, 0.088, 0.032, 0.024, 0.054, 0.048, 0.018, 0.076, 0.048)
class3 <- c(0.04, 0.04, 0.042, 0.036, 0.068, 0.054, 0.058, 0.022, 0.05, 0.06)
class4 <- c(0.07, 0.07, 0.04, 0.04, 0.048, 0.078, 0.054, 0.04, 0.072, 0.046)
class5 <- c(0.04, 0.04, 0.062, 0.084, 0.026, 0.05, 0.03, 0.042, 0.052, 0.042)
average_I <- c(0.054, 0.04933333, 0.05433333, 0.052, 0.03533333, 0.05633333, 0.05066667, 0.03666667, 0.056, 0.046)
oclass0 <- c(0.028, 0.028, 0.034, 0.004, 0.006, 0.012, 0.02, 0.038, 0.104, 0.034)
oclass1 <- c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
oclass2 <- c(0.174, 0.14, 0.188, 0.188, 0.09, 0.19, 0.188, 0.2, 0.242, 0.212)
oclass3 <- c(0.082, 0.01, 0.05, 0.01, 0.012, 0.088, 0.056, 0.036, 0.048, 0.034)
oclass4 <- c(0.038, 0.04, 0.032, 0.008, 0.006, 0.026, 0.016, 0.02, 0.018, 0.026)
oclass5 <- c(0.162, 0.076, 0.078, 0.164, 0.1, 0.052, 0.138, 0.15, 0.12, 0.166)
# 创建各个向量
inlier <- c(0.9063333, 0.9393333, 0.9236667, 0.9246667, 0.9593333, 0.9273333, 0.9183333, 0.916, 0.893, 0.9086667)
outlier <- c(0.8365, 0.8025, 0.8355, 0.863, 0.9045, 0.7525, 0.8005, 0.872, 0.873, 0.878)
average_II <- c(0.8784, 0.8846, 0.8884, 0.9, 0.9374, 0.8574, 0.8712, 0.8984, 0.885, 0.8964)
average_length <- c(3.4392, 3.4392, 3.2708, 3.2934, 3.6344, 3.2584, 3.4008, 3.4432, 3.4234, 3.5086)

# 创建数据框
DC.typeII <- data.frame(inlier,outlier,average_II, average_length)
DC.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)


# 创建各个向量
class0 <- c(0.034, 0.08, 0.054, 0.024, 0.062, 0.044, 0.058, 0.058, 0.04, 0.052)
class1 <- c(0.05, 0.07, 0.032, 0.052, 0.028, 0.06, 0.052, 0.052, 0.06, 0.054)
class2 <- c(0.032, 0.056, 0.074, 0.062, 0.044, 0.026, 0.058, 0.034, 0.038, 0.034)
class3 <- c(0.036, 0.054, 0.042, 0.044, 0.036, 0.04, 0.084, 0.042, 0.06, 0.052)
class4 <- c(0.038, 0.046, 0.036, 0.042, 0.036, 0.034, 0.032, 0.024, 0.032, 0.066)
class5 <- c(0.04, 0.074, 0.068, 0.054, 0.06, 0.064, 0.048, 0.03, 0.058, 0.02)
average_I <- c(0.03833333, 0.06333333, 0.051, 0.04633333, 0.04433333, 0.04466667, 0.05533333, 0.04, 0.048, 0.04633333)
oclass0 <- c(0.958, 0.92, 0.946, 0.964, 0.936, 0.952, 0.934, 0.936, 0.956, 0.946)
oclass1 <- c(0.95, 0.93, 0.968, 0.948, 0.966, 0.934, 0.948, 0.948, 0.934, 0.94)
oclass2 <- c(0.912, 0.912, 0.908, 0.9, 0.912, 0.936, 0.918, 0.926, 0.928, 0.93)
oclass3 <- c(0.85, 0.84, 0.886, 0.876, 0.892, 0.852, 0.818, 0.786, 0.818, 0.828)
oclass4 <- c(0.96, 0.938, 0.964, 0.958, 0.952, 0.962, 0.96, 0.96, 0.944, 0.924)
oclass5 <- c(0.73, 0.822, 0.826, 0.806, 0.776, 0.792, 0.832, 0.818, 0.834, 0.828)
inlier <- c(0.09133333, 0.06333333, 0.053, 0.073, 0.07433333, 0.07333333, 0.069, 0.08866667, 0.07833333, 0.07933333)
outlier <- c(0.7295, 0.6705, 0.585, 0.7075, 0.7485, 0.779, 0.6885, 0.841, 0.799, 0.779)
average_II <- c(0.3466, 0.3062, 0.2658, 0.3268, 0.344, 0.3556, 0.3168, 0.3896, 0.3666, 0.3592)
average_length <- c(0.9492, 0.8806, 0.8416, 0.9194, 0.9398, 0.9562, 0.8942, 1.0078, 0.9682, 0.9568)

# 创建数据框
CRF.typeII <- data.frame(inlier,outlier,average_II, average_length)
CRF.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)

# 创建各个向量
class0 <- c(0.014, 0.02, 0.02, 0.02, 0.01, 0.01, 0.028, 0.024, 0.024, 0.012)
class1 <- c(0.024, 0.016, 0.014, 0.026, 0.014, 0.026, 0.02, 0.018, 0.024, 0.038)
class2 <- c(0.042, 0.066, 0.056, 0.072, 0.038, 0.04, 0.07, 0.048, 0.076, 0.046)
class3 <- c(0.088, 0.096, 0.078, 0.076, 0.086, 0.09, 0.12, 0.076, 0.116, 0.092)
class4 <- c(0.012, 0.024, 0.014, 0.008, 0.028, 0.01, 0.016, 0.01, 0.016, 0.034)
class5 <- c(0.074, 0.096, 0.072, 0.08, 0.082, 0.044, 0.078, 0.056, 0.068, 0.056)
average_I <- c(0.04233333, 0.053, 0.04233333, 0.047, 0.043, 0.03666667, 0.05533333, 0.03866667, 0.054, 0.04633333)
oclass0 <- c(0.986, 0.98, 0.972, 0.98, 0.97, 0.978, 0.972, 0.976, 0.976, 0.988)
oclass1 <- c(0.976, 0.984, 0.984, 0.974, 0.98, 0.962, 0.98, 0.982, 0.976, 0.962)
oclass2 <- c(0.958, 0.934, 0.944, 0.928, 0.944, 0.948, 0.93, 0.952, 0.924, 0.954)
oclass3 <- c(0.912, 0.904, 0.91, 0.924, 0.898, 0.894, 0.88, 0.92, 0.884, 0.908)
oclass4 <- c(0.988, 0.976, 0.984, 0.992, 0.964, 0.978, 0.984, 0.99, 0.984, 0.966)
oclass5 <- c(0.926, 0.904, 0.916, 0.92, 0.908, 0.942, 0.922, 0.942, 0.932, 0.944)
inlier <- c(0.04233333, 0.053, 0.04833333, 0.047, 0.056, 0.04966667, 0.05533333, 0.03966667, 0.054, 0.04633333)
outlier <- rep(1, 10)
average_II <- c(0.4254, 0.4318, 0.429, 0.4282, 0.4336, 0.4298, 0.4332, 0.4238, 0.4324, 0.4278)
average_length <- rep(1, 10)
# 创建数据框
ACRF.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)
ACRF.typeII <- data.frame(inlier,outlier,average_II, average_length)

# 创建各个向量
class0 <- c(0.014, 0.02, 0.02, 0.02, 0.01, 0.01, 0.028, 0.024, 0.024, 0.012)
class1 <- c(0.024, 0.016, 0.014, 0.026, 0.014, 0.026, 0.02, 0.018, 0.024, 0.038)
class2 <- c(0.042, 0.066, 0.056, 0.072, 0.038, 0.04, 0.07, 0.048, 0.076, 0.046)
class3 <- c(0.088, 0.096, 0.078, 0.076, 0.086, 0.09, 0.12, 0.076, 0.116, 0.092)
class4 <- c(0.012, 0.024, 0.014, 0.008, 0.028, 0.01, 0.016, 0.01, 0.016, 0.034)
class5 <- c(0.074, 0.096, 0.072, 0.08, 0.082, 0.044, 0.078, 0.056, 0.068, 0.056)
average_I <- c(0.04233333, 0.053, 0.04233333, 0.047, 0.043, 0.03666667, 0.05533333, 0.03866667, 0.054, 0.04633333)
oclass0 <- c(0.986, 0.98, 0.972, 0.98, 0.97, 0.978, 0.972, 0.976, 0.976, 0.988)
oclass1 <- c(0.976, 0.984, 0.984, 0.974, 0.98, 0.962, 0.98, 0.982, 0.976, 0.962)
oclass2 <- c(0.958, 0.934, 0.944, 0.928, 0.944, 0.948, 0.93, 0.952, 0.924, 0.954)
oclass3 <- c(0.912, 0.904, 0.91, 0.924, 0.898, 0.894, 0.88, 0.92, 0.884, 0.908)
oclass4 <- c(0.988, 0.976, 0.984, 0.992, 0.964, 0.978, 0.984, 0.99, 0.984, 0.966)
oclass5 <- c(0.926, 0.904, 0.916, 0.92, 0.908, 0.942, 0.922, 0.942, 0.932, 0.944)
# 创建各个向量
inlier <- c(0.054, 0.06666667, 0.054, 0.07533333, 0.05366667, 0.07633333, 0.05066667, 0.06166667, 0.06466667, 0.098)
outlier <- c(0.982, 0.9735, 0.979, 0.979, 0.9875, 0.9865, 0.98, 0.9815, 0.985, 0.984)
average_II <- c(0.4252, 0.4294, 0.424, 0.4368, 0.4272, 0.4404, 0.4226, 0.4296, 0.4328, 0.4524)
average_length <- c(1.1786, 1.1974, 1.1594, 1.271, 1.1252, 1.2762, 1.0448, 1.2398, 1.1986, 1.335)

# 创建数据框
ACRFshift.typeI <- data.frame(class0,class1,class2,class3,class4,class5,average_I,oclass0,oclass1,oclass2,oclass3,oclass4,oclass5)
ACRFshift.typeII <- data.frame(inlier,outlier,average_II, average_length)


### 结论1
# 假设数据框为 df，包含列 CSForest.typeI、BCOPS.typeI、DC.typeI、CRF.typeI、ACRF.typeI、ACRFshift.typeI
# 计算平均值
df=CSForest.typeII
means <- colMeans(df)

# 计算方差
variances <- apply(df, 2, sd)

# 输出结果
cat("列名\t\t平均值\t\t标准差\n")
for (i in 1:ncol(df)) {
  col_name <- names(df)[i]
  mean_val <- means[i]
  var_val <- variances[i]
  cat(col_name, "\t", mean_val, "\t", var_val, "\n")
}

# 画图 per_class
outputs_1<-matrix(0, ncol = 8, nrow = 14)
outputs_1<-as.data.frame(outputs_1)

colnames(outputs_1)<-c("prediction","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_1$prediction<-rep(c('C(x)={y}','multilabel'),7)
outputs_1$number<-c(0,0,1,1,2,2,3,3,4,4,5,5,'R','R')
outputs_1$CSForest<-c(colMeans(CSForest.typeI)[8],
                      1-colMeans(CSForest.typeI)[1]-colMeans(CSForest.typeI)[8],
                      colMeans(CSForest.typeI)[9],
                      1-colMeans(CSForest.typeI)[2]-colMeans(CSForest.typeI)[9],
                      colMeans(CSForest.typeI)[10],
                      1-colMeans(CSForest.typeI)[3]-colMeans(CSForest.typeI)[10],
                      colMeans(CSForest.typeI)[11],
                      1-colMeans(CSForest.typeI)[4]-colMeans(CSForest.typeI)[11],
                      colMeans(CSForest.typeI)[12],
                      1-colMeans(CSForest.typeI)[5]-colMeans(CSForest.typeI)[12],
                      colMeans(CSForest.typeI)[13],
                      1-colMeans(CSForest.typeI)[6]-colMeans(CSForest.typeI)[13],
                      1-colMeans(CSForest.typeII)[2],
                      0)

outputs_1$BCOPS<-c(colMeans(BCOPS.typeI)[8],
                   1-colMeans(BCOPS.typeI)[1]-colMeans(BCOPS.typeI)[8],
                   colMeans(BCOPS.typeI)[9],
                   1-colMeans(BCOPS.typeI)[2]-colMeans(BCOPS.typeI)[9],
                   colMeans(BCOPS.typeI)[10],
                   1-colMeans(BCOPS.typeI)[3]-colMeans(BCOPS.typeI)[10],
                   colMeans(BCOPS.typeI)[11],
                   1-colMeans(BCOPS.typeI)[4]-colMeans(BCOPS.typeI)[11],
                   colMeans(BCOPS.typeI)[12],
                   1-colMeans(BCOPS.typeI)[5]-colMeans(BCOPS.typeI)[12],
                   colMeans(BCOPS.typeI)[13],
                   1-colMeans(BCOPS.typeI)[6]-colMeans(BCOPS.typeI)[13],
                   1-colMeans(BCOPS.typeII)[2],
                   0)

outputs_1$DC<-c(colMeans(DC.typeI)[8],
                1-colMeans(DC.typeI)[1]-colMeans(DC.typeI)[8],
                colMeans(DC.typeI)[9],
                1-colMeans(DC.typeI)[2]-colMeans(DC.typeI)[9],
                colMeans(DC.typeI)[10],
                1-colMeans(DC.typeI)[3]-colMeans(DC.typeI)[10],
                colMeans(DC.typeI)[11],
                1-colMeans(DC.typeI)[4]-colMeans(DC.typeI)[11],
                colMeans(DC.typeI)[12],
                1-colMeans(DC.typeI)[5]-colMeans(DC.typeI)[12],
                colMeans(DC.typeI)[13],
                1-colMeans(DC.typeI)[6]-colMeans(DC.typeI)[13],
                1-colMeans(DC.typeII)[2],
                0)
outputs_1$CRF<-c(colMeans(CRF.typeI)[8],
                 1-colMeans(CRF.typeI)[1]-colMeans(CRF.typeI)[8],
                 colMeans(CRF.typeI)[9],
                 1-colMeans(CRF.typeI)[2]-colMeans(CRF.typeI)[9],
                 colMeans(CRF.typeI)[10],
                 1-colMeans(CRF.typeI)[3]-colMeans(CRF.typeI)[10],
                 colMeans(CRF.typeI)[11],
                 1-colMeans(CRF.typeI)[4]-colMeans(CRF.typeI)[11],
                 colMeans(CRF.typeI)[12],
                 1-colMeans(CRF.typeI)[5]-colMeans(CRF.typeI)[12],
                 colMeans(CRF.typeI)[13],
                 1-colMeans(CRF.typeI)[6]-colMeans(CRF.typeI)[13],
                 1-colMeans(CRF.typeII)[2],
                 0)


outputs_1$ACRF<-c(colMeans(ACRF.typeI)[8],
                  1-colMeans(ACRF.typeI)[1]-colMeans(ACRF.typeI)[8],
                  colMeans(ACRF.typeI)[9],
                  1-colMeans(ACRF.typeI)[2]-colMeans(ACRF.typeI)[9],
                  colMeans(ACRF.typeI)[10],
                  1-colMeans(ACRF.typeI)[3]-colMeans(ACRF.typeI)[10],
                  colMeans(ACRF.typeI)[11],
                  1-colMeans(ACRF.typeI)[4]-colMeans(ACRF.typeI)[11],
                  colMeans(ACRF.typeI)[12],
                  1-colMeans(ACRF.typeI)[5]-colMeans(ACRF.typeI)[12],
                  colMeans(ACRF.typeI)[13],
                  1-colMeans(ACRF.typeI)[6]-colMeans(ACRF.typeI)[13],
                  1-colMeans(ACRF.typeII)[2],
                  0)
outputs_1$ACRFshift<-c(colMeans(ACRFshift.typeI)[8],
                       1-colMeans(ACRFshift.typeI)[1]-colMeans(ACRFshift.typeI)[8],
                       colMeans(ACRFshift.typeI)[9],
                       1-colMeans(ACRFshift.typeI)[2]-colMeans(ACRFshift.typeI)[9],
                       colMeans(ACRFshift.typeI)[10],
                       1-colMeans(ACRFshift.typeI)[3]-colMeans(ACRFshift.typeI)[10],
                       colMeans(ACRFshift.typeI)[11],
                       1-colMeans(ACRFshift.typeI)[4]-colMeans(ACRFshift.typeI)[11],
                       colMeans(ACRFshift.typeI)[12],
                       1-colMeans(ACRFshift.typeI)[5]-colMeans(ACRFshift.typeI)[12],
                       colMeans(ACRFshift.typeI)[13],
                       1-colMeans(ACRFshift.typeI)[6]-colMeans(ACRFshift.typeI)[13],
                       1-colMeans(ACRFshift.typeII)[2],
                       0)

outputs_1 <-reshape2::melt(outputs_1 ,id=c('prediction','number'))
colnames(outputs_1)<-c('prediction','number','Method','Coverage')
outputs_1$Coverage<-100*outputs_1$Coverage
outputs_1$x<-ifelse(outputs_1$number==0,1,
                    ifelse(outputs_1$number==1,5,
                           ifelse(outputs_1$number==2,9,
                                  ifelse(outputs_1$number==3,13,
                                         ifelse(outputs_1$number==4,17,
                                                ifelse(outputs_1$number==5,21,25))))))
df1<-outputs_1[outputs_1$Method=='CSForest',]
df2<-outputs_1[outputs_1$Method=='BCOPS',]
df3<-outputs_1[outputs_1$Method=='DC',]
df4<-outputs_1[outputs_1$Method=='CRF',]
df5<-outputs_1[outputs_1$Method=='ACRF',]
df6<-outputs_1[outputs_1$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1,
           aes(x=x,y=Coverage,fill=prediction),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2,aes(x=x+0.5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3,aes(x=x+0.5*2,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4,aes(x=x+0.5*3,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5,aes(x=x+0.5*4,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6,aes(x=x+0.5*5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5,
                                25,25.5,26,26.5,27,27.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed",size=1)+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   3                   4                    5                  R')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))
# 8.46*6.58
# 画图 length

# 提取各列的 average_length 数据
csforest_average_length <- CSForest.typeII$average_length
bcops_average_length <- BCOPS.typeII$average_length
dc_average_length <- DC.typeII$average_length
crf_average_length <- CRF.typeII$average_length
acrf_average_length <- ACRF.typeII$average_length
acrfshift_average_length <- ACRFshift.typeII$average_length

# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(csforest_average_length)), levels = method_order))
data$Average_Length <- c(csforest_average_length, bcops_average_length, dc_average_length, crf_average_length, acrf_average_length, acrfshift_average_length)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()

# Mice Protein Expression

# CIFAR-10
library(reticulate)

load_image_file <- function(filename) {
  np <- import("numpy")
  data <- np$load(filename)
  return(data)
}

load_label_file <- function(filename) {
  np <- import("numpy")
  data <- np$load(filename)
  return(data)
}

# Note: Change the address here
train.file= load_image_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/train_fc_outputs_CIFAR10.npy")#load_image_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/train_fc_outputs_CIFAR10.npy")
test.file = load_image_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/test_fc_outputs_CIFAR10.npy")

train.label <- load_label_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/train_labels_CIFAR10.npy")
test.label <- load_label_file("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/test_labels_CIFAR10.npy")
trainData.full =  as.data.frame(train.file)
testData.full =  as.data.frame(test.file)
table(train.label)

# Train (60000, 32)
# Test (10000, 32)

# Experiment 1
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp1_output_CIFAR10.txt"
sink(file_path)
print('****************CSForest****************')
average.errors.CSForest.loop=CSForest.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.CSForest.loop$averagetypeI)
print(average.errors.CSForest.loop$averagetypeII)
print('****************BCOPS****************')
average.errors.BCOPS.loop=BCOPS.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.BCOPS.loop$averagetypeI)
print(average.errors.BCOPS.loop$averagetypeII)
print('****************DC****************')
average.errors.Density.loop=Density.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.Density.loop$averagetypeI)
print(average.errors.Density.loop$averagetypeII)
print('****************CRF****************')
average.errors.Insample.loop=Insample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,K=2)
print(average.errors.Insample.loop$averagetypeI)
print(average.errors.Insample.loop$averagetypeII)
print('****************ACRF****************')
average.errors.OptInsample.loop=OptInsample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=F)
print(average.errors.OptInsample.loop$averagetypeI)
print(average.errors.OptInsample.loop$averagetypeII)
print('****************ACRFShift****************')
average.errors.OptInsample.shift.loop=OptInsample.shift.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=F)
print(average.errors.OptInsample.shift.loop$averagetypeI)
print(average.errors.OptInsample.shift.loop$averagetypeII)
print(average.errors.OptInsample.shift.loop$alphas)
# 关闭文件
sink()

##
### 结论1
# 假设数据框为 df，包含列 CSForest.typeI、BCOPS.typeI、DC.typeI、CRF.typeI、ACRF.typeI、ACRFshift.typeI
# 计算平均值
df=average.errors.CSForest.loop$averagetypeI
means <- colMeans(df)

# 计算方差
variances <- apply(df, 2, sd)

# 输出结果
cat("列名\t\t平均值\t\t标准差\n")
for (i in 1:ncol(df)) {
  col_name <- names(df)[i]
  mean_val <- means[i]
  var_val <- variances[i]
  cat(col_name, "\t", mean_val, "\t", var_val, "\n")
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
  
  
  cat('Type I std', sd(sapply(data$averagetypeI,"[[",7)),'Type II std', sd(sapply(data$averagetypeII,"[[",3)),
      'Length std',sd(sapply(data$averagetypeII,"[[",4)))
  return(list(averagetypeI =datamean_1/length(data$averagetypeII), averagetypeII = datamean_2/length(data$averagetypeII)))
  
}

dat_typeIII<-function(average.errors,name){
  typeIerror=sapply(average.errors$averagetypeI,"[[",7)
  typeIIerror.inliers=sapply(average.errors$averagetypeII,"[[",1)
  typeIIerror.outliers=sapply(average.errors$averagetypeII,"[[",2)
  dat<-as.data.frame(matrix(,nrow=length(typeIerror),ncol=4))
  colnames(dat)<-c('typeI','typeII.inliers','typeII.outliers','Method')
  dat$typeI<-typeIerror
  dat$typeII.inliers<-typeIIerror.inliers
  dat$typeII.outliers<-typeIIerror.outliers
  dat$Method<-rep(name,length(typeIerror))
  return(dat)
}

average.errors.CSForest=data_mean(average.errors.CSForest.loop)

average.errors.BCOPS=data_mean(average.errors.BCOPS.loop)

average.errors.Density=data_mean(average.errors.Density.loop)

average.errors.Insample=data_mean(average.errors.Insample.loop)

average.errors.OptInsample=data_mean(average.errors.OptInsample.loop)

average.errors.OptInsample.shift=data_mean(average.errors.OptInsample.shift.loop)


# 画图 per_class
outputs_1<-matrix(0, ncol = 8, nrow = 14)
outputs_1<-as.data.frame(outputs_1)

colnames(outputs_1)<-c("prediction","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_1$prediction<-rep(c('C(x)={y}','multilabel'),7)
outputs_1$number<-c(0,0,1,1,2,2,3,3,4,4,5,5,'R','R')
outputs_1$CSForest<-c(average.errors.CSForest$averagetypeI[8],
                      1-average.errors.CSForest$averagetypeI[1]-average.errors.CSForest$averagetypeI[8],
                      average.errors.CSForest$averagetypeI[9],
                      1-average.errors.CSForest$averagetypeI[2]-average.errors.CSForest$averagetypeI[9],
                      average.errors.CSForest$averagetypeI[10],
                      1-average.errors.CSForest$averagetypeI[3]-average.errors.CSForest$averagetypeI[10],
                      average.errors.CSForest$averagetypeI[11],
                      1-average.errors.CSForest$averagetypeI[4]-average.errors.CSForest$averagetypeI[11],
                      average.errors.CSForest$averagetypeI[12],
                      1-average.errors.CSForest$averagetypeI[5]-average.errors.CSForest$averagetypeI[12],
                      average.errors.CSForest$averagetypeI[13],
                      1-average.errors.CSForest$averagetypeI[6]-average.errors.CSForest$averagetypeI[13],
                      1-average.errors.CSForest$averagetypeII[2],
                      0)

outputs_1$BCOPS<-c(average.errors.BCOPS$averagetypeI[8],
                   1-average.errors.BCOPS$averagetypeI[1]-average.errors.BCOPS$averagetypeI[8],
                   average.errors.BCOPS$averagetypeI[9],
                   1-average.errors.BCOPS$averagetypeI[2]-average.errors.BCOPS$averagetypeI[9],
                   average.errors.BCOPS$averagetypeI[10],
                   1-average.errors.BCOPS$averagetypeI[3]-average.errors.BCOPS$averagetypeI[10],
                   average.errors.BCOPS$averagetypeI[11],
                   1-average.errors.BCOPS$averagetypeI[4]-average.errors.BCOPS$averagetypeI[11],
                   average.errors.BCOPS$averagetypeI[12],
                   1-average.errors.BCOPS$averagetypeI[5]-average.errors.BCOPS$averagetypeI[12],
                   average.errors.BCOPS$averagetypeI[13],
                   1-average.errors.BCOPS$averagetypeI[6]-average.errors.BCOPS$averagetypeI[13],
                   1-average.errors.BCOPS$averagetypeII[2],
                   0)

outputs_1$DC<-c(average.errors.Density$averagetypeI[8],
                1-average.errors.Density$averagetypeI[1]-average.errors.Density$averagetypeI[8],
                average.errors.Density$averagetypeI[9],
                1-average.errors.Density$averagetypeI[2]-average.errors.Density$averagetypeI[9],
                average.errors.Density$averagetypeI[10],
                1-average.errors.Density$averagetypeI[3]-average.errors.Density$averagetypeI[10],
                average.errors.Density$averagetypeI[11],
                1-average.errors.Density$averagetypeI[4]-average.errors.Density$averagetypeI[11],
                average.errors.Density$averagetypeI[12],
                1-average.errors.Density$averagetypeI[5]-average.errors.Density$averagetypeI[12],
                average.errors.Density$averagetypeI[13],
                1-average.errors.Density$averagetypeI[6]-average.errors.Density$averagetypeI[13],
                1-average.errors.Density$averagetypeII[2],
                0)
outputs_1$CRF<-c(average.errors.Insample$averagetypeI[8],
                 1-average.errors.Insample$averagetypeI[1]-average.errors.Insample$averagetypeI[8],
                 average.errors.Insample$averagetypeI[9],
                 1-average.errors.Insample$averagetypeI[2]-average.errors.Insample$averagetypeI[9],
                 average.errors.Insample$averagetypeI[10],
                 1-average.errors.Insample$averagetypeI[3]-average.errors.Insample$averagetypeI[10],
                 average.errors.Insample$averagetypeI[11],
                 1-average.errors.Insample$averagetypeI[4]-average.errors.Insample$averagetypeI[11],
                 average.errors.Insample$averagetypeI[12],
                 1-average.errors.Insample$averagetypeI[5]-average.errors.Insample$averagetypeI[12],
                 average.errors.Insample$averagetypeI[13],
                 1-average.errors.Insample$averagetypeI[6]-average.errors.Insample$averagetypeI[13],
                 1-average.errors.Insample$averagetypeII[2],
                 0)


outputs_1$ACRF<-c(average.errors.OptInsample$averagetypeI[8],
                  1-average.errors.OptInsample$averagetypeI[1]-average.errors.OptInsample$averagetypeI[8],
                  average.errors.OptInsample$averagetypeI[9],
                  1-average.errors.OptInsample$averagetypeI[2]-average.errors.OptInsample$averagetypeI[9],
                  average.errors.OptInsample$averagetypeI[10],
                  1-average.errors.OptInsample$averagetypeI[3]-average.errors.OptInsample$averagetypeI[10],
                  average.errors.OptInsample$averagetypeI[11],
                  1-average.errors.OptInsample$averagetypeI[4]-average.errors.OptInsample$averagetypeI[11],
                  average.errors.OptInsample$averagetypeI[12],
                  1-average.errors.OptInsample$averagetypeI[5]-average.errors.OptInsample$averagetypeI[12],
                  average.errors.OptInsample$averagetypeI[13],
                  1-average.errors.OptInsample$averagetypeI[6]-average.errors.OptInsample$averagetypeI[13],
                  1-average.errors.OptInsample$averagetypeII[2],
                  0)
outputs_1$ACRFshift<-c(average.errors.OptInsample.shift$averagetypeI[8],
                       1-average.errors.OptInsample.shift$averagetypeI[1]-average.errors.OptInsample.shift$averagetypeI[8],
                       average.errors.OptInsample.shift$averagetypeI[9],
                       1-average.errors.OptInsample.shift$averagetypeI[2]-average.errors.OptInsample.shift$averagetypeI[9],
                       average.errors.OptInsample.shift$averagetypeI[10],
                       1-average.errors.OptInsample.shift$averagetypeI[3]-average.errors.OptInsample.shift$averagetypeI[10],
                       average.errors.OptInsample.shift$averagetypeI[11],
                       1-average.errors.OptInsample.shift$averagetypeI[4]-average.errors.OptInsample.shift$averagetypeI[11],
                       average.errors.OptInsample.shift$averagetypeI[12],
                       1-average.errors.OptInsample.shift$averagetypeI[5]-average.errors.OptInsample.shift$averagetypeI[12],
                       average.errors.OptInsample.shift$averagetypeI[13],
                       1-average.errors.OptInsample.shift$averagetypeI[6]-average.errors.OptInsample.shift$averagetypeI[13],
                       1-average.errors.OptInsample.shift$averagetypeII[2],
                       0)

outputs_1 <-reshape2::melt(outputs_1 ,id=c('prediction','number'))
colnames(outputs_1)<-c('prediction','number','Method','Coverage')
outputs_1$Coverage<-100*outputs_1$Coverage
outputs_1$x<-ifelse(outputs_1$number==0,1,
                    ifelse(outputs_1$number==1,5,
                           ifelse(outputs_1$number==2,9,
                                  ifelse(outputs_1$number==3,13,
                                         ifelse(outputs_1$number==4,17,
                                                ifelse(outputs_1$number==5,21,25))))))
df1<-outputs_1[outputs_1$Method=='CSForest',]
df2<-outputs_1[outputs_1$Method=='BCOPS',]
df3<-outputs_1[outputs_1$Method=='DC',]
df4<-outputs_1[outputs_1$Method=='CRF',]
df5<-outputs_1[outputs_1$Method=='ACRF',]
df6<-outputs_1[outputs_1$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1,
           aes(x=x,y=Coverage,fill=prediction),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2,aes(x=x+0.5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3,aes(x=x+0.5*2,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4,aes(x=x+0.5*3,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5,aes(x=x+0.5*4,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6,aes(x=x+0.5*5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5,
                                25,25.5,26,26.5,27,27.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed",size=1)+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   3                   4                    5                  R')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58

# 提取各列的 average_length 数据
csforest_average_length <- sapply(average.errors.CSForest.loop$averagetypeII,"[[",4)
bcops_average_length <- sapply(average.errors.BCOPS.loop$averagetypeII,"[[",4)
dc_average_length <- sapply(average.errors.Density.loop$averagetypeII,"[[",4)
crf_average_length <- sapply(average.errors.Insample.loop$averagetypeII,"[[",4)
acrf_average_length <- sapply(average.errors.OptInsample.loop$averagetypeII,"[[",4)
acrfshift_average_length <- sapply(average.errors.OptInsample.shift.loop$averagetypeII,"[[",4)




# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(csforest_average_length)), levels = method_order))
data$Average_Length <- c(csforest_average_length, bcops_average_length, dc_average_length, crf_average_length, acrf_average_length, acrfshift_average_length)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()


#Experiment 2
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp2_output_fashionmnist.txt"
sink(file_path)
average.errors.CSForest.loop.exp2=CSForest.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.CSForest.loop.exp2$averagetypeI)
print(average.errors.CSForest.loop.exp2$averagetypeII)
average.errors.BCOPS.loop.exp2=BCOPS.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.BCOPS.loop.exp2$averagetypeI)
print(average.errors.BCOPS.loop.exp2$averagetypeII)
average.errors.Density.loop.exp2=Density.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.Density.loop.exp2$averagetypeI)
print(average.errors.Density.loop.exp2$averagetypeII)
average.errors.Insample.loop.exp2=Insample.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
print(average.errors.Insample.loop.exp2$averagetypeI)
print(average.errors.Insample.loop.exp2$averagetypeII)
average.errors.OptInsample.loop.exp2=OptInsample.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=T)
print(average.errors.OptInsample.loop.exp2$averagetypeI)
print(average.errors.OptInsample.loop.exp2$averagetypeII)
average.errors.OptInsample.shift.loop.exp2=OptInsample.shift.loop(n=10,n_train=500,n_test=100,n_test_outlier=0,type='exp2',train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,randomness=T)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeI)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeII)
print(average.errors.OptInsample.shift.loop.exp2$alphas)
# 关闭文件
sink()

average.errors.CSForest.exp2=data_mean(average.errors.CSForest.loop.exp2)
average.errors.BCOPS.exp2=data_mean(average.errors.BCOPS.loop.exp2)
average.errors.Density.exp2=data_mean(average.errors.Density.loop.exp2)
average.errors.Insample.exp2=data_mean(average.errors.Insample.loop.exp2)
average.errors.OptInsample.exp2=data_mean(average.errors.OptInsample.loop.exp2)
average.errors.OptInsample.shift.exp2=data_mean(average.errors.OptInsample.shift.loop.exp2)

outputs_2<-matrix(0, ncol = 8, nrow = 12)
outputs_2<-as.data.frame(outputs_2)

colnames(outputs_2)<-c("Type","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_2$Type<-rep(c('Correct Only ','Multi-Label'),6)
outputs_2$number<-c(0,0,1,1,2,2,3,3,4,4,5,5)
outputs_2$CSForest<-c(average.errors.CSForest.exp2$averagetypeI[8],
                      1-average.errors.CSForest.exp2$averagetypeI[1]-average.errors.CSForest.exp2$averagetypeI[8],
                      average.errors.CSForest.exp2$averagetypeI[9],
                      1-average.errors.CSForest.exp2$averagetypeI[2]-average.errors.CSForest.exp2$averagetypeI[9],
                      average.errors.CSForest.exp2$averagetypeI[10],
                      1-average.errors.CSForest.exp2$averagetypeI[3]-average.errors.CSForest.exp2$averagetypeI[10],
                      average.errors.CSForest.exp2$averagetypeI[11],
                      1-average.errors.CSForest.exp2$averagetypeI[4]-average.errors.CSForest.exp2$averagetypeI[11],
                      average.errors.CSForest.exp2$averagetypeI[12],
                      1-average.errors.CSForest.exp2$averagetypeI[5]-average.errors.CSForest.exp2$averagetypeI[12],
                      average.errors.CSForest.exp2$averagetypeI[13],
                      1-average.errors.CSForest.exp2$averagetypeI[6]-average.errors.CSForest.exp2$averagetypeI[13])

outputs_2$BCOPS<-c(average.errors.BCOPS.exp2$averagetypeI[8],
                   1-average.errors.BCOPS.exp2$averagetypeI[1]-average.errors.BCOPS.exp2$averagetypeI[8],
                   average.errors.BCOPS.exp2$averagetypeI[9],
                   1-average.errors.BCOPS.exp2$averagetypeI[2]-average.errors.BCOPS.exp2$averagetypeI[9],
                   average.errors.BCOPS.exp2$averagetypeI[10],
                   1-average.errors.BCOPS.exp2$averagetypeI[3]-average.errors.BCOPS.exp2$averagetypeI[10],
                   average.errors.BCOPS.exp2$averagetypeI[11],
                   1-average.errors.BCOPS.exp2$averagetypeI[4]-average.errors.BCOPS.exp2$averagetypeI[11],
                   average.errors.BCOPS.exp2$averagetypeI[12],
                   1-average.errors.BCOPS.exp2$averagetypeI[5]-average.errors.BCOPS.exp2$averagetypeI[12],
                   average.errors.BCOPS.exp2$averagetypeI[13],
                   1-average.errors.BCOPS.exp2$averagetypeI[6]-average.errors.BCOPS.exp2$averagetypeI[13])

outputs_2$DC<-c(average.errors.Density.exp2$averagetypeI[8],
                1-average.errors.Density.exp2$averagetypeI[1]-average.errors.Density.exp2$averagetypeI[8],
                average.errors.Density.exp2$averagetypeI[9],
                1-average.errors.Density.exp2$averagetypeI[2]-average.errors.Density.exp2$averagetypeI[9],
                average.errors.Density.exp2$averagetypeI[10],
                1-average.errors.Density.exp2$averagetypeI[3]-average.errors.Density.exp2$averagetypeI[10],
                average.errors.Density.exp2$averagetypeI[11],
                1-average.errors.Density.exp2$averagetypeI[4]-average.errors.Density.exp2$averagetypeI[11],
                average.errors.Density.exp2$averagetypeI[12],
                1-average.errors.Density.exp2$averagetypeI[5]-average.errors.Density.exp2$averagetypeI[12],
                average.errors.Density.exp2$averagetypeI[13],
                1-average.errors.Density.exp2$averagetypeI[6]-average.errors.Density.exp2$averagetypeI[13])

outputs_2$CRF<-c(average.errors.Insample.exp2$averagetypeI[8],
                 1-average.errors.Insample.exp2$averagetypeI[1]-average.errors.Insample.exp2$averagetypeI[8],
                 average.errors.Insample.exp2$averagetypeI[9],
                 1-average.errors.Insample.exp2$averagetypeI[2]-average.errors.Insample.exp2$averagetypeI[9],
                 average.errors.Insample.exp2$averagetypeI[10],
                 1-average.errors.Insample.exp2$averagetypeI[3]-average.errors.Insample.exp2$averagetypeI[10],
                 average.errors.Insample.exp2$averagetypeI[11],
                 1-average.errors.Insample.exp2$averagetypeI[4]-average.errors.Insample.exp2$averagetypeI[11],
                 average.errors.Insample.exp2$averagetypeI[12],
                 1-average.errors.Insample.exp2$averagetypeI[5]-average.errors.Insample.exp2$averagetypeI[12],
                 average.errors.Insample.exp2$averagetypeI[13],
                 1-average.errors.Insample.exp2$averagetypeI[6]-average.errors.Insample.exp2$averagetypeI[13])


outputs_2$ACRF<-c(average.errors.OptInsample.exp2$averagetypeI[8],
                  1-average.errors.OptInsample.exp2$averagetypeI[1]-average.errors.OptInsample.exp2$averagetypeI[8],
                  average.errors.OptInsample.exp2$averagetypeI[9],
                  1-average.errors.OptInsample.exp2$averagetypeI[2]-average.errors.OptInsample.exp2$averagetypeI[9],
                  average.errors.OptInsample.exp2$averagetypeI[10],
                  1-average.errors.OptInsample.exp2$averagetypeI[3]-average.errors.OptInsample.exp2$averagetypeI[10],
                  average.errors.OptInsample.exp2$averagetypeI[11],
                  1-average.errors.OptInsample.exp2$averagetypeI[4]-average.errors.OptInsample.exp2$averagetypeI[11],
                  average.errors.OptInsample.exp2$averagetypeI[12],
                  1-average.errors.OptInsample.exp2$averagetypeI[5]-average.errors.OptInsample.exp2$averagetypeI[12],
                  average.errors.OptInsample.exp2$averagetypeI[13],
                  1-average.errors.OptInsample.exp2$averagetypeI[6]-average.errors.OptInsample.exp2$averagetypeI[13])

outputs_2$ACRFshift<-c(average.errors.OptInsample.shift.exp2$averagetypeI[8],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[1]-average.errors.OptInsample.shift.exp2$averagetypeI[8],
                       average.errors.OptInsample.shift.exp2$averagetypeI[9],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[2]-average.errors.OptInsample.shift.exp2$averagetypeI[9],
                       average.errors.OptInsample.shift.exp2$averagetypeI[10],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[3]-average.errors.OptInsample.shift.exp2$averagetypeI[10],
                       average.errors.OptInsample.shift.exp2$averagetypeI[11],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[4]-average.errors.OptInsample.shift.exp2$averagetypeI[11],
                       average.errors.OptInsample.shift.exp2$averagetypeI[12],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[5]-average.errors.OptInsample.shift.exp2$averagetypeI[12],
                       average.errors.OptInsample.shift.exp2$averagetypeI[13],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[6]-average.errors.OptInsample.shift.exp2$averagetypeI[13])

outputs_2 <-reshape2::melt(outputs_2 ,id=c('Type','number'))
colnames(outputs_2)<-c('Type','number','Method','Coverage')
outputs_2$Coverage<-100*outputs_2$Coverage
outputs_2$x<-ifelse(outputs_2$number==0,1,
                    ifelse(outputs_2$number==1,5,
                           ifelse(outputs_2$number==2,9,
                                  ifelse(outputs_2$number==3,13,
                                         ifelse(outputs_2$number==4,17,21)))))
df1.exp2<-outputs_2[outputs_2$Method=='CSForest',]
df2.exp2<-outputs_2[outputs_2$Method=='BCOPS',]
df3.exp2<-outputs_2[outputs_2$Method=='DC',]
df4.exp2<-outputs_2[outputs_2$Method=='CRF',]
df5.exp2<-outputs_2[outputs_2$Method=='ACRF',]
df6.exp2<-outputs_2[outputs_2$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1.exp2,
           aes(x=x,y=Coverage,fill=Type),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2.exp2,aes(x=x+0.5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3.exp2,aes(x=x+0.5*2,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4.exp2,aes(x=x+0.5*3,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5.exp2,aes(x=x+0.5*4,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6.exp2,aes(x=x+0.5*5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed")+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   3                   4                    5 ')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58
CSForest.len<-sapply(average.errors.CSForest.loop.exp2$averagetypeII,"[[",4)
BCOPS.len<-sapply(average.errors.BCOPS.loop.exp2$averagetypeII,"[[",4)
Density.len<-sapply(average.errors.Density.loop.exp2$averagetypeII,"[[",4)
Insample.len<-sapply(average.errors.Insample.loop.exp2$averagetypeII,"[[",4)
OptInsample.len<-sapply(average.errors.OptInsample.loop.exp2$averagetypeII,"[[",4)
OptInsample.shift.len<-sapply(average.errors.OptInsample.shift.loop.exp2$averagetypeII,"[[",4)

# 创建一个包含多个变量的数据框
# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(csforest_average_length)), levels = method_order))
data$Average_Length <- c(CSForest.len, BCOPS.len, Density.len, Insample.len, OptInsample.len, OptInsample.shift.len)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()

# Experiment 3
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp3_output_FashionMnist.txt"
sink(file_path)

models.size.loop<-function(n,n_train_lists,n_test_lists){
  #I means both II means train III means test
  num=length(n_train_lists)
  
  dat05<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat05)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat05['Train Size']=sort(rep(n_train_lists,n))
  dat05['Test Size']=sort(rep(n_test_lists,n))
  
  dat69<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat69)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat69['Train Size']=sort(rep(n_train_lists,n))
  dat69['Test Size']=sort(rep(n_test_lists,n))
  
  datI<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(datI)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  datI['Train Size']=sort(rep(n_train_lists,n))
  datI['Test Size']=sort(rep(n_test_lists,n))
  
  print('Training CSForest...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['CSForest'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CSForest'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CSForest'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  print('Training BCOPS...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=BCOPS.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['BCOPS'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['BCOPS'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['BCOPS'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  print('Training Density...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Density.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['DC'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['DC'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['DC'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  
  print('Training CRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Insample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['CRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  print('Training ACRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['ACRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  print('Training ACRFshift...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.shift.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full)
    dat05['ACRFshift'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRFshift'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRFshift'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",7)
  }
  
  return(list(datI=datI,dat05=dat05, dat69 = dat69))
}


# MNIST
res.exp3.row1=models.size.loop(n=10,n_train_lists=c(50,80,100,150,200),n_test_lists=c(50,80,100,150,200))
res.exp3.row2=models.size.loop(n=10,n_train_lists=c(50,80,100,150,200),n_test_lists=c(200,200,200,200,200)) #fix test
res.exp3.row3=models.size.loop(n=10,n_train_lists=c(200,200,200,200,200),n_test_lists=c(50,80,100,150,200)) #fix train
print(res.exp3.row1)
print(res.exp3.row2)
print(res.exp3.row3)
sink()


#Plot

dat05.exp3.row1<-reshape2::melt(res.exp3.row1$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row1$Method<-as.factor(dat05.exp3.row1$Method)
dat69.exp3.row1<-reshape2::melt(res.exp3.row1$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row1$Method<-as.factor(dat69.exp3.row1$Method)
datI.exp3.row1<-reshape2::melt(res.exp3.row1$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row1$Method<-as.factor(datI.exp3.row1$Method)
#View(dat05.3)
#View(dat69.3)
#dat.3<-rbind(dat05.exp3.row1,dat69.exp3.row1,datI.exp3.row1)

##Row 1
datI.exp3.row1$Size=datI.exp3.row1$`Train Size`
p1.1<-ggplot(datI.exp3.row1,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class size (both)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A1')

dat05.exp3.row1$Size=dat05.exp3.row1$`Train Size`
p1.2<-ggplot(dat05.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A2')

dat69.exp3.row1$Size=dat69.exp3.row1$`Train Size`
p1.3<-ggplot(dat69.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A3')

##Row 2

dat05.exp3.row2<-reshape2::melt(res.exp3.row2$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row2$Method<-as.factor(dat05.exp3.row2$Method)
dat69.exp3.row2<-reshape2::melt(res.exp3.row2$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row2$Method<-as.factor(dat69.exp3.row2$Method)
datI.exp3.row2<-reshape2::melt(res.exp3.row2$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row2$Method<-as.factor(datI.exp3.row2$Method)

datI.exp3.row2$Size=datI.exp3.row2$`Train Size`
p2.1<-ggplot(datI.exp3.row2,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B1')

dat05.exp3.row2$Size=dat05.exp3.row2$`Train Size`
p2.2<-ggplot(dat05.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B2')

dat69.exp3.row2$Size=dat69.exp3.row2$`Train Size`
p2.3<-ggplot(dat69.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B3')



##Row 3

dat05.exp3.row3<-reshape2::melt(res.exp3.row3$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row3$Method<-as.factor(dat05.exp3.row3$Method)
dat69.exp3.row3<-reshape2::melt(res.exp3.row3$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row3$Method<-as.factor(dat69.exp3.row3$Method)
datI.exp3.row3<-reshape2::melt(res.exp3.row3$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row3$Method<-as.factor(datI.exp3.row3$Method)

datI.exp3.row3$Size=datI.exp3.row3$`Test Size`
p3.1<-ggplot(datI.exp3.row3,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C1')

dat05.exp3.row3$Size=dat05.exp3.row3$`Test Size`
p3.2<-ggplot(dat05.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C2')

dat69.exp3.row3$Size=dat69.exp3.row3$`Test Size`
p3.3<-ggplot(dat69.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  #scale_color_aaas()+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C3')

#range plots
library(ggpubr)
ggarrange(
          p1.2,p2.2,p3.2,
          p1.3,p2.3,p3.3,
          # labels = c("A1", "B1",'C1',
          #            "A2", "B2","C2"),
          ncol = 3, nrow = 2,common.legend = T,legend = "bottom",font.label = list(size = 8, color = "black"))


# Experiment 4

n_train=200
n_test=n_test_outlier=50
n=10
alpha=0.05
w=0
average.errors.CSForest.loop.a05w0=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w0=data_mean(average.errors.CSForest.loop.a05w0)
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",2))


w="LOG"
average.errors.CSForest.loop.a05wl=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05wl=data_mean(average.errors.CSForest.loop.a05wl)
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",2))



w=1
average.errors.CSForest.loop.a05w1=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w1=data_mean(average.errors.CSForest.loop.a05w1)
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",2))


w=1.5
average.errors.CSForest.loop.a05w15=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w15=data_mean(average.errors.CSForest.loop.a05w15)
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",2))

w=2
average.errors.CSForest.loop.a05w2=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w25=data_mean(average.errors.CSForest.loop.a05w2)
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",2))



w=5
average.errors.CSForest.loop.a05w5=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w5=data_mean(average.errors.CSForest.loop.a05w5)
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",2))



w=10
average.errors.CSForest.loop.a05w10=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w10=data_mean(average.errors.CSForest.loop.a05w10)
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",2))


w=100
average.errors.CSForest.loop.a05w100=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,K=5,train.label=train.label,test.label=test.label,trainData.full=trainData.full,testData.full=testData.full,alpha=alpha,weight=w)
average.errors.CSForest.a05w100=data_mean(average.errors.CSForest.loop.a05w100)
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",2))

# 3.Protein Data
library(readxl)
df <- read_excel("/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/exp/mnistFull/Data_Cortex_Nuclear.xls")
View(df)
df <- df[, -1]
# 假设你的数据集名为df
str(df);dim(df) #1080   82
# 删除缺失值超过95%的列
df <- df[, colMeans(is.na(df)) <= 0.90]
df[is.na(df)] <- 0# 以0替换缺失值
dim(df)
# 将类别变量转换为数值变量
library(dplyr)
# 或者使用基础R的赋值操作符$
df$Genotype <- as.numeric(factor(df$Genotype, levels = unique(df$Genotype), labels = 0:(length(unique(df$Genotype)) - 1)))
df$Treatment <- as.numeric(factor(df$Treatment, levels = unique(df$Treatment), labels = 0:(length(unique(df$Treatment)) - 1)))
df$Behavior <- as.numeric(factor(df$Behavior, levels = unique(df$Behavior), labels = 0:(length(unique(df$Behavior)) - 1)))
df$class <- as.numeric(factor(df$class, levels = unique(df$class), labels = 0:(length(unique(df$class)) - 1)))-1
table(df$class)
# 0   1   2   3   4   5   6   7 
# 150 150 135 135 135 135 105 135 

###EXP 1
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp1_output_protein.txt"
sink(file_path)
print('****************CSForest****************')
average.errors.CSForest.loop=CSForest.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.CSForest.loop$averagetypeI)
print(average.errors.CSForest.loop$averagetypeII)
print('****************BCOPS****************')
average.errors.BCOPS.loop=BCOPS.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.BCOPS.loop$averagetypeI)
print(average.errors.BCOPS.loop$averagetypeII)
print('****************DC****************')
average.errors.Density.loop=Density.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.Density.loop$averagetypeI)
print(average.errors.Density.loop$averagetypeII)
print('****************CRF****************')
average.errors.Insample.loop=Insample.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.Insample.loop$averagetypeI)
print(average.errors.Insample.loop$averagetypeII)
print('****************ACRF****************')
average.errors.OptInsample.loop=OptInsample.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.OptInsample.loop$averagetypeI)
print(average.errors.OptInsample.loop$averagetypeII)
print('****************ACRFShift****************')
average.errors.OptInsample.shift.loop=OptInsample.shift.loop(n=10,n_train=50,n_test=50,n_test_outlier=50,mydata='protein',networkD=df,K=3)
print(average.errors.OptInsample.shift.loop$averagetypeI)
print(average.errors.OptInsample.shift.loop$averagetypeII)
print(average.errors.OptInsample.shift.loop$alphas)
# 关闭文件
sink()


data_mean<-function(data){
  datamean_1=0
  for (i in 1:length(data$averagetypeI)) {
    datamean_1=datamean_1+data$averagetypeI[[i]]
  }
  
  datamean_2=0
  for (i in 1:length(data$averagetypeII)) {
    datamean_2=datamean_2+data$averagetypeII[[i]]
  }
  
  
  cat('Type I std', sd(sapply(data$averagetypeI,"[[",5)),'Type II std', sd(sapply(data$averagetypeII,"[[",3)),
      'Length std',sd(sapply(data$averagetypeII,"[[",4)))
  return(list(averagetypeI =datamean_1/length(data$averagetypeII), averagetypeII = datamean_2/length(data$averagetypeII)))
  
}

average.errors.CSForest=data_mean(average.errors.CSForest.loop)

average.errors.BCOPS=data_mean(average.errors.BCOPS.loop)

average.errors.Density=data_mean(average.errors.Density.loop)

average.errors.Insample=data_mean(average.errors.Insample.loop)

average.errors.OptInsample=data_mean(average.errors.OptInsample.loop)

average.errors.OptInsample.shift=data_mean(average.errors.OptInsample.shift.loop)

# 画图 per_class
outputs_1<-matrix(0, ncol = 8, nrow = 10)
outputs_1<-as.data.frame(outputs_1)

colnames(outputs_1)<-c("prediction","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_1$prediction<-rep(c('C(x)={y}','multilabel'),5)
outputs_1$number<-c(0,0,1,1,2,2,3,3,'R','R')
outputs_1$CSForest<-c(average.errors.CSForest$averagetypeI[6],
                      1-average.errors.CSForest$averagetypeI[1]-average.errors.CSForest$averagetypeI[6],
                      average.errors.CSForest$averagetypeI[7],
                      1-average.errors.CSForest$averagetypeI[2]-average.errors.CSForest$averagetypeI[7],
                      average.errors.CSForest$averagetypeI[8],
                      1-average.errors.CSForest$averagetypeI[3]-average.errors.CSForest$averagetypeI[8],
                      average.errors.CSForest$averagetypeI[9],
                      1-average.errors.CSForest$averagetypeI[4]-average.errors.CSForest$averagetypeI[9],
                      1-average.errors.CSForest$averagetypeII[2],
                      0)

outputs_1$BCOPS<-c(average.errors.BCOPS$averagetypeI[6],
                   1-average.errors.BCOPS$averagetypeI[1]-average.errors.BCOPS$averagetypeI[6],
                   average.errors.BCOPS$averagetypeI[7],
                   1-average.errors.BCOPS$averagetypeI[2]-average.errors.BCOPS$averagetypeI[7],
                   average.errors.BCOPS$averagetypeI[8],
                   1-average.errors.BCOPS$averagetypeI[3]-average.errors.BCOPS$averagetypeI[8],
                   average.errors.BCOPS$averagetypeI[9],
                   1-average.errors.BCOPS$averagetypeI[4]-average.errors.BCOPS$averagetypeI[9],
                   1-average.errors.BCOPS$averagetypeII[2],
                   0)

outputs_1$DC<-c(average.errors.Density$averagetypeI[6],
                1-average.errors.Density$averagetypeI[1]-average.errors.Density$averagetypeI[6],
                average.errors.Density$averagetypeI[7],
                1-average.errors.Density$averagetypeI[2]-average.errors.Density$averagetypeI[7],
                average.errors.Density$averagetypeI[8],
                1-average.errors.Density$averagetypeI[3]-average.errors.Density$averagetypeI[8],
                average.errors.Density$averagetypeI[9],
                1-average.errors.Density$averagetypeI[4]-average.errors.Density$averagetypeI[9],
                1-average.errors.Density$averagetypeII[2],
                0)
outputs_1$CRF<-c(average.errors.Insample$averagetypeI[6],
                 1-average.errors.Insample$averagetypeI[1]-average.errors.Insample$averagetypeI[6],
                 average.errors.Insample$averagetypeI[7],
                 1-average.errors.Insample$averagetypeI[2]-average.errors.Insample$averagetypeI[7],
                 average.errors.Insample$averagetypeI[8],
                 1-average.errors.Insample$averagetypeI[3]-average.errors.Insample$averagetypeI[8],
                 average.errors.Insample$averagetypeI[9],
                 1-average.errors.Insample$averagetypeI[4]-average.errors.Insample$averagetypeI[9],
                 1-average.errors.Insample$averagetypeII[2],
                 0)


outputs_1$ACRF<-c(average.errors.OptInsample$averagetypeI[6],
                  1-average.errors.OptInsample$averagetypeI[1]-average.errors.OptInsample$averagetypeI[6],
                  average.errors.OptInsample$averagetypeI[7],
                  1-average.errors.OptInsample$averagetypeI[2]-average.errors.OptInsample$averagetypeI[7],
                  average.errors.OptInsample$averagetypeI[8],
                  1-average.errors.OptInsample$averagetypeI[3]-average.errors.OptInsample$averagetypeI[8],
                  average.errors.OptInsample$averagetypeI[9],
                  1-average.errors.OptInsample$averagetypeI[4]-average.errors.OptInsample$averagetypeI[9],
                  1-average.errors.OptInsample$averagetypeII[2],
                  0)
outputs_1$ACRFshift<-c(average.errors.OptInsample.shift$averagetypeI[6],
                       1-average.errors.OptInsample.shift$averagetypeI[1]-average.errors.OptInsample.shift$averagetypeI[6],
                       average.errors.OptInsample.shift$averagetypeI[7],
                       1-average.errors.OptInsample.shift$averagetypeI[2]-average.errors.OptInsample.shift$averagetypeI[7],
                       average.errors.OptInsample.shift$averagetypeI[8],
                       1-average.errors.OptInsample.shift$averagetypeI[3]-average.errors.OptInsample.shift$averagetypeI[8],
                       average.errors.OptInsample.shift$averagetypeI[9],
                       1-average.errors.OptInsample.shift$averagetypeI[4]-average.errors.OptInsample.shift$averagetypeI[9],
                       1-average.errors.OptInsample.shift$averagetypeII[2],
                       0)

outputs_1 <-reshape2::melt(outputs_1 ,id=c('prediction','number'))
colnames(outputs_1)<-c('prediction','number','Method','Coverage')
outputs_1$Coverage<-100*outputs_1$Coverage
outputs_1$x<-ifelse(outputs_1$number==0,1,
                    ifelse(outputs_1$number==1,5,
                           ifelse(outputs_1$number==2,9,
                                  ifelse(outputs_1$number==3,13,17))))
df1<-outputs_1[outputs_1$Method=='CSForest',]
df2<-outputs_1[outputs_1$Method=='BCOPS',]
df3<-outputs_1[outputs_1$Method=='DC',]
df4<-outputs_1[outputs_1$Method=='CRF',]
df5<-outputs_1[outputs_1$Method=='ACRF',]
df6<-outputs_1[outputs_1$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1,
           aes(x=x,y=Coverage,fill=prediction),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2,aes(x=x+0.5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3,aes(x=x+0.5*2,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4,aes(x=x+0.5*3,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5,aes(x=x+0.5*4,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6,aes(x=x+0.5*5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5,
                                25,25.5,26,26.5,27,27.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed",size=1)+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   3                   R')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58

# 提取各列的 average_length 数据
csforest_average_length <- sapply(average.errors.CSForest.loop$averagetypeII,"[[",4)
bcops_average_length <- sapply(average.errors.BCOPS.loop$averagetypeII,"[[",4)
dc_average_length <- sapply(average.errors.Density.loop$averagetypeII,"[[",4)
crf_average_length <- sapply(average.errors.Insample.loop$averagetypeII,"[[",4)
acrf_average_length <- sapply(average.errors.OptInsample.loop$averagetypeII,"[[",4)
acrfshift_average_length <- sapply(average.errors.OptInsample.shift.loop$averagetypeII,"[[",4)




# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(csforest_average_length)), levels = method_order))
data$Average_Length <- c(csforest_average_length, bcops_average_length, dc_average_length, crf_average_length, acrf_average_length, acrfshift_average_length)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()




#Experiment 2
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp2_output_protein.txt"
sink(file_path)
average.errors.CSForest.loop.exp2=CSForest.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.CSForest.loop.exp2$averagetypeI)
print(average.errors.CSForest.loop.exp2$averagetypeII)
average.errors.BCOPS.loop.exp2=BCOPS.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.BCOPS.loop.exp2$averagetypeI)
print(average.errors.BCOPS.loop.exp2$averagetypeII)
average.errors.Density.loop.exp2=Density.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.Density.loop.exp2$averagetypeI)
print(average.errors.Density.loop.exp2$averagetypeII)
average.errors.Insample.loop.exp2=Insample.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.Insample.loop.exp2$averagetypeI)
print(average.errors.Insample.loop.exp2$averagetypeII)
average.errors.OptInsample.loop.exp2=OptInsample.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.OptInsample.loop.exp2$averagetypeI)
print(average.errors.OptInsample.loop.exp2$averagetypeII)
average.errors.OptInsample.shift.loop.exp2=OptInsample.shift.loop(n=10,n_train=50,n_test=30,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=3)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeI)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeII)
print(average.errors.OptInsample.shift.loop.exp2$alphas)
# 关闭文件
sink()

average.errors.CSForest.exp2=data_mean(average.errors.CSForest.loop.exp2)
average.errors.BCOPS.exp2=data_mean(average.errors.BCOPS.loop.exp2)
average.errors.Density.exp2=data_mean(average.errors.Density.loop.exp2)
average.errors.Insample.exp2=data_mean(average.errors.Insample.loop.exp2)
average.errors.OptInsample.exp2=data_mean(average.errors.OptInsample.loop.exp2)
average.errors.OptInsample.shift.exp2=data_mean(average.errors.OptInsample.shift.loop.exp2)

outputs_2<-matrix(0, ncol = 8, nrow = 8)
outputs_2<-as.data.frame(outputs_2)

colnames(outputs_2)<-c("Type","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_2$Type<-rep(c('Correct Only ','Multi-Label'),4)
outputs_2$number<-c(0,0,1,1,2,2,3,3)
outputs_2$CSForest<-c(average.errors.CSForest.exp2$averagetypeI[6],
                      1-average.errors.CSForest.exp2$averagetypeI[1]-average.errors.CSForest.exp2$averagetypeI[6],
                      average.errors.CSForest.exp2$averagetypeI[7],
                      1-average.errors.CSForest.exp2$averagetypeI[2]-average.errors.CSForest.exp2$averagetypeI[7],
                      average.errors.CSForest.exp2$averagetypeI[8],
                      1-average.errors.CSForest.exp2$averagetypeI[3]-average.errors.CSForest.exp2$averagetypeI[8],
                      average.errors.CSForest.exp2$averagetypeI[9],
                      1-average.errors.CSForest.exp2$averagetypeI[4]-average.errors.CSForest.exp2$averagetypeI[9])

outputs_2$BCOPS<-c(average.errors.BCOPS.exp2$averagetypeI[6],
                   1-average.errors.BCOPS.exp2$averagetypeI[1]-average.errors.BCOPS.exp2$averagetypeI[6],
                   average.errors.BCOPS.exp2$averagetypeI[7],
                   1-average.errors.BCOPS.exp2$averagetypeI[2]-average.errors.BCOPS.exp2$averagetypeI[7],
                   average.errors.BCOPS.exp2$averagetypeI[8],
                   1-average.errors.BCOPS.exp2$averagetypeI[3]-average.errors.BCOPS.exp2$averagetypeI[8],
                   average.errors.BCOPS.exp2$averagetypeI[9],
                   1-average.errors.BCOPS.exp2$averagetypeI[4]-average.errors.BCOPS.exp2$averagetypeI[9])

outputs_2$DC<-c(average.errors.Density.exp2$averagetypeI[6],
                1-average.errors.Density.exp2$averagetypeI[1]-average.errors.Density.exp2$averagetypeI[6],
                average.errors.Density.exp2$averagetypeI[7],
                1-average.errors.Density.exp2$averagetypeI[2]-average.errors.Density.exp2$averagetypeI[7],
                average.errors.Density.exp2$averagetypeI[8],
                1-average.errors.Density.exp2$averagetypeI[3]-average.errors.Density.exp2$averagetypeI[8],
                average.errors.Density.exp2$averagetypeI[9],
                1-average.errors.Density.exp2$averagetypeI[4]-average.errors.Density.exp2$averagetypeI[9])

outputs_2$CRF<-c(average.errors.Insample.exp2$averagetypeI[6],
                 1-average.errors.Insample.exp2$averagetypeI[1]-average.errors.Insample.exp2$averagetypeI[6],
                 average.errors.Insample.exp2$averagetypeI[7],
                 1-average.errors.Insample.exp2$averagetypeI[2]-average.errors.Insample.exp2$averagetypeI[7],
                 average.errors.Insample.exp2$averagetypeI[8],
                 1-average.errors.Insample.exp2$averagetypeI[3]-average.errors.Insample.exp2$averagetypeI[8],
                 average.errors.Insample.exp2$averagetypeI[9],
                 1-average.errors.Insample.exp2$averagetypeI[4]-average.errors.Insample.exp2$averagetypeI[9])


outputs_2$ACRF<-c(average.errors.OptInsample.exp2$averagetypeI[6],
                  1-average.errors.OptInsample.exp2$averagetypeI[1]-average.errors.OptInsample.exp2$averagetypeI[6],
                  average.errors.OptInsample.exp2$averagetypeI[7],
                  1-average.errors.OptInsample.exp2$averagetypeI[2]-average.errors.OptInsample.exp2$averagetypeI[7],
                  average.errors.OptInsample.exp2$averagetypeI[8],
                  1-average.errors.OptInsample.exp2$averagetypeI[3]-average.errors.OptInsample.exp2$averagetypeI[8],
                  average.errors.OptInsample.exp2$averagetypeI[9],
                  1-average.errors.OptInsample.exp2$averagetypeI[4]-average.errors.OptInsample.exp2$averagetypeI[9])

outputs_2$ACRFshift<-c(average.errors.OptInsample.shift.exp2$averagetypeI[6],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[1]-average.errors.OptInsample.shift.exp2$averagetypeI[6],
                       average.errors.OptInsample.shift.exp2$averagetypeI[7],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[2]-average.errors.OptInsample.shift.exp2$averagetypeI[7],
                       average.errors.OptInsample.shift.exp2$averagetypeI[8],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[3]-average.errors.OptInsample.shift.exp2$averagetypeI[8],
                       average.errors.OptInsample.shift.exp2$averagetypeI[9],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[4]-average.errors.OptInsample.shift.exp2$averagetypeI[9])

outputs_2 <-reshape2::melt(outputs_2 ,id=c('Type','number'))
colnames(outputs_2)<-c('Type','number','Method','Coverage')
outputs_2$Coverage<-100*outputs_2$Coverage
outputs_2$x<-ifelse(outputs_2$number==0,1,
                    ifelse(outputs_2$number==1,5,
                           ifelse(outputs_2$number==2,9,13)))
df1.exp2<-outputs_2[outputs_2$Method=='CSForest',]
df2.exp2<-outputs_2[outputs_2$Method=='BCOPS',]
df3.exp2<-outputs_2[outputs_2$Method=='DC',]
df4.exp2<-outputs_2[outputs_2$Method=='CRF',]
df5.exp2<-outputs_2[outputs_2$Method=='ACRF',]
df6.exp2<-outputs_2[outputs_2$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1.exp2,
           aes(x=x,y=Coverage,fill=Type),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2.exp2,aes(x=x+0.5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3.exp2,aes(x=x+0.5*2,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4.exp2,aes(x=x+0.5*3,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5.exp2,aes(x=x+0.5*4,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6.exp2,aes(x=x+0.5*5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed")+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   3')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58
CSForest.len<-sapply(average.errors.CSForest.loop.exp2$averagetypeII,"[[",4)
BCOPS.len<-sapply(average.errors.BCOPS.loop.exp2$averagetypeII,"[[",4)
Density.len<-sapply(average.errors.Density.loop.exp2$averagetypeII,"[[",4)
Insample.len<-sapply(average.errors.Insample.loop.exp2$averagetypeII,"[[",4)
OptInsample.len<-sapply(average.errors.OptInsample.loop.exp2$averagetypeII,"[[",4)
OptInsample.shift.len<-sapply(average.errors.OptInsample.shift.loop.exp2$averagetypeII,"[[",4)

# 创建一个包含多个变量的数据框
# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(CSForest.len)), levels = method_order))
data$Average_Length <- c(CSForest.len, BCOPS.len, Density.len, Insample.len, OptInsample.len, OptInsample.shift.len)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()


# Experiment 3
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp3_output_protein.txt"
sink(file_path)

models.size.loop<-function(n,n_train_lists,n_test_lists){
  #I means both II means train III means test
  num=length(n_train_lists)
  
  dat05<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat05)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat05['Train Size']=sort(rep(n_train_lists,n))
  dat05['Test Size']=sort(rep(n_test_lists,n))
  
  dat69<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat69)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat69['Train Size']=sort(rep(n_train_lists,n))
  dat69['Test Size']=sort(rep(n_test_lists,n))
  
  datI<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(datI)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  datI['Train Size']=sort(rep(n_train_lists,n))
  datI['Test Size']=sort(rep(n_test_lists,n))
  
  print('Training CSForest...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['CSForest'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CSForest'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CSForest'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  print('Training BCOPS...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=BCOPS.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['BCOPS'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['BCOPS'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['BCOPS'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  print('Training Density...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Density.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['DC'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['DC'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['DC'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  
  print('Training CRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Insample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['CRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  print('Training ACRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['ACRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  print('Training ACRFshift...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.shift.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=3)
    dat05['ACRFshift'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRFshift'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRFshift'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",5)
  }
  
  return(list(datI=datI,dat05=dat05, dat69 = dat69))
}


# MNIST
res.exp3.row1=models.size.loop(n=10,n_train_lists=c(30,40,50),n_test_lists=c(30,40,50))
res.exp3.row2=models.size.loop(n=10,n_train_lists=c(30,40,50),n_test_lists=c(50,50,50)) #fix test
res.exp3.row3=models.size.loop(n=10,n_train_lists=c(50,50,50),n_test_lists=c(30,40,50)) #fix train
print(res.exp3.row1)
print(res.exp3.row2)
print(res.exp3.row3)
sink()


#Plot

dat05.exp3.row1<-reshape2::melt(res.exp3.row1$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row1$Method<-as.factor(dat05.exp3.row1$Method)
dat69.exp3.row1<-reshape2::melt(res.exp3.row1$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row1$Method<-as.factor(dat69.exp3.row1$Method)
datI.exp3.row1<-reshape2::melt(res.exp3.row1$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row1$Method<-as.factor(datI.exp3.row1$Method)
#View(dat05.3)
#View(dat69.3)
#dat.3<-rbind(dat05.exp3.row1,dat69.exp3.row1,datI.exp3.row1)

##Row 1
datI.exp3.row1$Size=datI.exp3.row1$`Train Size`
p1.1<-ggplot(datI.exp3.row1,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class size (both)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A1')

dat05.exp3.row1$Size=dat05.exp3.row1$`Train Size`
p1.2<-ggplot(dat05.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A2')

dat69.exp3.row1$Size=dat69.exp3.row1$`Train Size`
p1.3<-ggplot(dat69.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A3')

##Row 2

dat05.exp3.row2<-reshape2::melt(res.exp3.row2$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row2$Method<-as.factor(dat05.exp3.row2$Method)
dat69.exp3.row2<-reshape2::melt(res.exp3.row2$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row2$Method<-as.factor(dat69.exp3.row2$Method)
datI.exp3.row2<-reshape2::melt(res.exp3.row2$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row2$Method<-as.factor(datI.exp3.row2$Method)

datI.exp3.row2$Size=datI.exp3.row2$`Train Size`
p2.1<-ggplot(datI.exp3.row2,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B1')

dat05.exp3.row2$Size=dat05.exp3.row2$`Train Size`
p2.2<-ggplot(dat05.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B2')

dat69.exp3.row2$Size=dat69.exp3.row2$`Train Size`
p2.3<-ggplot(dat69.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B3')



##Row 3

dat05.exp3.row3<-reshape2::melt(res.exp3.row3$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row3$Method<-as.factor(dat05.exp3.row3$Method)
dat69.exp3.row3<-reshape2::melt(res.exp3.row3$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row3$Method<-as.factor(dat69.exp3.row3$Method)
datI.exp3.row3<-reshape2::melt(res.exp3.row3$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row3$Method<-as.factor(datI.exp3.row3$Method)

datI.exp3.row3$Size=datI.exp3.row3$`Test Size`
p3.1<-ggplot(datI.exp3.row3,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C1')

dat05.exp3.row3$Size=dat05.exp3.row3$`Test Size`
p3.2<-ggplot(dat05.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C2')

dat69.exp3.row3$Size=dat69.exp3.row3$`Test Size`
p3.3<-ggplot(dat69.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  #scale_color_aaas()+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C3')

#range plots
library(ggpubr)
ggarrange(
  # p1.2,p2.2,p3.2,
  p1.3,p2.3,p3.3,
  # labels = c("A1", "B1",'C1',
  #            "A2", "B2","C2"),
  ncol = 3, nrow = 1,common.legend = T,legend = "bottom",font.label = list(size = 8, color = "black"))

# Exp 4

n_train=50
n_test=n_test_outlier=50
n=10
alpha=0.05

w=0
average.errors.CSForest.loop.a05w0=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w0=data_mean(average.errors.CSForest.loop.a05w0)
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",2))


w="LOG"
average.errors.CSForest.loop.a05wl=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05wl=data_mean(average.errors.CSForest.loop.a05wl)
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",2))



w=0.15
average.errors.CSForest.loop.a05w1=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w1=data_mean(average.errors.CSForest.loop.a05w1)
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",2))


w=0.375
average.errors.CSForest.loop.a05w15=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w15=data_mean(average.errors.CSForest.loop.a05w15)
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",2))

w=0.5
average.errors.CSForest.loop.a05w2=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w25=data_mean(average.errors.CSForest.loop.a05w2)
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",2))



w=5
average.errors.CSForest.loop.a05w5=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w5=data_mean(average.errors.CSForest.loop.a05w5)
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",2))



w=10
average.errors.CSForest.loop.a05w10=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w10=data_mean(average.errors.CSForest.loop.a05w10)
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",2))


w=100
average.errors.CSForest.loop.a05w100=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=3,alpha=alpha,weight=w)
average.errors.CSForest.a05w100=data_mean(average.errors.CSForest.loop.a05w100)
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",2))



########################################################################NetworkD########################################################################
# Load the RData file
df <- read.table("mnistFull/Network/kddcup.data_10_percent", sep = ",")
# Show the first 10 rows of the loaded data
df$V2 <- as.numeric(as.factor(df$V2))
df$V3 <- as.numeric(as.factor(df$V3))
df$V4 <- as.numeric(as.factor(df$V4))
# Assuming 'df' is your data frame
colnames(df)[ncol(df)] <- "result"
# Assuming 'df' is your data frame
df[, -ncol(df)] <- lapply(df[, -ncol(df)], as.numeric)
head(df, 10)
dim(df)
length(table(df$result))
###EXP 1
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp1_output_Network.txt"
sink(file_path)
print('****************CSForest****************')
average.errors.CSForest.loop=CSForest.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.CSForest.loop$averagetypeI)
print(average.errors.CSForest.loop$averagetypeII)
print('****************BCOPS****************')
average.errors.BCOPS.loop=BCOPS.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.BCOPS.loop$averagetypeI)
print(average.errors.BCOPS.loop$averagetypeII)
print('****************DC****************')
average.errors.Density.loop=Density.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.Density.loop$averagetypeI)
print(average.errors.Density.loop$averagetypeII)
print('****************CRF****************')
average.errors.Insample.loop=Insample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.Insample.loop$averagetypeI)
print(average.errors.Insample.loop$averagetypeII)
print('****************ACRF****************')
average.errors.OptInsample.loop=OptInsample.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.OptInsample.loop$averagetypeI)
print(average.errors.OptInsample.loop$averagetypeII)
print('****************ACRFShift****************')
average.errors.OptInsample.shift.loop=OptInsample.shift.loop(n=10,n_train=500,n_test=500,n_test_outlier=500,mydata='Network',networkD=df,K=2)
print(average.errors.OptInsample.shift.loop$averagetypeI)
print(average.errors.OptInsample.shift.loop$averagetypeII)
print(average.errors.OptInsample.shift.loop$alphas)
# 关闭文件
sink()


data_mean<-function(data){
  datamean_1=0
  for (i in 1:length(data$averagetypeI)) {
    datamean_1=datamean_1+data$averagetypeI[[i]]
  }
  
  datamean_2=0
  for (i in 1:length(data$averagetypeII)) {
    datamean_2=datamean_2+data$averagetypeII[[i]]
  }
  
  
  cat('Type I std', sd(sapply(data$averagetypeI,"[[",4)),'Type II std', sd(sapply(data$averagetypeII,"[[",3)),
      'Length std',sd(sapply(data$averagetypeII,"[[",4)))
  return(list(averagetypeI =datamean_1/length(data$averagetypeII), averagetypeII = datamean_2/length(data$averagetypeII)))
  
}

average.errors.CSForest=data_mean(average.errors.CSForest.loop)

average.errors.BCOPS=data_mean(average.errors.BCOPS.loop)

average.errors.Density=data_mean(average.errors.Density.loop)

average.errors.Insample=data_mean(average.errors.Insample.loop)

average.errors.OptInsample=data_mean(average.errors.OptInsample.loop)

average.errors.OptInsample.shift=data_mean(average.errors.OptInsample.shift.loop)

# 画图 per_class
outputs_1<-matrix(0, ncol = 8, nrow = 8)
outputs_1<-as.data.frame(outputs_1)

colnames(outputs_1)<-c("prediction","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_1$prediction<-rep(c('C(x)={y}','multilabel'),4)
outputs_1$number<-c(0,0,1,1,2,2,'R','R')
outputs_1$CSForest<-c(average.errors.CSForest$averagetypeI[5],
                      1-average.errors.CSForest$averagetypeI[1]-average.errors.CSForest$averagetypeI[5],
                      average.errors.CSForest$averagetypeI[6],
                      1-average.errors.CSForest$averagetypeI[2]-average.errors.CSForest$averagetypeI[6],
                      average.errors.CSForest$averagetypeI[7],
                      1-average.errors.CSForest$averagetypeI[3]-average.errors.CSForest$averagetypeI[7],
                      1-average.errors.CSForest$averagetypeII[2],
                      0)

outputs_1$BCOPS<-c(average.errors.BCOPS$averagetypeI[5],
                   1-average.errors.BCOPS$averagetypeI[1]-average.errors.BCOPS$averagetypeI[5],
                   average.errors.BCOPS$averagetypeI[6],
                   1-average.errors.BCOPS$averagetypeI[2]-average.errors.BCOPS$averagetypeI[6],
                   average.errors.BCOPS$averagetypeI[7],
                   1-average.errors.BCOPS$averagetypeI[3]-average.errors.BCOPS$averagetypeI[7],
                   1-average.errors.BCOPS$averagetypeII[2],
                   0)

outputs_1$DC<-c(average.errors.Density$averagetypeI[5],
                1-average.errors.Density$averagetypeI[1]-average.errors.Density$averagetypeI[5],
                average.errors.Density$averagetypeI[6],
                1-average.errors.Density$averagetypeI[2]-average.errors.Density$averagetypeI[6],
                average.errors.Density$averagetypeI[7],
                1-average.errors.Density$averagetypeI[3]-average.errors.Density$averagetypeI[7],
                1-average.errors.Density$averagetypeII[2],
                0)
outputs_1$CRF<-c(average.errors.Insample$averagetypeI[5],
                 1-average.errors.Insample$averagetypeI[1]-average.errors.Insample$averagetypeI[5],
                 average.errors.Insample$averagetypeI[6],
                 1-average.errors.Insample$averagetypeI[2]-average.errors.Insample$averagetypeI[6],
                 average.errors.Insample$averagetypeI[7],
                 1-average.errors.Insample$averagetypeI[3]-average.errors.Insample$averagetypeI[7],
                 1-average.errors.Insample$averagetypeII[2],
                 0)


outputs_1$ACRF<-c(average.errors.OptInsample$averagetypeI[5],
                  1-average.errors.OptInsample$averagetypeI[1]-average.errors.OptInsample$averagetypeI[5],
                  average.errors.OptInsample$averagetypeI[6],
                  1-average.errors.OptInsample$averagetypeI[2]-average.errors.OptInsample$averagetypeI[6],
                  average.errors.OptInsample$averagetypeI[7],
                  1-average.errors.OptInsample$averagetypeI[3]-average.errors.OptInsample$averagetypeI[7],
                  1-average.errors.OptInsample$averagetypeII[2],
                  0)
outputs_1$ACRFshift<-c(average.errors.OptInsample.shift$averagetypeI[5],
                       1-average.errors.OptInsample.shift$averagetypeI[1]-average.errors.OptInsample.shift$averagetypeI[5],
                       average.errors.OptInsample.shift$averagetypeI[6],
                       1-average.errors.OptInsample.shift$averagetypeI[2]-average.errors.OptInsample.shift$averagetypeI[6],
                       average.errors.OptInsample.shift$averagetypeI[7],
                       1-average.errors.OptInsample.shift$averagetypeI[3]-average.errors.OptInsample.shift$averagetypeI[7],
                       1-average.errors.OptInsample.shift$averagetypeII[2],
                       0)

outputs_1 <-reshape2::melt(outputs_1 ,id=c('prediction','number'))
colnames(outputs_1)<-c('prediction','number','Method','Coverage')
outputs_1$Coverage<-100*outputs_1$Coverage
outputs_1$x<-ifelse(outputs_1$number==0,1,
                    ifelse(outputs_1$number==1,5,
                           ifelse(outputs_1$number==2,9,13)))
df1<-outputs_1[outputs_1$Method=='CSForest',]
df2<-outputs_1[outputs_1$Method=='BCOPS',]
df3<-outputs_1[outputs_1$Method=='DC',]
df4<-outputs_1[outputs_1$Method=='CRF',]
df5<-outputs_1[outputs_1$Method=='ACRF',]
df6<-outputs_1[outputs_1$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1,
           aes(x=x,y=Coverage,fill=prediction),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2,aes(x=x+0.5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3,aes(x=x+0.5*2,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4,aes(x=x+0.5*3,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5,aes(x=x+0.5*4,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6,aes(x=x+0.5*5,y=Coverage,fill=prediction),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5,
                                25,25.5,26,26.5,27,27.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed",size=1)+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2                   R')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58

# 提取各列的 average_length 数据
csforest_average_length <- sapply(average.errors.CSForest.loop$averagetypeII,"[[",4)
bcops_average_length <- sapply(average.errors.BCOPS.loop$averagetypeII,"[[",4)
dc_average_length <- sapply(average.errors.Density.loop$averagetypeII,"[[",4)
crf_average_length <- sapply(average.errors.Insample.loop$averagetypeII,"[[",4)
acrf_average_length <- sapply(average.errors.OptInsample.loop$averagetypeII,"[[",4)
acrfshift_average_length <- sapply(average.errors.OptInsample.shift.loop$averagetypeII,"[[",4)




# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(csforest_average_length)), levels = method_order))
data$Average_Length <- c(csforest_average_length, bcops_average_length, dc_average_length, crf_average_length, acrf_average_length, acrfshift_average_length)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()




#Experiment 2
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp2_output_Network.txt"
sink(file_path)
average.errors.CSForest.loop.exp2=CSForest.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.CSForest.loop.exp2$averagetypeI)
print(average.errors.CSForest.loop.exp2$averagetypeII)
average.errors.BCOPS.loop.exp2=BCOPS.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.BCOPS.loop.exp2$averagetypeI)
print(average.errors.BCOPS.loop.exp2$averagetypeII)
average.errors.Density.loop.exp2=Density.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.Density.loop.exp2$averagetypeI)
print(average.errors.Density.loop.exp2$averagetypeII)
average.errors.Insample.loop.exp2=Insample.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.Insample.loop.exp2$averagetypeI)
print(average.errors.Insample.loop.exp2$averagetypeII)
average.errors.OptInsample.loop.exp2=OptInsample.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.OptInsample.loop.exp2$averagetypeI)
print(average.errors.OptInsample.loop.exp2$averagetypeII)
average.errors.OptInsample.shift.loop.exp2=OptInsample.shift.loop(n=10,n_train=500,n_test=300,n_test_outlier=0,type='exp2',mydata='protein',networkD=df,K=2)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeI)
print(average.errors.OptInsample.shift.loop.exp2$averagetypeII)
print(average.errors.OptInsample.shift.loop.exp2$alphas)
# 关闭文件
sink()

average.errors.CSForest.exp2=data_mean(average.errors.CSForest.loop.exp2)
average.errors.BCOPS.exp2=data_mean(average.errors.BCOPS.loop.exp2)
average.errors.Density.exp2=data_mean(average.errors.Density.loop.exp2)
average.errors.Insample.exp2=data_mean(average.errors.Insample.loop.exp2)
average.errors.OptInsample.exp2=data_mean(average.errors.OptInsample.loop.exp2)
average.errors.OptInsample.shift.exp2=data_mean(average.errors.OptInsample.shift.loop.exp2)

outputs_2<-matrix(0, ncol = 8, nrow = 6)
outputs_2<-as.data.frame(outputs_2)

colnames(outputs_2)<-c("Type","number","CSForest","BCOPS",
                       "DC","CRF","ACRF","ACRFshift")

outputs_2$Type<-rep(c('Correct Only ','Multi-Label'),3)
outputs_2$number<-c(0,0,1,1,2,2)
outputs_2$CSForest<-c(average.errors.CSForest.exp2$averagetypeI[5],
                      1-average.errors.CSForest.exp2$averagetypeI[1]-average.errors.CSForest.exp2$averagetypeI[5],
                      average.errors.CSForest.exp2$averagetypeI[6],
                      1-average.errors.CSForest.exp2$averagetypeI[2]-average.errors.CSForest.exp2$averagetypeI[6],
                      average.errors.CSForest.exp2$averagetypeI[7],
                      1-average.errors.CSForest.exp2$averagetypeI[3]-average.errors.CSForest.exp2$averagetypeI[7])

outputs_2$BCOPS<-c(average.errors.BCOPS.exp2$averagetypeI[5],
                   1-average.errors.BCOPS.exp2$averagetypeI[1]-average.errors.BCOPS.exp2$averagetypeI[5],
                   average.errors.BCOPS.exp2$averagetypeI[6],
                   1-average.errors.BCOPS.exp2$averagetypeI[2]-average.errors.BCOPS.exp2$averagetypeI[6],
                   average.errors.BCOPS.exp2$averagetypeI[7],
                   1-average.errors.BCOPS.exp2$averagetypeI[3]-average.errors.BCOPS.exp2$averagetypeI[7])

outputs_2$DC<-c(average.errors.Density.exp2$averagetypeI[5],
                1-average.errors.Density.exp2$averagetypeI[1]-average.errors.Density.exp2$averagetypeI[5],
                average.errors.Density.exp2$averagetypeI[6],
                1-average.errors.Density.exp2$averagetypeI[2]-average.errors.Density.exp2$averagetypeI[6],
                average.errors.Density.exp2$averagetypeI[7],
                1-average.errors.Density.exp2$averagetypeI[3]-average.errors.Density.exp2$averagetypeI[7])

outputs_2$CRF<-c(average.errors.Insample.exp2$averagetypeI[5],
                 1-average.errors.Insample.exp2$averagetypeI[1]-average.errors.Insample.exp2$averagetypeI[5],
                 average.errors.Insample.exp2$averagetypeI[6],
                 1-average.errors.Insample.exp2$averagetypeI[2]-average.errors.Insample.exp2$averagetypeI[6],
                 average.errors.Insample.exp2$averagetypeI[7],
                 1-average.errors.Insample.exp2$averagetypeI[3]-average.errors.Insample.exp2$averagetypeI[7])


outputs_2$ACRF<-c(average.errors.OptInsample.exp2$averagetypeI[5],
                  1-average.errors.OptInsample.exp2$averagetypeI[1]-average.errors.OptInsample.exp2$averagetypeI[5],
                  average.errors.OptInsample.exp2$averagetypeI[6],
                  1-average.errors.OptInsample.exp2$averagetypeI[2]-average.errors.OptInsample.exp2$averagetypeI[6],
                  average.errors.OptInsample.exp2$averagetypeI[7],
                  1-average.errors.OptInsample.exp2$averagetypeI[3]-average.errors.OptInsample.exp2$averagetypeI[7])

outputs_2$ACRFshift<-c(average.errors.OptInsample.shift.exp2$averagetypeI[5],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[1]-average.errors.OptInsample.shift.exp2$averagetypeI[5],
                       average.errors.OptInsample.shift.exp2$averagetypeI[6],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[2]-average.errors.OptInsample.shift.exp2$averagetypeI[6],
                       average.errors.OptInsample.shift.exp2$averagetypeI[7],
                       1-average.errors.OptInsample.shift.exp2$averagetypeI[3]-average.errors.OptInsample.shift.exp2$averagetypeI[7])

outputs_2 <-reshape2::melt(outputs_2 ,id=c('Type','number'))
colnames(outputs_2)<-c('Type','number','Method','Coverage')
outputs_2$Coverage<-100*outputs_2$Coverage
outputs_2$x<-ifelse(outputs_2$number==0,1,
                    ifelse(outputs_2$number==1,5,9))
df1.exp2<-outputs_2[outputs_2$Method=='CSForest',]
df2.exp2<-outputs_2[outputs_2$Method=='BCOPS',]
df3.exp2<-outputs_2[outputs_2$Method=='DC',]
df4.exp2<-outputs_2[outputs_2$Method=='CRF',]
df5.exp2<-outputs_2[outputs_2$Method=='ACRF',]
df6.exp2<-outputs_2[outputs_2$Method=='ACRFshift',]

ggplot()+
  geom_bar(data=df1.exp2,
           aes(x=x,y=Coverage,fill=Type),
           stat = "identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df2.exp2,aes(x=x+0.5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df3.exp2,aes(x=x+0.5*2,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df4.exp2,aes(x=x+0.5*3,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df5.exp2,aes(x=x+0.5*4,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  geom_bar(data=df6.exp2,aes(x=x+0.5*5,y=Coverage,fill=Type),
           stat="identity",position = position_stack(reverse = T),width=0.4)+
  theme_minimal()+
  scale_x_continuous(breaks = c(1,1.5,2,2.5,3,3.5,
                                5,5.5,6,6.5,7,7.5,
                                9,9.5,10,10.5,11,11.5,
                                13,13.5,14,14.5,15,15.5,
                                17,17.5,18,18.5,19,19.5,
                                21,21.5,22,22.5,23,23.5),
                     labels = c("CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift",
                                "CSForest","BCOPS","DC","CRF","ACRF","ACRFshift"))+
  scale_fill_manual(values = c("dodgerblue3","gray"))+
  geom_hline(aes(yintercept=95), colour="black", linetype="dashed")+
  xlab('')+ylab('percent')+
  ggtitle(label  = '',
          subtitle = '             0                    1                     2')+
  theme(axis.text.x = element_text(angle=90, hjust=1, vjust=.5))

#8.46*6.58
CSForest.len<-sapply(average.errors.CSForest.loop.exp2$averagetypeII,"[[",4)
BCOPS.len<-sapply(average.errors.BCOPS.loop.exp2$averagetypeII,"[[",4)
Density.len<-sapply(average.errors.Density.loop.exp2$averagetypeII,"[[",4)
Insample.len<-sapply(average.errors.Insample.loop.exp2$averagetypeII,"[[",4)
OptInsample.len<-sapply(average.errors.OptInsample.loop.exp2$averagetypeII,"[[",4)
OptInsample.shift.len<-sapply(average.errors.OptInsample.shift.loop.exp2$averagetypeII,"[[",4)

# 创建一个包含多个变量的数据框
# 创建有序因子顺序
method_order <- c("CSForest", "BCOPS", "DC", "CRF", "ACRF", "ACRFshift")

# 创建 ggplot 的 boxplot
data <- data.frame(Method = factor(rep(method_order, each = length(CSForest.len)), levels = method_order))
data$Average_Length <- c(CSForest.len, BCOPS.len, Density.len, Insample.len, OptInsample.len, OptInsample.shift.len)

ggplot(data, aes(x = Method, y = Average_Length)) +
  geom_boxplot() +#fill = "dodgerblue3"
  xlab(" ")+
  ylab("Average Length") +
  theme_minimal()


# Experiment 3
file_path <- "/Users/hanbujishenmebuhui/Desktop/X/BCOPS+/CSForest/logs/exp3_output_Network.txt"
sink(file_path)

models.size.loop<-function(n,n_train_lists,n_test_lists){
  #I means both II means train III means test
  num=length(n_train_lists)
  
  dat05<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat05)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat05['Train Size']=sort(rep(n_train_lists,n))
  dat05['Test Size']=sort(rep(n_test_lists,n))
  
  dat69<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(dat69)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  dat69['Train Size']=sort(rep(n_train_lists,n))
  dat69['Test Size']=sort(rep(n_test_lists,n))
  
  datI<-as.data.frame(matrix(,nrow=length(n_train_lists)*n,ncol=8))
  colnames(datI)<-c('Train Size','Test Size','CSForest','BCOPS','DC','CRF','ACRF','ACRFshift')
  datI['Train Size']=sort(rep(n_train_lists,n))
  datI['Test Size']=sort(rep(n_test_lists,n))
  
  print('Training CSForest...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['CSForest'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CSForest'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CSForest'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  print('Training BCOPS...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=BCOPS.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['BCOPS'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['BCOPS'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['BCOPS'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  print('Training Density...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Density.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['DC'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['DC'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['DC'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  
  print('Training CRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=Insample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['CRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['CRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['CRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  print('Training ACRF...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['ACRF'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRF'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRF'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  print('Training ACRFshift...')
  for (i in 1:num) {
    n_train=n_train_lists[i]
    n_test=n_test_lists[i]
    average.errors=OptInsample.shift.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test,mydata='protein',networkD=df,K=2)
    dat05['ACRFshift'][dat05['Train Size']==n_train&dat05['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",1)
    dat69['ACRFshift'][dat69['Train Size']==n_train&dat69['Test Size']==n_test]=sapply(average.errors$averagetypeII,"[[",2)
    datI['ACRFshift'][datI['Train Size']==n_train&datI['Test Size']==n_test]=sapply(average.errors$averagetypeI,"[[",4)
  }
  
  return(list(datI=datI,dat05=dat05, dat69 = dat69))
}


# MNIST
res.exp3.row1=models.size.loop(n=10,n_train_lists=c(50,80,100,150,200),n_test_lists=c(50,80,100,150,200))
res.exp3.row2=models.size.loop(n=10,n_train_lists=c(50,80,100,150,200),n_test_lists=c(200,200,200,200,200)) #fix test
res.exp3.row3=models.size.loop(n=10,n_train_lists=c(200,200,200,200,200),n_test_lists=c(50,80,100,150,200)) #fix train
print(res.exp3.row1)
print(res.exp3.row2)
print(res.exp3.row3)
sink()


#Plot

dat05.exp3.row1<-reshape2::melt(res.exp3.row1$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row1$Method<-as.factor(dat05.exp3.row1$Method)
dat69.exp3.row1<-reshape2::melt(res.exp3.row1$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row1$Method<-as.factor(dat69.exp3.row1$Method)
datI.exp3.row1<-reshape2::melt(res.exp3.row1$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row1)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row1$Method<-as.factor(datI.exp3.row1$Method)
#View(dat05.3)
#View(dat69.3)
#dat.3<-rbind(dat05.exp3.row1,dat69.exp3.row1,datI.exp3.row1)

##Row 1
datI.exp3.row1$Size=datI.exp3.row1$`Train Size`
p1.1<-ggplot(datI.exp3.row1,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class size (both)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A1')

dat05.exp3.row1$Size=dat05.exp3.row1$`Train Size`
p1.2<-ggplot(dat05.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A2')

dat69.exp3.row1$Size=dat69.exp3.row1$`Train Size`
p1.3<-ggplot(dat69.exp3.row1,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (both)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'A3')

##Row 2

dat05.exp3.row2<-reshape2::melt(res.exp3.row2$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row2$Method<-as.factor(dat05.exp3.row2$Method)
dat69.exp3.row2<-reshape2::melt(res.exp3.row2$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row2$Method<-as.factor(dat69.exp3.row2$Method)
datI.exp3.row2<-reshape2::melt(res.exp3.row2$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row2)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row2$Method<-as.factor(datI.exp3.row2$Method)

datI.exp3.row2$Size=datI.exp3.row2$`Train Size`
p2.1<-ggplot(datI.exp3.row2,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B1')

dat05.exp3.row2$Size=dat05.exp3.row2$`Train Size`
p2.2<-ggplot(dat05.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B2')

dat69.exp3.row2$Size=dat69.exp3.row2$`Train Size`
p2.3<-ggplot(dat69.exp3.row2,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (train)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'B3')



##Row 3

dat05.exp3.row3<-reshape2::melt(res.exp3.row3$dat05,id=c('Train Size',"Test Size"))
colnames(dat05.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat05.exp3.row3$Method<-as.factor(dat05.exp3.row3$Method)
dat69.exp3.row3<-reshape2::melt(res.exp3.row3$dat69,id=c('Train Size',"Test Size"))
colnames(dat69.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
dat69.exp3.row3$Method<-as.factor(dat69.exp3.row3$Method)
datI.exp3.row3<-reshape2::melt(res.exp3.row3$datI,id=c('Train Size',"Test Size"))
colnames(datI.exp3.row3)<-c('Train Size',"Test Size",'Method','value')
datI.exp3.row3$Method<-as.factor(datI.exp3.row3$Method)

datI.exp3.row3$Size=datI.exp3.row3$`Test Size`
p3.1<-ggplot(datI.exp3.row3,aes(x = Size,y =value,
                                group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type I error')+
  geom_hline(aes(yintercept=0.05), colour="black", linetype="dashed")+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C1')

dat05.exp3.row3$Size=dat05.exp3.row3$`Test Size`
p3.2<-ggplot(dat05.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (inlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C2')

dat69.exp3.row3$Size=dat69.exp3.row3$`Test Size`
p3.3<-ggplot(dat69.exp3.row3,aes(x = Size,y =value,
                                 group=Method,color=Method)) +                            
  stat_summary(fun="mean",geom="point",size=2) +        
  stat_summary(fun="mean",geom="line") +  
  stat_summary(fun.data = "mean_se",geom = "errorbar",width=0.05)+
  #scale_color_aaas()+
  scale_color_aaas()+
  xlab("per class (test)")+
  ylab('Type II error (outlier)')+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank())#+ggtitle(label  = 'C3')

#range plots
library(ggpubr)
ggarrange(
  p1.2,p2.2,p3.2,
  p1.3,p2.3,p3.3,
  # labels = c("A1", "B1",'C1',
  #            "A2", "B2","C2"),
  ncol = 3, nrow = 2,common.legend = T,legend = "bottom",font.label = list(size = 8, color = "black"))

# Exp 4

n_train=50
n_test=n_test_outlier=50
n=10
alpha=0.05

w=0
average.errors.CSForest.loop.a05w0=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w0=data_mean(average.errors.CSForest.loop.a05w0)
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w0$averagetypeII,"[[",2))


w="LOG"
average.errors.CSForest.loop.a05wl=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05wl=data_mean(average.errors.CSForest.loop.a05wl)
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05wl$averagetypeII,"[[",2))



w=0.005
average.errors.CSForest.loop.a05w1=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w1=data_mean(average.errors.CSForest.loop.a05w1)
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w1$averagetypeII,"[[",2))


w=0.105
average.errors.CSForest.loop.a05w15=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w15=data_mean(average.errors.CSForest.loop.a05w15)
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w15$averagetypeII,"[[",2))

w=0.15
average.errors.CSForest.loop.a05w2=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w25=data_mean(average.errors.CSForest.loop.a05w2)
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w2$averagetypeII,"[[",2))



w=0.5
average.errors.CSForest.loop.a05w5=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w5=data_mean(average.errors.CSForest.loop.a05w5)
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w5$averagetypeII,"[[",2))



w=1
average.errors.CSForest.loop.a05w10=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w10=data_mean(average.errors.CSForest.loop.a05w10)
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w10$averagetypeII,"[[",2))


w=10
average.errors.CSForest.loop.a05w100=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w100=data_mean(average.errors.CSForest.loop.a05w100)
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w100$averagetypeII,"[[",2))

w=100
average.errors.CSForest.loop.a05w1002=CSForest.loop(n=n,n_train=n_train,n_test=n_test,n_test_outlier=n_test_outlier,mydata='protein',networkD=df,K=2,alpha=alpha,weight=w)
average.errors.CSForest.a05w1002=data_mean(average.errors.CSForest.loop.a05w1002)
sd(sapply(average.errors.CSForest.loop.a05w1002$averagetypeII,"[[",1))
sd(sapply(average.errors.CSForest.loop.a05w1002$averagetypeII,"[[",2))



