#!/usr/bin/env Rscript
library(ggplot2)
library(reshape)


args = commandArgs(trailingOnly=TRUE)
inputXY = args[1]
outputXY = args[2]
reference = args[3]
Xmodel_path = paste(reference, "/offsetCorrection.Xmodel.rda", sep = "")
Ymodel_path = paste(reference, "/offsetCorrection.Ymodel.rda", sep = "")
load(Xmodel_path)
load(Ymodel_path)

data.csv <- read.csv(inputXY, header = F)
input.df <- data.frame(real_X = data.csv$V1, real_Y = data.csv$V2)
offset_X <- predict(data_offsetX.model, input.df)
offset_Y <- predict(data_offsetY.model, input.df)
output.df <- round(data.frame(input_X = data.csv$V1 - offset_X, input_Y = data.csv$V2 - offset_Y))
write.table(output.df, outputXY, sep = ",", quote = F, col.names = F, row.names = F)

