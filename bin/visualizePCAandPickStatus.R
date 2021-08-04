#!/usr/bin/env Rscript
library(ggplot2)
library(reshape)

args <- commandArgs(TRUE)
inputData <- args[1]
outputDir <- args[2]

data <- read.csv(inputData, stringsAsFactors = F)

pdf(paste(outputDir, "/PCAdata.pickStatus.pdf", sep = ""), width = 5, height = 4)
ggplot(data, aes(x = PCA1, y = PCA2, color = pickStatus, alpha = pickStatus)) +
    geom_point(size = 0.5) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_classic() + scale_color_manual(values = c("blue", "grey", "red")) + 
	scale_alpha_discrete(range = c(0.3, 1)) + 
    labs(x = "PCA1", y = "PCA2", title = "Feature-based optimized picking") +
    theme(plot.title = element_text(hjust = 0.5))
dev.off()

pdf(paste(outputDir, "/PCAdata.plateBarcode.pdf", sep = ""), width = 5, height = 4)
ggplot(data, aes(x = PCA1, y = PCA2, color = plateBarcode)) +
    geom_point(alpha = 0.7, size = 0.3) + 
    theme(plot.title = element_text(hjust = 0.5)) +
    theme_classic() + 
#	scale_color_manual(values = c("#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"))+
    labs(x = "PCA1", y = "PCA2", title = "Feature-based optimized picking") +
    theme(plot.title = element_text(hjust = 0.5))
dev.off()





