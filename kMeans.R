install.packages("tidyverse")
library(tidyverse)
install.packages("cluster")
library(cluster)
install.packages("factoextra")
library(factoextra)

setwd("C:/Users/Nitin Rangarajan")
read_bcp_data <- read.csv("wpbc.csv")
original_data <- read.csv("wpbc.csv")
summary(read_bcp_data)

# Question (a) 
# We need to scale the data 
read_bcp_data <- scale(read_bcp_data[-1])
# Runnings k-means with k=4
head(read_bcp_data)

#Calculate distance between the scaled features
distance_between_points <- get_dist(read_bcp_data)
distance_between_points
#Visualize distance between points using fviz_dist
fviz_dist(distance_between_points, gradient = list(low = "#cdf441", mid = "white", high = "#41f4c1"))

#Generate k-means
kmeans <- kmeans(read_bcp_data, centers = 4)
kmeans

#fviz_cluster(kmeans, data = read_bcp_data)

#Identify cluster centers
kmeans$centers
#SSE value
kmeans$withinss
# Total SSE
kmeans$tot.withinss

#Repeating the clustering twice again. 
kmeans_trial2 <-  kmeans(read_bcp_data, centers = 4)
kmeans_trial2

fviz_cluster(kmeans_trial2, data = read_bcp_data)

#Identify cluster centers
kmeans_trial2$centers
#SSE value
kmeans_trial2$withinss
#Total SSE
kmeans_trial2$tot.withinss

#Repeating the clustering third time. 
kmeans_trial3 <-  kmeans(read_bcp_data, centers = 4)
kmeans_trial3

fviz_cluster(kmeans_trial2, data = read_bcp_data)

#Identify cluster centers
kmeans_trial3$centers
#SSE value
kmeans_trial3$withinss
#Total SSE
kmeans_trial3$tot.withinss

# Comparison of cluster SSE's
kmeans$tot.withinss
kmeans_trial2$tot.withinss
kmeans_trial3$tot.withinss

#We shall proceed with cluster 2

#To find ideal cluster size bu elbow method
set.seed(123)
fviz_nbclust(read_bcp_data, kmeans_trial2, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

# Function to compute average sillhoute
sil_kmeans <- silhouette(kmeans_trial2$cluster, dist(read_bcp_data))
(sil_kmeans)

sil_grp<-data.frame(cluster=(sil_kmeans[,1]),sil_width=sil_kmeans[,3])
plot(sil_kmeans)

#Summary Plot
kmeans_1 <- fviz_cluster(kmeans, geom = "point", data = read_bcp_data) + ggtitle("Trial 1, k = 3")
kmeans_2 <- fviz_cluster(kmeans_trial2, geom = "point", data = read_bcp_data) + ggtitle("Trial 2, k = 3")
kmeans_3 <- fviz_cluster(kmeans_trial3, geom = "point", data = read_bcp_data) + ggtitle("Trial 3, k = 3")
install.packages("gridExtra")
library(gridExtra)
grid.arrange(kmeans_1, kmeans_2, kmeans_3, nrow = 2)


clusterCenters <- as.data.frame(kmeans_trial2$centers)
clusterCenters
class(clusterCenters)
clusterCenters$Class <- 0
clusterCenters$Class <- c('N','N','N','N')
clusterCenters

fviz_dist(get_dist(bcpData1Norm,method = "euclidean"))

bcpData1NormModified <- bcpData1Norm
nrow(bcpData1Norm)
nrow(bcpData1NormModified)
bcpData1NormModified <- rbind(bcpData1NormModified,clusterCenters[,-31])
nrow(bcpData1NormModified)
tail(bcpData1NormModified)
str(bcpData1NormModified)

distMatrix <- get_dist(bcpData1NormModified,method = "euclidean")
distWRTcentres <- tail(as.matrix(distMatrix),4)
distWRTcentres <- distWRTcentres[,-c(199,200,201,202)]
distWRTcentres

assignPoint <- function(x){
  which.min(x)
}

#d
Class_labels = original_data$Outcome
Cluster_labels = kmeans_trial2$cluster
table(original_data$Outcome, kmeans_trial2$cluster)


#Merge Cluster Center with Scaled Data
new_data <- rbind(read_bcp_data,as.data.frame(kmeans_trial2$centers))
euclidean_dist = as.matrix(get_dist(new_data,method = "euclidean"))
euclidean_dist
