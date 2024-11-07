# https://github.com/kendomaniac/rCASC

```console
R/autoencoderforclusteringgpu.R:#autoencoder4clusteringGPU(group=c("sudo"), scratch.folder=scratch, file=file,separator=",", bias="ALL", permutation=10, nEpochs=100,patiencePercentage=5,seed=1111,projectName=projectName,bN="NULL",lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.00000001,decay=0.0,loss="mean_squared_error",regularization=10)
R/autoencoderforclusteringgpu.R:autoencoder4clusteringGPU <- function(group=c("sudo","docker"), scratch.folder, file,separator, bias, permutation, nEpochs,patiencePercentage=5,seed=1111,projectName,bN="NULL",lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.00000001,decay=0.0,loss="mean_squared_error",regularization=10){
R/autoencoderforclusteringgpu.R:    params <- paste("--cidfile ",data.folder,"/dockerID -v ",scrat_tmp.folder,":/scratch -v ", data.folder, ":/data --gpus all -d repbioinfo/autoencoderforclusteringgpu python3 /home/autoencoder.py ",matrixNameC,".",format," ",separator," ",bias," ",permutation," ",nEpochs," ",patiencePercentage," ",projectName," ",seed," ",bN," ",lr," ",beta_1," ",beta_2," ",epsilon," ",decay," ",loss,sep="")
docs/sitemap.xml:    <loc>/reference/autoencoder4clusteringGPU.html</loc>
NAMESPACE:export(autoencoder4clusteringGPU)
man/autoencoder4clusteringGPU.Rd:% Please edit documentation in R/autoencoderforclusteringgpu.R
man/autoencoder4clusteringGPU.Rd:\name{autoencoder4clusteringGPU}
man/autoencoder4clusteringGPU.Rd:\alias{autoencoder4clusteringGPU}
man/autoencoder4clusteringGPU.Rd:autoencoder4clusteringGPU(

```
