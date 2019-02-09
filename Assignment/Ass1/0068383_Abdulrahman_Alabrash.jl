#@Abdulrahman Alabrash

using DelimitedFiles, Statistics, Random
Random.seed!(1);

@doc readdlm;
download("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data","housing.data.txt")
input=readdlm("housing.data.txt", Float64)

y=reshape(input[:,14],(1,506))

newmatrix=transpose(((input.-mean(input,dims=1))./std(input,dims=1))[:,1:13])

ids=randperm(506);

traIndices=ids[1:400]
xtra=newmatrix[:,traIndices]
ytra=y[:,traIndices]

testIndices=ids[401:506]
xtest=newmatrix[:,testIndices]
ytest=y[:,testIndices]

#Ex 5 
w=randn((1,13)).*0.1

function predict(w,x)
    w*x
end
ypredTest=predict(w,xtest)
ypredTrain=predict(w,xtra)

function MSE(y,ŷ)
    n=length(ŷ)
    j=(.5/n)*(sum(abs2.(y-ŷ) ))
end

jTrain=MSE(ytra,ypredTrain)

jTest=MSE(ytest,ypredTest)

absoluteDiff=(abs.((ypredTrain-ytra)))
jSqrt=sqrt.(jTrain)
count(sqrt.(jTrain).>absoluteDiff)
