function [ Fitness ] = MCI_Conversion_FitnessRusBoost( pop )

global X;
global Y;

X1 = X(:, find(pop==1));
Y1 = Y;

[model, confMat] = GenerateRUSBoostModel(X1, Y1);
Fitness = (confMat(1,1)+2*confMat(2,2))/300;

end

