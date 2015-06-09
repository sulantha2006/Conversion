function [ fitness ] = getFitnessFromConfusionMat( confumat )

precision = confumat(2,2)/(confumat(1,2)+confumat(2,2));
recall = confumat(2,2)/(confumat(2,1)+confumat(2,2));

fitness = 2*precision*recall/(precision+recall);

end

