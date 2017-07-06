% Converts labels into binary vectors (One Hot Encoding)
function encodedLabels = encodeLabels(labels)

valueLabels = unique(labels);
nLabels = length(valueLabels);
nSamples = size(labels,1);

encodedLabels = zeros(nSamples, nLabels);

for i = 1:nLabels
	encodedLabels(:,i) = (labels == valueLabels(i));
end