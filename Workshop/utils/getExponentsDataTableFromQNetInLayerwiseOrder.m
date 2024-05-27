function [exponents, qInfoComposite] = getExponentsDataTableFromQNetInLayerwiseOrder(qNet)
%getExponentsDataTableFromQNetInLayerwiseOrder returns exponents for all the calibrated
%entities : Activation, Weights, Bias, Parameter. The output is formatted in a specific format.

%   Copyright 2024 The MathWorks, Inc.

% The output of this utility 'exponents' looks like this :
% exponents =
%
%     struct with fields:
%
%     exponentsData: [1×29 struct]

% exponentsData is a struct array. Each struct of this array has
% three fields : Exponent, DLT_LayerName, and EntityType.

arguments
    qNet {mustBeA(qNet,{'DAGNetwork', 'dlnetwork'})}
end

% Using QuantizationInfoComposite for future compatibility. As exponents for
% intermediate entities will be possible to be fetched from this composite.
% An empty qInfoComposite means 'quantize' wasn't already run on the
% network
qDetails = quantizationDetails(qNet);
qInfoComposite = deep.internal.quantization.getQuantizationInfoComposite(qNet);
qInfoTable  = getQuantizationInfoTable(qNet, qDetails);
if isempty(qInfoTable)
    error('dlq:qNetToExponentsUtil:UnquantizedNetwork','The input is not a quantized deep learning network');
end

% Parse for Activations, Weights, Bias, Parameter
layerNames = {qInfoTable.LayerName};
exponentsData = struct('Exponent', [], 'DLT_LayerName', [], ...
    'EntityType', []);
expIdx = 1;
valueNamesToLookFor = {'Activations', 'Weights', 'Bias', 'Parameter'};
for idx = 1:numel(layerNames)
    layerQuantizationInfo = qInfoTable(idx).QuantizationInfo;
    for jdx = 1:numel(valueNamesToLookFor)
        % Add entry for one entity in the exponentsData struct array
        if layerQuantizationInfo.hasValueConfig(valueNamesToLookFor{jdx})
            vc = layerQuantizationInfo.getValueConfig(valueNamesToLookFor{jdx});
            exponentsData(expIdx) = struct('Exponent', ...
                vc.Codegen.ScalingExponent,...
                'DLT_LayerName', layerNames{idx}, 'EntityType', valueNamesToLookFor{jdx});
            expIdx = expIdx+1;
        end
    end
end
exponents.exponentsData = exponentsData;
end

function quantizationStruct = getQuantizationInfoTable(qNet, ~)
% quantizationStruct = Struct (LayerName --> QuantizationInfo)
quantizationStruct = struct('LayerName', [], 'QuantizationInfo', []);
% internalLayers: cell array of nnet.internal.cnn.layer.Layer
% for each internal layer, extract the quantization info and
% save it as a value in a struct
internalLayers = deep.internal.quantization.getInternalLayers(qNet);
ii = 1;
for idx = 1:numel(internalLayers)
    layer = internalLayers{idx};
    if ~isempty(layer.QuantizationInfo)
        % layN = layer.Name;
        % if ismember(layN, qDetails.QuantizedLayerNames)
        quantizationStruct(ii).LayerName = layer.Name;
        quantizationStruct(ii).QuantizationInfo = layer.QuantizationInfo;
        ii = ii+1;
        % end
    end
end
end