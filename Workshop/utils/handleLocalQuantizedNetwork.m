function handleLocalQuantizedNetwork(NNTNetwork, Filename, varargin)
%

% Copyright 2024 The Mathworks, Inc.

defaultOpset = 8;

nnet.internal.cnn.onnx.setAdditionalResourceLocation();     % For SPKG resource catalog.
% Check input
[NNTNetwork, Filename, NetworkName, OpsetVersion, BatchSize] = iValidateInputs(NNTNetwork, Filename, 10, varargin{:});
% Make Projected Networks exportable.
NNTNetwork = deep.internal.sdk.projection.prepareProjectedNetworkFor3pExporters(NNTNetwork);

% Store the quantization details
qNetLocalDetails = quantizationDetails(NNTNetwork); 
[exponentsDataTable, qInfo] = getExponentsDataTableFromQNetInLayerwiseOrder(NNTNetwork);
qInfoComposite.qInfo = qInfo; 
qInfoComposite.exponentsDataTable = exponentsDataTable.exponentsData; 

% Make Network ST compatible
NNTNetwork = iMakeNetworkSTCompatible(NNTNetwork);
% Update metadata
metadata = nnet.internal.cnn.onnx.NetworkMetadata;
metadata.NetworkName = NetworkName;
metadata.OpsetVersion = OpsetVersion;
% Convert
converter   = ConverterForQuantizedNetwork(NNTNetwork, metadata, BatchSize, qNetLocalDetails, qInfoComposite);
modelProto  = toOnnx(converter);
% Write
writeToFile(modelProto, Filename);
end

function [NNTNetwork, Filename, NetworkName, OpsetVersion, BatchSize] = iValidateInputs(NNTNetwork, Filename, defaultOpset, varargin)
% Setup parser
par = inputParser();
par.addRequired('NNTNetwork');
par.addRequired('Filename');
par.addParameter('NetworkName', "Network");
par.addParameter('OpsetVersion', defaultOpset);
par.addParameter('BatchSize', []);
% Parse
par.parse(NNTNetwork, Filename, varargin{:});
NetworkName = par.Results.NetworkName;
OpsetVersion = par.Results.OpsetVersion;
BatchSize = par.Results.BatchSize;
% Validate
NNTNetwork  = iValidateNetwork(NNTNetwork);
Filename    = iValidateFilename(Filename);
NetworkName = iValidateNetworkName(NetworkName);
OpsetVersion = iValidateOpsetVersion(OpsetVersion);
BatchSize = iValidateBatchSize(BatchSize);
end

function Network = iValidateNetwork(Network)
if ~(isa(Network, 'DAGNetwork') || isa(Network, 'SeriesNetwork') ...
        || isa(Network, 'nnet.cnn.LayerGraph')|| isa(Network, 'dlnetwork'))
    error(message('nnet_cnn_onnx:onnx:NetworkWrongType'));
end
% Warn if quantized
if (isa(Network, "SeriesNetwork") || isa(Network, "DAGNetwork") || isa(Network, "dlnetwork")) ...
        && quantizationDetails(Network).IsQuantized
    disp("Start exporting Quantized Network to ONNX"); 
end
end

function Filename = iValidateFilename(Filename)
if ~(isstring(Filename) || ischar(Filename))
    error(message('nnet_cnn_onnx:onnx:FilenameWrongType'));
end
Filename = char(Filename);
end

function NetworkName = iValidateNetworkName(NetworkName)
if ~(isstring(NetworkName) || ischar(NetworkName))
    error(message('nnet_cnn_onnx:onnx:NetworkNameWrongType'));
end
NetworkName = char(NetworkName);
end

function OpsetVersion = iValidateOpsetVersion(OpsetVersion)
SupportedOpsetsForExport = 6:14;
if ~any(OpsetVersion == SupportedOpsetsForExport)
    error(message('nnet_cnn_onnx:onnx:OpsetVersionUnsupportedForExport', num2str(SupportedOpsetsForExport)));
end
OpsetVersion = double(OpsetVersion);
end

function BatchSize = iValidateBatchSize(BatchSize)
% Must be a single integer greater than zero, or []. Integers are made
% double, and [] is changed to "BatchSize".
isvalid = isequal(BatchSize,[]) || ...
    (isreal(BatchSize) && isscalar(BatchSize) && isfinite(BatchSize) ...
    && BatchSize>0 && BatchSize==floor(BatchSize));
if ~isvalid
    error(message('nnet_cnn_onnx:onnx:BatchSizeWrongType'));
end
if isempty(BatchSize)
    BatchSize = "BatchSize";
else
    BatchSize = double(BatchSize);
end
end

function iWarningNoBacktrace(msg)
    warnstate = warning('off','backtrace');
    C = onCleanup(@()warning(warnstate));
    warning(msg);
end

function NNTNetwork = iMakeNetworkSTCompatible(NNTNetwork)
    % Add flattenLayers before RNN and FullyConnected layers only for
    % dlnetworks
    if isa(NNTNetwork, 'dlnetwork')    
        lg = layerGraph(NNTNetwork);  
        addFlattenIdx = arrayfun(@(x) (isLayerRNN(x) || isLayerFullyConnected(x)), lg.Layers, 'UniformOutput', true);
        % Replace the current layer to be modified with a layer array containing
        % flattenLayer followed by the current layer
        layerIdx = find(addFlattenIdx);
        if ~isempty(layerIdx)
            layersToModify = lg.Layers(addFlattenIdx);
            newLayerArrays = arrayfun(@(x) [flattenLayer("Name",sprintf("flatten_%s", x.Name)) x], layersToModify, 'UniformOutput', false);
            
            for i=1:numel(layerIdx)
                [lg, updateIdx] = prependFlattenLayer(lg, layerIdx(i), newLayerArrays{i}');
                % Updating position of subsequent RNN and FC layers to account for
                % inserted flatten layer
                if updateIdx
                    layerIdx = layerIdx+1;
                end
            end
        end
        NNTNetwork = dlnetwork(lg);
    end
end

function tf = isLayerRNN(layer)
    tf = (isa(layer, 'nnet.cnn.layer.LSTMLayer') || ...
        isa(layer, 'nnet.cnn.layer.LSTMProjectedLayer') || ...
        isa(layer, 'nnet.cnn.layer.BiLSTMLayer') ||...
        isa(layer, 'nnet.cnn.layer.GRULayer') || ...
        isa(layer, 'nnet.cnn.layer.GRUProjectedLayer'));
end

function tf = isLayerFullyConnected(layer)
    tf = isa(layer, 'nnet.cnn.layer.FullyConnectedLayer');
end

function tf = isLayerFlatten(layer)
    tf = isa(layer, 'nnet.cnn.layer.FlattenLayer');
end

function [lg, updateIdx] = prependFlattenLayer(lg, layerIdx, newLayerArray)
    updateIdx = false;
    currLayer = lg.Layers(layerIdx);

    % Identify input to the current layer
    lgDiGraph = lg.extractPrivateDirectedGraph;
    inConnsIdx = lgDiGraph.predecessors(layerIdx);

    % Check whether the layer is intermediate or first in the network
    hasPrevLayer = ~isempty(inConnsIdx);
    if hasPrevLayer
        % If the layer is intermediate, only add flatten layer if previous 
        % layer isn't an RNN, fullyConnected or flatten layer  
        inputConnIdx = inConnsIdx(1);
        prevLayer = lg.Layers(inputConnIdx);          
        if ~(isLayerRNN(prevLayer) || isLayerFullyConnected(prevLayer) || isLayerFlatten(prevLayer))
            % For RNN Layers with multi-inputs            
            % Cannot use replaceLayer to replace multi-input layers w/ single
            % input layers
            if isLayerRNN(currLayer) && currLayer.HasStateInputs  
                % Insert flattenLayer before input to the RNN Layer
                lg = disconnectLayers(lg,prevLayer.Name, sprintf('%s/in', currLayer.Name));
                lg = addLayers(lg, flattenLayer("Name", sprintf("flatten_%s", currLayer.Name)));
                lg = connectLayers(lg, prevLayer.Name, sprintf("flatten_%s", currLayer.Name));
                lg = connectLayers(lg, sprintf("flatten_%s", currLayer.Name), sprintf('%s/in', currLayer.Name));
                % Don't update position of subsequent RNN and FC layers since the inserted flatten
                % layer is added at the end of the layerGraph
                updateIdx = false;
            else
                % For FullyConnected and RNN Layers w/ single input
                lg = replaceLayer(lg, currLayer.Name, newLayerArray);
                updateIdx = true;
            end
        end
    else
        % Replace the layer if it is first in the network
        lg = replaceLayer(lg, currLayer.Name, newLayerArray);
        updateIdx = true;        
    end
end