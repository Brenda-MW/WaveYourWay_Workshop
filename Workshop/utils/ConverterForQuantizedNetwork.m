classdef ConverterForQuantizedNetwork
    % Class to convert a Network into ONNX

    %  Example ModelProto:
    %           ir_version: 1
    %         opset_import: []
    %        producer_name: 'onnx-caffe2'
    %     producer_version: []
    %               domain: []
    %        model_version: []
    %           doc_string: []
    %                graph: [1x1 nnet.internal.cnn.onnx.GraphProto]
    %       metadata_props: []

    % Copyright 2018-2023 The Mathworks, Inc.

    properties
        Network
        Metadata
        BatchSizeToExport   % An integer or a string
        qDetails % Quantization details
        qInfoComposite
    end

    methods
        function this = ConverterForQuantizedNetwork(network, metadata, BatchSizeToExport, qNetLocalDetails, qInfoComposite)
            this.Network = network;
            this.Metadata = metadata;
            this.BatchSizeToExport = BatchSizeToExport;
            this.qDetails = qNetLocalDetails; 
            this.qInfoComposite =qInfoComposite; 
        end

        function modelProto = toOnnx(this)
            import nnet.internal.cnn.onnx.*
            % Set ONNX operatorSet version number
            opsetIdProto = OperatorSetIdProto;
            opsetIdProto.version = int64(this.Metadata.OpsetVersion);

            % ModelProto fields
            modelProto                  = ModelProto;
            modelProto.ir_version       = int64(this.Metadata.IrVersion);
            modelProto.opset_import     = opsetIdProto;
            modelProto.producer_name    = char(this.Metadata.ProducerName);
            modelProto.producer_version = char(this.Metadata.ProducerVersion);
            modelProto.domain           = char(this.Metadata.Domain);
            modelProto.model_version  	= int64(this.Metadata.ModelVersion);
            modelProto.doc_string       = char(this.Metadata.DocString);
            modelProto.graph            = networkToGraphProto(this);
            modelProto.metadata_props	= [];

            % Optionally add MathWorks operatorSet
            if MathWorksOperatorsUsed(modelProto)
                opsetIdProto = OperatorSetIdProto;
                opsetIdProto.domain = 'com.mathworks';
                opsetIdProto.version = int64(this.Metadata.MathWorksOpsetVersion);
                modelProto.opset_import(end+1) = opsetIdProto;
            end
        end

        function graphProto = networkToGraphProto(this)
            %   Example GraphProto:
            %            node: [1×66 nnet.internal.cnn.onnx.NodeProto]
            %            name: 'squeezenet'
            %     initializer: [1×52 nnet.internal.cnn.onnx.TensorProto]
            %      doc_string: []
            %           input: [1×53 nnet.internal.cnn.onnx.ValueInfoProto]
            %          output: [1×1 nnet.internal.cnn.onnx.ValueInfoProto]
            %      value_info: []
            import nnet.internal.cnn.onnx.*

            % Convert layers while gathering nodes, initializers, network
            % inputs, network outputs, and the TensorNameMap.
            nodeProtos              = NodeProto.empty;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];

            % The TensorNameMap maps a tensor name in the DLT network to
            % the name of the ONNX tensor that plays the same role in the
            % ONNX graph. For example: Suppose a DLT fullyConnectedLayer
            % has the name "FC1". We consider the name "FC1" to refer to
            % the layer's output tensor. When exported to ONNX, that layer
            % may expand into 2 operators, e.g., Flatten followed by Gemm.
            % The output tensor of the Gemm operator may be given the name
            % "FC1_Gemm". In that case, the association {"FC1", "FC1_Gemm"}
            % will be added to the TensorNameMap, so that
            % TensorNameMap('FC1') = 'FC1_Gemm'. All tensors whose names
            % change will be added to the TensorNameMap.
            TensorNameMap           = containers.Map;

            % The TensorLayoutMap maps an ONNX tensor name to a layout
            % string. Not all tensors are included. Examples:
            %   TensorLayoutMap('Conv_1',   'nchw')
            %   TensorLayoutMap('FC1_Gemm', 'nc')
            %   TensorLayoutMap('LSTM_1',   'snc').
            TensorLayoutMap         = containers.Map;

            % If current network is dlnetwork, change it to layergraph
            if isa(this.Network, 'dlnetwork')
                this.Network = layerGraph(this.Network);
            end

            networkAnalysis         = nnet.internal.cnn.analyzer.NetworkAnalyzer(this.Network);

            % Valid network for dlnetwork and layergraph
            % return names of dangling layers if no output layer for
            % dlnetwork and layergraph
            danglingLayerNames = iValidateNetwork(this, networkAnalysis);

            % Quantization information
            qLayerNames = this.qDetails.QuantizedLayerNames; 
            qLearnablesTable = this.qDetails.QuantizedLearnables; 
            qInfoFields = this.qInfoComposite; 

            IsRecurrentNetwork      = isa(this.Network.Layers(1), 'nnet.cnn.layer.SequenceInputLayer');
            for layerNum = 1:numel(networkAnalysis.LayerAnalyzers)
                % LayerAnalyzers are in topological order
                layerAnalyzer           = networkAnalysis.LayerAnalyzers(layerNum);
                isDanglingLayer = isa(this.Network, 'nnet.cnn.LayerGraph') && ismember(layerAnalyzer.Name, danglingLayerNames);
                isQuantizedLayer = ismember(layerAnalyzer.Name, qLayerNames); 
                layerConverter          = NNTLayerConverter.makeLayerConverter(layerAnalyzer, ...
                    this.Metadata.OpsetVersion, IsRecurrentNetwork, isDanglingLayer, this.BatchSizeToExport);
                [nodeProtos, paramInitializers, netInputs, netOutputs, TensorNameMap, TensorLayoutMap] ...
                    = toOnnx(layerConverter, nodeProtos, TensorNameMap, TensorLayoutMap);

                % retrieve information from the table
                % idx = (qLearnablesTable.Layer ==layerAnalyzer.Name);
                % tbl_learnables = qLearnablesTable(idx, :);
                % qlayerConverter  = ConverterForQuantizedWeightsBiasEntity(layerAnalyzer,...
                %                             tbl_learnables.Parameter{1}, -1, class(tbl_learnables.Value{1}), this.Metadata.OpsetVersion);
                % % qlayerConverter.NNTLayer.Name  = layerAnalyzer.Name;
                % [nodeProtos, parameterInitializers, ~] = toOnnx(qlayerConverter, nodeProtos,...
                %                 paramInitializers, TensorNameMap, TensorLayoutMap);


                parameterInitializers   = [parameterInitializers, paramInitializers];
                networkInputs           = [networkInputs, netInputs];
                networkOutputs          = [networkOutputs, netOutputs];
            end

            if this.qDetails.IsQuantized
                [nodeProtos, parameterInitializers, networkOutputs] = addQuantizationInfo(this, nodeProtos,...
                    parameterInitializers, networkOutputs, TensorNameMap, networkAnalysis, qInfoFields.exponentsDataTable, this.qDetails, qInfoFields.qInfo);

            end

            % Set graphProto fields
            graphProto              = GraphProto;
            graphProto.name         = this.Metadata.NetworkName;
            graphProto.node         = nodeProtos;
            graphProto.initializer	= parameterInitializers;
            graphProto.input        = networkInputs;
            graphProto.output       = networkOutputs;
        end

        function danglingLayerNames = iValidateNetwork(this, networkAnalysis)
            % Validate network only for layergraph and dlnetwork
            danglingLayerNames = {};

            if isa(this.Network, 'nnet.cnn.LayerGraph')
                constraints = nnet.internal.cnn.onnx.getConverterConstraints();
                converterConstraintFilter = nnet.internal.cnn.onnx.constraints.ConverterConstraintFilter(constraints);
                networkAnalysis.applyConstraints(converterConstraintFilter);
                ids = networkAnalysis.Issues.Id;

                if ~isempty(ids)
                    % If has any error other than the filtered issues,
                    % throw error
                    networkAnalysis.throwIssuesIfAny();
                end

                if ~converterConstraintFilter.HasMissingOutput
                    % If there are no missing outputs, return
                    return
                end

                % Get dangling nodes for layergraph
                digraph = this.Network.extractPrivateDirectedGraph();
                outputLayers = this.Network.Layers(digraph.outdegree == 0);
                danglingLayerNames = {outputLayers.Name};
            end
        end
        function [nodeProtos, parameterInitializers, networkOutputs] = addQuantizationInfo(this, nodeProtos,...
                parameterInitializers, networkOutputs, TensorNameMap, networkAnalysis, qdqTable, qDetails, qInfoComposite)
            import nnet.internal.cnn.onnx.*
            allExistingNodeProtoNames = {nodeProtos.name};
            % fuse batchnorm layer with conv layers.
            qTensorNameMap         = containers.Map;
            for layer = 1:length(this.Network.Layers)
                % if isa(this.Network.Layers(layer), 'nnet.cnn.layer.BatchNormalizationLayer')
                %     bnLayerAnalyzer  = networkAnalysis.LayerAnalyzers(layer);
                %     tf               = isBatchnormFused(bnLayerAnalyzer.Name, qInfoComposite);
                %     qLayerConverter  = dlq.ConverterForFusedBatchnormLayers(bnLayerAnalyzer, tf, this.Metadata.OpsetVersion);
                %     [nodeProtos, parameterInitializers, qTensorNameMap] = ...
                %         toOnnx(qLayerConverter, nodeProtos, parameterInitializers,...
                %         TensorNameMap, qTensorNameMap);
                % end
            end
            % Embed quantization information to the already created nodeprotos.
            % based on the entity type in the exponents data table,
            % create following nodes.
            %    Entity Type                ONNX Node
            % 1. Activation ---> QuantizeLinear and DequantizeLinear
            % 2. Weights    ---> DequantizeLinear node (for weights)
            % 3. Bias       ---> DequantizeLinear node (for bias)
            % 4. Parameter  ---> no operation

            allLayerNames          = {this.Network.Layers.Name};
            % Obtain the number of nodeprotos after removing batchnorm
            % nodes. This is obtained to change the inputs of the existing
            % nodeprotos to updated QDQ nodes and dequantizelinear
            % weights/bias nodes.
            % numExistingNodeProtos  = length(nodeProtos);
            for qLayer = 1:length(qdqTable)
                checkEntity      = qdqTable(qLayer).EntityType;
                scaleValue       = qdqTable(qLayer).Exponent;
                qLayerNum        = matches(allLayerNames, qdqTable(qLayer).DLT_LayerName);
                qLayerAnalyzer   = networkAnalysis.LayerAnalyzers(qLayerNum);
                switch checkEntity
                    case 'Activations'
                        qLayerConverter = ConverterForQuantizedActivationsEntity(qLayerAnalyzer, scaleValue, this.Metadata.OpsetVersion);
                        [nodeProtos, parameterInitializers, qTensorNameMap] = toOnnx(qLayerConverter,...
                            nodeProtos, parameterInitializers, TensorNameMap, qTensorNameMap);
                    case {'Weights', 'Bias'}
                        hasQuantizedLearnableParam = matches(qDetails.QuantizedLearnables.Layer, qdqTable(qLayer).DLT_LayerName)...
                            & matches(qDetails.QuantizedLearnables.Parameter, checkEntity);
                        if any(hasQuantizedLearnableParam)
                            entityDatatype = class(qDetails.QuantizedLearnables.Value{hasQuantizedLearnableParam});
                            entityDatatype = strrep(entityDatatype, 'single', 'float');
                        else
                            entityDatatype = '';
                        end
                        qLayerConverter  = ConverterForQuantizedWeightsBiasEntity(qLayerAnalyzer, ...
                            checkEntity, scaleValue, entityDatatype, this.Metadata.OpsetVersion);
                        [nodeProtos, parameterInitializers, qTensorNameMap] = toOnnx(qLayerConverter,...
                            nodeProtos, parameterInitializers, TensorNameMap, qTensorNameMap);
                    case 'Parameter'
                        % no operation.
                    otherwise
                        error(message("nnet_cnn_onnx:onnx:UnsupportedEntityType", checkEntity));
                end
            end

            % This changes the input-output connections of the existing
            % nodeprotos (i.e. the nodeprotos without the QDQ nodes).
            nodeProtos = updateInputOutputConnections(this, nodeProtos, allExistingNodeProtoNames, qTensorNameMap);

            % Update network for dangling layers.
            % If the layer is dangling, obtain its QDQ nodes output and
            % update it "networkOutputs" ValueInfoProto variable.
            for numOutputLayer = 1:length(networkOutputs)
                while (isKey(qTensorNameMap, networkOutputs(numOutputLayer).name))
                    outputProtoName = qTensorNameMap(networkOutputs(numOutputLayer).name);
                    networkOutputs(numOutputLayer).name = outputProtoName.dqnode;
                end
            end
        end

        function nodeProtos = updateInputOutputConnections(~, nodeProtos, allExistingNodeProtoNames, qTensorNameMap)
            for numNodeProto = 1:length(nodeProtos)
                if any(strcmpi(nodeProtos(numNodeProto).name, allExistingNodeProtoNames))
                    for numInput = 1:numel(nodeProtos(numNodeProto).input)
                        while (isKey(qTensorNameMap, nodeProtos(numNodeProto).input{numInput}) && isstruct(qTensorNameMap(nodeProtos(numNodeProto).input{numInput})))
                            valueNames = qTensorNameMap(nodeProtos(numNodeProto).input{numInput});
                            nodeProtos(numNodeProto).input{numInput} = valueNames.dqnode;
                        end
                    end
                end
            end
        end
    end
end
function tf = MathWorksOperatorsUsed(modelProto)
nodes = modelProto.graph.node;
tf = any(arrayfun(@(n)isequal(lower(n.domain), 'com.mathworks'), nodes));
end

function tf = isBatchnormFused(layerName, qInfoComposite)
% this checks if the batch-norm is fused with conv/grouped-conv layer or
% not. This is checked by "getValueConfig("Layer").Fusion" value and it
% returns True if the BN layer is fused else it returns false.
tf = false;
if ~isempty(qInfoComposite(layerName).getValueConfig("Layer").Fusion)
    tf = qInfoComposite(layerName).getValueConfig("Layer").Fusion.PassThrough;
end
end