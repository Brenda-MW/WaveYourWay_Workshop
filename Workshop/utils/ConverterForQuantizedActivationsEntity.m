classdef ConverterForQuantizedActivationsEntity < nnet.internal.cnn.onnx.NNTLayerConverter
    % class to export the quantized activation entity to ONNX.

    %   Copyright 2024 The MathWorks, Inc.

    properties
        ExponentValue (1,1) {mustBeNumeric}
        IsFullyConnectedLayer
    end

    methods
        function this = ConverterForQuantizedActivationsEntity(layerAnalyzer, exponentValue, opsetVersion)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.IsRecurrentNetwork    = layerAnalyzer.IsRNNLayer;
            this.IsFullyConnectedLayer = layerAnalyzer.IsFullyConnectedLayer;
            this.ExponentValue         = exponentValue;
            this.OpsetVersion          = opsetVersion;
        end

        function [nodeProtos, parameterInitializers, qTensorNameMap] = toOnnx(this, nodeProtos,...
                parameterInitializers, TensorNameMap, qTensorNameMap)
            import nnet.internal.cnn.onnx.*
            inputLayerName = mapTensorNames(this, cellstr(this.NNTLayer.Name), TensorNameMap);

            % If fully-connected (FC) or recurrent neural network (RNN) layer 
            % is present in the DL network, a flatten layer is added in MATLAB.
            % In ConverterForFlattenLayer class, the toOnnx method of that class adds the 
            % transpose node followed by the reshape node. As this happen in 
            % runtime while exporting the network to ONNX, the exponent data 
            % information is not available for flatten layer in MATLAB. To 
            % mitigate this issue,the method "addQDQNodesForFlattenAndTansposeLayer" 
            % adds the quantize linear and dequantize linear (QDQ) nodes after 
            % transpose and reshape node by taking previous activation 
            % entity's exponent value. 

            % Note: As RNN layers are not yet supported for quantization as 
            % of R2024b, the conditional check only validates if the layer
            % is FC. RNN layer condition will be added in Future. 
            % Use this.IsRecurrentNetwork property in future in the
            % following condition.
            if this.IsFullyConnectedLayer % || this.IsRecurrentNetwork
                isQDQForReshapeTranspose = any(strcmpi('QDQ added for transpose and reshape', {nodeProtos.doc_string}));
                if ~isQDQForReshapeTranspose
                    [qDqNodesForFcLayer, tensorProtosForFcLayer, qTensorNameMap] = addQDQNodesForFlattenAndTansposeLayer(this, nodeProtos,...
                        parameterInitializers, TensorNameMap, qTensorNameMap);
                    nodeProtos            = [nodeProtos, qDqNodesForFcLayer];
                    parameterInitializers = [parameterInitializers, tensorProtosForFcLayer];
                end
            elseif isa(this.NNTLayer, 'nnet.cnn.layer.Convolution1DLayer') ||...
                        isa(this.NNTLayer, 'nnet.cnn.layer.Convolution2DLayer') ||...
                        isa(this.NNTLayer, 'nnet.cnn.layer.Convolution3DLayer')
                if (isequal(this.NNTLayer.PaddingMode, 'same') && this.OpsetVersion > 10) ||...
                        (this.NNTLayer.PaddingValue~=0)
                    [qDqNodesForPadLayer, tensorProtosForPadLayer, qTensorNameMap] = addQDQNodesForPadLayer(this, nodeProtos,...
                        parameterInitializers, TensorNameMap, qTensorNameMap);
                    nodeProtos            = [nodeProtos, qDqNodesForPadLayer];
                    parameterInitializers = [parameterInitializers, tensorProtosForPadLayer];
                end
            end
            % obtain all nodeprotos names to avoid creating duplicate names 
            % in quantize linear and dequantize linear nodes.
            allNodeProtosNames    = {nodeProtos.name};
            % Create 
            [qDqnodeProto, paramInitializer, qTensorNameMap] = createQDQOnnxWrapperNodes(this, allNodeProtosNames, inputLayerName{1}, qTensorNameMap);
            nodeProtos            = [nodeProtos, qDqnodeProto];
            parameterInitializers = [parameterInitializers, paramInitializer];
        end
    end

    methods(Access=protected)
        function [qDqnodeProto, parameterInitializer, qTensorNameMap] = createQDQOnnxWrapperNodes(this, allNodeProtosNames, inputLayerName, qTensorNameMap)
            nextOperatorInput   = inputLayerName;

            % Make the nodeProto for Scale and Zero-point for
            % QuantizeLinear node
            opType = 'QuantizeLinear';
            [quantizeLinearNode, qTensor, nextOperatorInput] = createNodeForQuantization(this, inputLayerName, nextOperatorInput, opType, allNodeProtosNames);

            % Make the nodeProto for Scale and Zero-point for
            % DequantizeLinear Node
            opType = 'DequantizeLinear';
            [dequantizeLinearNode, dQTensor, ~] = createNodeForQuantization(this, inputLayerName, nextOperatorInput, opType, allNodeProtosNames);

            % Update maps
            qTensorNameMap(inputLayerName) = struct('dqnode', dequantizeLinearNode.output{1, 1});
            % for quantization, TensorNameMap is updated via struct i.e.,
            % different from usual TensorNamemap values. This allows to
            % change the input-output connections
            % (method in ConverterFornetwork.UpdateInputOutputConnections)
            % only for the quantization information embedded nodes. To
            % distinguish between the non-quantized nodeprotos and
            % quantized information embedded nodeprotos, different data structure is used.
            parameterInitializer  = [qTensor, dQTensor];
            qDqnodeProto          = [quantizeLinearNode, dequantizeLinearNode];
        end

        function [reqNode, tensor, nextOperatorInput] = createNodeForQuantization(this, inputLayerName, nextOperatorInput, opType, allNodeProtosNames)
            import nnet.internal.cnn.onnx.*
            [onnxName, ~] = legalizeNNTName(this, [inputLayerName, '_', opType]);
            if ismember(onnxName, allNodeProtosNames)
                onnxName  = makeUniqueName(allNodeProtosNames, onnxName);
            end

            switch opType
                case 'QuantizeLinear'
                    scaleValueName  = [onnxName, '_YscaleValue'];
                    zeroPointName   = [onnxName, '_YzeroPoint'];
                case 'DequantizeLinear'
                    scaleValueName  = [onnxName, '_XscaleValue'];
                    zeroPointName   = [onnxName, '_XzeroPoint'];
            end

            reqNode         = NodeProto;
            reqNode.op_type = opType;
            reqNode.name    = onnxName;
            reqNode.input   = {nextOperatorInput, scaleValueName, zeroPointName};
            reqNode.output  = cellstr([onnxName, '_Output']);
            reqNode.doc_string  = 'Activation entity';

            % Make parameter Initializers for Scale
            t1              = TensorProto;
            t1.name         = scaleValueName;
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(2^(single(this.ExponentValue)));
            t1.dims         = []; % scalar
            t1.doc_string   = [inputLayerName, '_', opType, ' tensorproto'];

            % Make parameter Initializers for Zero Point
            t2              = TensorProto;
            t2.name         = zeroPointName;
            t2.data_type    = TensorProto_DataType.INT8;
            zeroPointValue  = cast(0, 'int8');
            t2.raw_data     = rawData(zeroPointValue);
            t2.dims         = []; % scalar

            nextOperatorInput = reqNode.output{1, 1};
            tensor = [t1, t2];
        end

        function [qDqNodesForPadLayer, tensorProtosForPadLayer, qTensorNameMap] = addQDQNodesForPadLayer(this, nodeProtos,...
                        parameterInitializers, TensorNameMap, qTensorNameMap)
            % find the last (QDQ) tensorproto of the input of the current
            % layer. If the QDQ nodes are not present for the input layer,
            % find it using nodeprotos.
            inputTensorName = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            tensorprotoLoc  = strcmpi([inputTensorName{1}, '_QuantizeLinear', ' tensorproto'], {parameterInitializers.doc_string});
            % if the QDQ nodes are not present for the input layer of the
            % current nodeproto. (edge case)
            if ~any(tensorprotoLoc)
                for node = length(nodeProtos):-1:1
                    if strcmpi(nodeProtos(node).doc_string, 'Activation entity')
                        scaleValueName = nodeProtos(node).input{2};
                        break
                    end
                end
                % obtain the tensorproto of the scale value
                tensorprotoLoc = matches({parameterInitializers.name}, scaleValueName);
            end
            
            qDqTensorProto    = parameterInitializers(tensorprotoLoc);
            convNodeprotoName = TensorNameMap(this.NNTLayer.Name);

            % obtain input layer name
            allNodeProtosNames = {nodeProtos.name};
            inputLayerName     = [convNodeprotoName, '_Pad'];
            % add QDQ nodes after Transpose node
            [qDqNodesForPadLayer, tensorProtosForPadLayer, qTensorNameMap] = createQDQOnnxWrapperNodes(this, allNodeProtosNames, inputLayerName, qTensorNameMap);
            % scale value of quantize linear node of the previous activation
            % node.
            tensorProtosForPadLayer(1).raw_data = qDqTensorProto.raw_data;
            % scale value of dequantize linear node of the previous activation
            % node.
            tensorProtosForPadLayer(3).raw_data = qDqTensorProto.raw_data;
        end

        function [qDqNodesForFcLayer, tensorProtosForFcLayer, qTensorNameMap] = addQDQNodesForFlattenAndTansposeLayer(this, nodeProtos,...
                parameterInitializers, TensorNameMap, qTensorNameMap)
            % find the previous node QDQ nodeproto
            for node = length(nodeProtos):-1:1
                if strcmpi(nodeProtos(node).doc_string, 'Activation entity')
                    scaleValueName = nodeProtos(node).input{2};
                    break
                end
            end
            % obtain the tensorproto of the scale value
            tensorprotoLoc = matches({parameterInitializers.name}, scaleValueName);
            
            qDqTensorProto = parameterInitializers(tensorprotoLoc);

            % obtain input layer name (transpose node for FC layer)
            allNodeProtosNames = {nodeProtos.name};
            inputLayerName     = [this.InputLayerNames{1, 1}, '_Transpose'];
            inputLayerName     = legalizeNNTName(this, inputLayerName);
            % add QDQ nodes after Transpose node
            [qDqnodeProtoTransposeNode, paramInitTransposeNode, qTensorNameMap] = createQDQOnnxWrapperNodes(this, allNodeProtosNames, inputLayerName, qTensorNameMap);
            % scale value of quantize linear node of the previous activation
            % node.
            paramInitTransposeNode(1).raw_data = qDqTensorProto.raw_data;
            % scale value of dequantize linear node of the previous activation
            % node.
            paramInitTransposeNode(3).raw_data = qDqTensorProto.raw_data;

            % obtain input layer name (reshape node for FC layer) from TensorNameMap
            inputLayerName     = TensorNameMap(this.InputLayerNames{1, 1});
            % add QDQ nodes after reshape node
            [qDqnodeProtoReshapeNode, paramInitReshapeNode, qTensorNameMap] = createQDQOnnxWrapperNodes(this, allNodeProtosNames, inputLayerName, qTensorNameMap);
            % scale value of quantize linear node of the previous activation
            % node.
            paramInitReshapeNode(1).raw_data = qDqTensorProto.raw_data;
            % scale value of dequantize linear node of the previous activation
            % node.
            paramInitReshapeNode(3).raw_data = qDqTensorProto.raw_data;

            % In quantized DL network, a flatten node
            % is added if the network contains RNN or FC layer (refer the
            % local function "iMakeNetworkSTCompatible" in the internal
            % "exportONNXNetwork" function). Note that the only flatten layer
            % is added even if multiple FC or RNN layers are present and flatten
            % layer is not added for each FC or RNN layer.
            %
            % When the toONNX method of ConverterForFlattenLayer class is
            % called, it adds transpose node followed by reshape node.
            % This doc string text will help to identify if the QDQ nodes
            % are added to the transpose node. This doc_string will be used
            % if multiple FC layer are present and helps to avoid adding
            % QDQ nodes after transpose and reshape node after every FC
            % layer is encountered in Activation entity.

            % Add the doc_string text to the dequantize linear node of
            % the transpose node.
            qDqnodeProtoTransposeNode(2).doc_string = 'QDQ added for transpose and reshape';

            % append the QDQ nodeprotos and tensorprotos of transpose and
            % reshape node.
            qDqNodesForFcLayer     = [qDqnodeProtoTransposeNode, qDqnodeProtoReshapeNode];
            tensorProtosForFcLayer = [paramInitTransposeNode, paramInitReshapeNode];
        end
    end
end