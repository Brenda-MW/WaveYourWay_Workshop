classdef ConverterForQuantizedWeightsBiasEntity < nnet.internal.cnn.onnx.NNTLayerConverter
    % class to export the quantized learnable parameters based layers to ONNX.

    %   Copyright 2024 The MathWorks, Inc.

    properties
        CheckEntity {mustBeMember(CheckEntity, {'Weights', 'Bias'})} = 'Weights'
        ExponentValue  (1,1) {mustBeNumeric}
        EntityDatatype {mustBeText} = ''
        IsFullyConnectedLayer {mustBeNumericOrLogical}
    end

    methods
        function this = ConverterForQuantizedWeightsBiasEntity(layerAnalyzer,...
                checkEntity, exponentValue, entityDatatype, opsetVersion)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.IsRecurrentNetwork    = layerAnalyzer.IsRNNLayer; % may require in future
            this.IsFullyConnectedLayer = layerAnalyzer.IsFullyConnectedLayer;
            this.OpsetVersion   = opsetVersion;  % may require in future to update the opset
            this.CheckEntity    = checkEntity;
            this.ExponentValue  = exponentValue;
            this.EntityDatatype = entityDatatype;
        end

        function [nodeProtos, parameterInitializers, qTensorNameMap] = toOnnx(this, nodeProtos,...
                parameterInitializers, TensorNameMap, qTensorNameMap)
            import nnet.internal.cnn.onnx.*
            inputTensorName = mapTensorNames(this, cellstr(this.NNTLayer.Name), TensorNameMap);
            inputLayerName  = inputTensorName{1};

            if ~isempty(this.EntityDatatype)
                % extract the required tensor where quantized
                % weights/bias is stored.
                [qtensor, parameterInitializers] = extractTensorProtobyOnnxName(this,...
                    parameterInitializers, inputLayerName);
                % create the dequantize-linear node for weights/bias
                % containing quantized values.
                [dQLinearNodeProto, paramInitializer, qTensorNameMap] = createDqLinearOnnxNode(this,...
                    nodeProtos, inputLayerName, qTensorNameMap, qtensor);
                % append nodeprotos and tensorprotos.
                allNodeProtosNames    = {nodeProtos.name};
                % to add nodes in topological order
                if this.IsFullyConnectedLayer && strcmpi(this.CheckEntity, 'Weights')
                    inputLayerMatMulName = strrep(inputLayerName, '_Add', '_MatMul');
                    layerNodeProtoIdx    = find(strcmpi(inputLayerMatMulName, allNodeProtosNames));
                else
                    layerNodeProtoIdx    = find(strcmpi(inputLayerName, allNodeProtosNames));
                end
                nodeProtos            = [nodeProtos(1:layerNodeProtoIdx-1), dQLinearNodeProto,...
                    nodeProtos(layerNodeProtoIdx:end)];
                parameterInitializers = [parameterInitializers, paramInitializer];
            end
        end
    end

    methods(Access=protected)
        function [dQLinearNodeProto, parameterInitializer, qTensorNameMap] = createDqLinearOnnxNode(this,...
                nodeProtos, inputLayerName, qTensorNameMap, qTensor)
            import nnet.internal.cnn.onnx.*
            % Make the nodeProto for Weights, Scale and Zero-point for
            % DequantizeLinear Weights/bias node
            [onnxName, ~]             = legalizeNNTName(this, [inputLayerName, '_DequantizeLinear']);
            onnxNameParamName         = [onnxName, this.CheckEntity];
            if ismember(onnxNameParamName, {nodeProtos.name})
                onnxNameParamName     = makeUniqueName({nodeProtos.name}, onnxNameParamName);
            end
            paramName                 = [onnxName, this.CheckEntity(1)]; % first word of checkEntity W/B
            paramScaleName            = [onnxName, '_', this.CheckEntity(1), 'ScaleValue'];
            paramZeroPointName        = [onnxName, '_', this.CheckEntity(1), 'ZeroPoint'];
            dQLinearNodeProto         = NodeProto;
            dQLinearNodeProto.op_type = 'DequantizeLinear';
            dQLinearNodeProto.name    = onnxNameParamName;
            dQLinearNodeProto.input   = {paramName, paramScaleName, paramZeroPointName};
            dQLinearNodeProto.output  = cellstr([onnxNameParamName, '_Output']);
            dQLinearNodeProto.doc_string  = ['Quantized ', this.CheckEntity, ' Entity'];

            % extract raw_data from tensor
            parsedData     = typecast(qTensor.raw_data, 'single');
            if ~strcmpi(this.EntityDatatype, 'float')
                parsedData = cast(parsedData, this.EntityDatatype);
            end
            % Make parameter Initializers for weights/bias
            t1           = TensorProto;
            t1.name      = paramName;
            t1.data_type = TensorProto_DataType.(sprintf(upper(this.EntityDatatype)));
            t1.raw_data  = rawData(parsedData);
            t1.dims      = qTensor.dims;

            % Make parameter Initializers for Scale
            t2           = TensorProto;
            t2.name      = paramScaleName;
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data  = rawData(2^(single(this.ExponentValue)));
            t2.dims      = []; % scalar

            % Make parameter Initializers for Zero Point
            t3             = TensorProto;
            t3.name        = paramZeroPointName;
            t3.data_type   = TensorProto_DataType.(sprintf(upper(this.EntityDatatype)));
            zeroPointValue = cast(0, class(parsedData));
            t3.raw_data    = rawData(zeroPointValue);
            t3.dims        = []; % scalar

            % append tensorprotos
            parameterInitializer = [t1, t2, t3];
            % obtain the nodeproto index using nodeproto layername to
            % access the nodeproto inputs
            nodeProtoIdx  = matches({nodeProtos.name}, inputLayerName);
            % For Fully connected layer, nodeproto layername is appended
            % with _MatMul for Weights entity
            if this.IsFullyConnectedLayer && strcmpi(this.CheckEntity(1), 'W')
                inputLayerName = strrep(inputLayerName, '_Add', '_MatMul');
                nodeProtoIdx   = matches({nodeProtos.name}, inputLayerName);
            end
            % Access the nodeproto inputs.
            paramInputIdx = matches(nodeProtos(nodeProtoIdx).input, [inputLayerName, '_', this.CheckEntity(1)]);
            qTensorNameMap(nodeProtos(nodeProtoIdx).input{paramInputIdx}) = struct('dqnode', dQLinearNodeProto.output{1});
        end

        function [tensors, parameterInitializers] = extractTensorProtobyOnnxName(this, parameterInitializers, onnxName)
            idx = matches({parameterInitializers.name}, [onnxName, '_', this.CheckEntity(1)]);
            if ~any(idx) % this means the node belongs to FC layer with Matrix multiplication part
                matMulNodeName = strrep(onnxName, '_Add', '_MatMul');
                idx            = matches({parameterInitializers.name}, [matMulNodeName, '_', this.CheckEntity(1)]);
            end
            % extract tensor to retrieve to quantized learnable parameters
            % (weights/bias) information such as raw_data, dims etc.
            tensors = parameterInitializers(idx);
            % remove the tensor from tensorprotos to save memory.
            parameterInitializers(idx) = [];
        end
    end
end