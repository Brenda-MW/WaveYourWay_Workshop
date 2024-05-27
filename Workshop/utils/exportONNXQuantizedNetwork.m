function exportONNXQuantizedNetwork(qNet, filename)

% check if network is quantized
qDetails = quantizationDetails(qNet); 

if ~qDetails.IsQuantized
    % pass through
    exportONNXNetwork(qNet, filename, opsetVersion = 10);
    return;
end

% handle quantized network
handleLocalQuantizedNetwork(qNet, filename);
