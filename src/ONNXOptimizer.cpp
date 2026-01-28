// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "ONNXOptimizer.h"

#define ONNX_ML 1
#define ONNX_NAMESPACE onnx
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <string>

namespace lacemodelica {

namespace {

// Find node that produces a given output tensor
const onnx::NodeProto* findNodeByOutput(const onnx::GraphProto& graph,
                                         const std::string& outputName) {
    for (const auto& node : graph.node()) {
        for (const auto& output : node.output()) {
            if (output == outputName) {
                return &node;
            }
        }
    }
    return nullptr;
}

// Check if a tensor name is a graph input
bool isGraphInput(const onnx::GraphProto& graph, const std::string& name) {
    for (const auto& input : graph.input()) {
        if (input.name() == name) {
            return true;
        }
    }
    return false;
}

// Get the body graph attribute from a Loop node
const onnx::GraphProto* getLoopBody(const onnx::NodeProto& loopNode) {
    for (const auto& attr : loopNode.attribute()) {
        if (attr.name() == "body" && attr.has_g()) {
            return &attr.g();
        }
    }
    return nullptr;
}

// Check if body contains nested Loop or Scan nodes
bool hasNestedSubgraphs(const onnx::GraphProto& bodyGraph) {
    for (const auto& node : bodyGraph.node()) {
        if (node.op_type() == "Loop" || node.op_type() == "Scan") {
            return true;
        }
    }
    return false;
}

// Check if an index is derived from loop variable with simple pattern:
// Sub(Add(loopVar, 1), 1) for 1-based to 0-based conversion
bool isSimpleLoopIndex(const onnx::GraphProto& bodyGraph,
                       const std::string& indexName,
                       const std::string& loopVarName) {
    if (indexName == loopVarName) {
        return true;
    }

    const auto* node = findNodeByOutput(bodyGraph, indexName);
    if (!node) {
        return false;
    }

    // Pattern: Sub(something, 1) where something involves loop var
    if (node->op_type() == "Sub" && node->input_size() >= 2) {
        const std::string& minuend = node->input(0);

        // Check if minuend is Add(loopVar + offset, ...) or similar
        const auto* addNode = findNodeByOutput(bodyGraph, minuend);
        if (addNode && addNode->op_type() == "Add" && addNode->input_size() >= 2) {
            // Check if one input is the loop variable
            for (const auto& inp : addNode->input()) {
                if (inp == loopVarName) {
                    return true;
                }
            }
        }

        // Direct Sub from loop var
        if (minuend == loopVarName) {
            return true;
        }
    }

    // Pattern: Add(loopVar, offset)
    if (node->op_type() == "Add" && node->input_size() >= 2) {
        for (const auto& inp : node->input()) {
            if (inp == loopVarName) {
                return true;
            }
        }
    }

    return false;
}

// Analyze which arrays are gathered using the loop variable
// Returns map: array_name -> list of Gather output tensor names
std::map<std::string, std::vector<std::string>> findGatheredArrays(
    const onnx::GraphProto& bodyGraph,
    const std::string& loopVarName) {

    std::map<std::string, std::vector<std::string>> result;

    for (const auto& node : bodyGraph.node()) {
        if (node.op_type() == "Gather" && node.input_size() >= 2) {
            const std::string& arrayName = node.input(0);
            const std::string& indexName = node.input(1);

            if (isSimpleLoopIndex(bodyGraph, indexName, loopVarName)) {
                result[arrayName].push_back(node.output(0));
            }
        }
    }

    return result;
}

// Get names of scan outputs from body (outputs that contain "scan" in name)
std::vector<std::string> findScanOutputs(const onnx::GraphProto& bodyGraph) {
    std::vector<std::string> scanOutputs;
    for (const auto& output : bodyGraph.output()) {
        const std::string& name = output.name();
        if (name.find("scan") != std::string::npos) {
            scanOutputs.push_back(name);
        }
    }
    return scanOutputs;
}

// Collect all nodes that should be removed when converting to Scan:
// - Gather nodes that index arrays with loop variable
// - Index computation nodes (Sub, Add, Constant used for indexing)
// - Identity pass-through nodes for loop-carried deps
// - Condition pass-through
std::set<std::string> findNodesToRemove(
    const onnx::GraphProto& bodyGraph,
    const std::string& loopVarName,
    const std::map<std::string, std::vector<std::string>>& gatheredArrays) {

    std::set<std::string> toRemove;

    for (const auto& node : bodyGraph.node()) {
        // Remove Gather nodes for scanned arrays
        if (node.op_type() == "Gather" && node.input_size() >= 2) {
            const std::string& arrayName = node.input(0);
            if (gatheredArrays.count(arrayName)) {
                toRemove.insert(node.name());

                // Also remove index computation chain
                const std::string& indexName = node.input(1);
                const auto* indexNode = findNodeByOutput(bodyGraph, indexName);
                if (indexNode) {
                    toRemove.insert(indexNode->name());
                    // Go one more level for Sub(Add(...), const) pattern
                    for (const auto& inp : indexNode->input()) {
                        const auto* inputNode = findNodeByOutput(bodyGraph, inp);
                        if (inputNode) {
                            toRemove.insert(inputNode->name());
                        }
                    }
                }
            }
        }

        // Remove Identity pass-throughs
        if (node.op_type() == "Identity") {
            if (node.name().find("_pass") != std::string::npos ||
                node.name().find("cond") != std::string::npos) {
                toRemove.insert(node.name());
            }
        }

        // Remove constants used for index computation
        if (node.op_type() == "Constant") {
            const std::string& name = node.name();
            if (name.find("one") != std::string::npos ||
                name.find("idx") != std::string::npos ||
                name.find("0based") != std::string::npos) {
                toRemove.insert(node.name());
            }
        }
    }

    return toRemove;
}

} // anonymous namespace

int ONNXOptimizer::optimize(onnx::ModelProto& model) {
    return optimizeGraph(*model.mutable_graph());
}

int ONNXOptimizer::optimizeGraph(onnx::GraphProto& graph) {
    int transformations = 0;

    // First, recursively optimize subgraphs in all nodes
    for (int i = 0; i < graph.node_size(); i++) {
        auto* node = graph.mutable_node(i);
        for (int j = 0; j < node->attribute_size(); j++) {
            auto* attr = node->mutable_attribute(j);
            if (attr->has_g()) {
                transformations += optimizeGraph(*attr->mutable_g());
            }
            for (int k = 0; k < attr->graphs_size(); k++) {
                transformations += optimizeGraph(*attr->mutable_graphs(k));
            }
        }
    }

    // Then attempt Loop->Scan on this graph's nodes
    // We need to iterate carefully since we modify the graph
    for (int i = 0; i < graph.node_size(); i++) {
        const auto& node = graph.node(i);
        if (node.op_type() == "Loop") {
            if (loopToScan(node, graph)) {
                transformations++;
                // Node was replaced, adjust index to re-check this position
                i--;
            }
        }
    }

    return transformations;
}

bool ONNXOptimizer::loopToScan(const onnx::NodeProto& loopNode, onnx::GraphProto& graph) {
    // Get body graph
    const onnx::GraphProto* bodyGraph = getLoopBody(loopNode);
    if (!bodyGraph) {
        return false;
    }

    // Check for nested loops - don't convert
    if (hasNestedSubgraphs(*bodyGraph)) {
        return false;
    }

    // Body must have at least iter_num, cond inputs
    if (bodyGraph->input_size() < 2) {
        return false;
    }

    const std::string loopVarName = bodyGraph->input(0).name();

    // Find arrays gathered using loop variable
    auto gatheredArrays = findGatheredArrays(*bodyGraph, loopVarName);
    if (gatheredArrays.empty()) {
        return false;
    }

    // Find scan outputs
    auto scanOutputNames = findScanOutputs(*bodyGraph);
    if (scanOutputNames.empty()) {
        return false;
    }

    // Verify gathered arrays are body inputs (not computed inside)
    std::set<std::string> bodyInputNames;
    for (const auto& input : bodyGraph->input()) {
        bodyInputNames.insert(input.name());
    }

    for (const auto& [arrayName, _] : gatheredArrays) {
        if (bodyInputNames.find(arrayName) == bodyInputNames.end()) {
            return false;  // Array is computed inside loop, can't convert
        }
    }

    // Build the new Scan node
    onnx::NodeProto scanNode;
    scanNode.set_op_type("Scan");
    scanNode.set_name(loopNode.name() + "_as_scan");

    // Scan inputs are the arrays being scanned
    std::vector<std::string> scanInputArrays;
    for (const auto& [arrayName, _] : gatheredArrays) {
        scanInputArrays.push_back(arrayName);
    }

    // Map body input names to loop input tensors
    // Loop inputs: [trip_count, cond, loop_carried_deps...]
    // Body inputs: [iter_num, cond, loop_carried_deps...]
    std::map<std::string, std::string> bodyInputToLoopInput;
    for (int i = 2; i < bodyGraph->input_size() && i < loopNode.input_size(); i++) {
        bodyInputToLoopInput[bodyGraph->input(i).name()] = loopNode.input(i);
    }

    // Add scan inputs to node
    for (const auto& arrayName : scanInputArrays) {
        auto it = bodyInputToLoopInput.find(arrayName);
        if (it != bodyInputToLoopInput.end()) {
            scanNode.add_input(it->second);
        }
    }

    // Scan outputs correspond to loop scan outputs
    // Loop outputs: [loop_carried_deps..., scan_outputs...]
    for (const auto& loopOutput : loopNode.output()) {
        if (loopOutput.find("eq[") != std::string::npos) {
            scanNode.add_output(loopOutput);
        }
    }

    // If no eq[] outputs found, use the last N outputs
    if (scanNode.output_size() == 0) {
        int numScanOutputs = scanOutputNames.size();
        for (int i = loopNode.output_size() - numScanOutputs; i < loopNode.output_size(); i++) {
            if (i >= 0) {
                scanNode.add_output(loopNode.output(i));
            }
        }
    }

    // Create new body graph for Scan
    auto* bodyAttr = scanNode.add_attribute();
    bodyAttr->set_name("body");
    bodyAttr->set_type(onnx::AttributeProto::GRAPH);
    auto* newBody = bodyAttr->mutable_g();
    newBody->set_name(loopNode.name() + "_scan_body");

    // Map from Gather outputs to element input names
    std::map<std::string, std::string> gatherToElement;

    // Add element inputs for each scanned array
    for (const auto& arrayName : scanInputArrays) {
        std::string elemName = arrayName + "_elem";

        auto* elemInput = newBody->add_input();
        elemInput->set_name(elemName);
        auto* tensorType = elemInput->mutable_type()->mutable_tensor_type();
        tensorType->set_elem_type(onnx::TensorProto::DOUBLE);
        // Scalar element - no shape dimensions

        // Map all Gather outputs from this array to the element input
        for (const auto& gatherOutput : gatheredArrays[arrayName]) {
            gatherToElement[gatherOutput] = elemName;
        }
    }

    // Copy initializers from original body to new body
    for (const auto& init : bodyGraph->initializer()) {
        auto* newInit = newBody->add_initializer();
        newInit->CopyFrom(init);
    }

    // Find nodes to remove
    auto nodesToRemove = findNodesToRemove(*bodyGraph, loopVarName, gatheredArrays);

    // Copy nodes, renaming inputs as needed
    for (const auto& node : bodyGraph->node()) {
        if (nodesToRemove.count(node.name())) {
            continue;
        }

        auto* newNode = newBody->add_node();
        newNode->CopyFrom(node);

        // Rename inputs that were Gather outputs
        for (int i = 0; i < newNode->input_size(); i++) {
            auto it = gatherToElement.find(newNode->input(i));
            if (it != gatherToElement.end()) {
                newNode->set_input(i, it->second);
            }
        }
    }

    // Add scan outputs to body
    for (const auto& outputName : scanOutputNames) {
        // Find the original output info
        for (const auto& output : bodyGraph->output()) {
            if (output.name() == outputName) {
                auto* newOutput = newBody->add_output();
                newOutput->CopyFrom(output);
                break;
            }
        }
    }

    // Set Scan attributes
    auto* numScanInputsAttr = scanNode.add_attribute();
    numScanInputsAttr->set_name("num_scan_inputs");
    numScanInputsAttr->set_type(onnx::AttributeProto::INT);
    numScanInputsAttr->set_i(scanInputArrays.size());

    // scan_input_axes: all axis 0
    auto* scanInputAxesAttr = scanNode.add_attribute();
    scanInputAxesAttr->set_name("scan_input_axes");
    scanInputAxesAttr->set_type(onnx::AttributeProto::INTS);
    for (size_t i = 0; i < scanInputArrays.size(); i++) {
        scanInputAxesAttr->add_ints(0);
    }

    // scan_output_axes: all axis 0
    auto* scanOutputAxesAttr = scanNode.add_attribute();
    scanOutputAxesAttr->set_name("scan_output_axes");
    scanOutputAxesAttr->set_type(onnx::AttributeProto::INTS);
    for (int i = 0; i < scanNode.output_size(); i++) {
        scanOutputAxesAttr->add_ints(0);
    }

    // Track Loop's trip_count and cond inputs for cleanup
    std::set<std::string> loopOnlyInputs;
    if (loopNode.input_size() >= 2) {
        loopOnlyInputs.insert(loopNode.input(0));  // trip_count
        loopOnlyInputs.insert(loopNode.input(1));  // cond
    }

    // Find and remove the original Loop node, add Scan node
    for (int i = 0; i < graph.node_size(); i++) {
        if (graph.node(i).name() == loopNode.name()) {
            graph.mutable_node()->DeleteSubrange(i, 1);
            break;
        }
    }

    auto* addedNode = graph.add_node();
    addedNode->CopyFrom(scanNode);

    // Clean up orphaned nodes (trip_count and cond constants)
    // Check if any node still uses these tensors
    std::set<std::string> usedInputs;
    for (const auto& node : graph.node()) {
        for (const auto& input : node.input()) {
            usedInputs.insert(input);
        }
    }

    // Remove nodes that produce orphaned tensors
    for (const auto& orphan : loopOnlyInputs) {
        if (usedInputs.find(orphan) == usedInputs.end()) {
            // Find and remove the producer node
            for (int i = 0; i < graph.node_size(); i++) {
                bool found = false;
                for (const auto& output : graph.node(i).output()) {
                    if (output == orphan) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    graph.mutable_node()->DeleteSubrange(i, 1);
                    break;
                }
            }
        }
    }

    return true;
}

} // namespace lacemodelica
