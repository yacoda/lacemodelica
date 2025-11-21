// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Joris Gillis, YACODA

#include "FMUGenerator.h"
#include "tinyxml2.h"
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace tinyxml2;

namespace lacemodelica {

bool FMUGenerator::generateFMU(const ModelInfo& info, const std::string& outputPath) {
    // Create FMU directory structure
    fs::create_directories(outputPath);

    // Generate modelDescription.xml
    std::string xmlPath = outputPath + "/modelDescription.xml";
    return generateModelDescription(info, xmlPath);
}

bool FMUGenerator::generateModelDescription(const ModelInfo& info, const std::string& xmlPath) {
    XMLDocument doc;

    // XML declaration
    XMLDeclaration* decl = doc.NewDeclaration();
    doc.InsertFirstChild(decl);

    // Root element
    XMLElement* root = doc.NewElement("fmiModelDescription");
    root->SetAttribute("fmiVersion", "3.0");
    root->SetAttribute("modelName", info.modelName.c_str());
    root->SetAttribute("instantiationToken", generateGUID().c_str());
    root->SetAttribute("description", info.description.c_str());
    root->SetAttribute("generationTool", "lacemodelica 0.1.0");
    root->SetAttribute("generationDateAndTime", "2025-01-01T00:00:00Z");
    root->SetAttribute("variableNamingConvention", "flat");
    doc.InsertEndChild(root);

    // ModelExchange element
    XMLElement* modelExchange = doc.NewElement("ModelExchange");
    modelExchange->SetAttribute("modelIdentifier", info.modelName.c_str());
    root->InsertEndChild(modelExchange);

    // ModelVariables
    XMLElement* modelVariables = doc.NewElement("ModelVariables");

    for (const auto& var : info.variables) {
        XMLElement* varElem = nullptr;

        // Create appropriate type element (types are never quoted)
        if (var.type == "Real") {
            varElem = doc.NewElement("Float64");
        } else if (var.type == "Integer") {
            varElem = doc.NewElement("Int32");
        } else if (var.type == "Boolean") {
            varElem = doc.NewElement("Boolean");
        } else {
            varElem = doc.NewElement("Float64");  // Default
        }

        varElem->SetAttribute("name", var.name.c_str());
        varElem->SetAttribute("valueReference", var.valueReference);

        if (!var.description.empty()) {
            varElem->SetAttribute("description", var.description.c_str());
        }

        varElem->SetAttribute("causality", var.causality.c_str());
        varElem->SetAttribute("variability", var.variability.c_str());

        if (!var.initial.empty()) {
            varElem->SetAttribute("initial", var.initial.c_str());
        }

        if (!var.startValue.empty()) {
            varElem->SetAttribute("start", var.startValue.c_str());
        }

        if (var.isDerivative && var.derivativeOf >= 0) {
            varElem->SetAttribute("derivative", var.derivativeOf);
        }

        // Handle array dimensions as child elements (FMI 3.0)
        if (!var.dimensions.empty()) {
            for (const auto& dim : var.dimensions) {
                XMLElement* dimElem = doc.NewElement("Dimension");
                // If dimension is a number, use start attribute
                // If dimension is a variable name, look up its valueReference
                if (std::isdigit(dim[0])) {
                    dimElem->SetAttribute("start", dim.c_str());
                } else {
                    // It's a structural parameter reference - find its valueReference
                    auto* dimVar = info.findVariable(dim);
                    if (dimVar) {
                        dimElem->SetAttribute("valueReference", dimVar->valueReference);
                    } else {
                        // Fallback if not found - use start with the value
                        dimElem->SetAttribute("start", dim.c_str());
                    }
                }
                varElem->InsertEndChild(dimElem);
            }
        }

        modelVariables->InsertEndChild(varElem);
    }

    root->InsertEndChild(modelVariables);

    // ModelStructure
    XMLElement* modelStructure = doc.NewElement("ModelStructure");

    // Add outputs
    for (const auto& var : info.getOutputs()) {
        XMLElement* output = doc.NewElement("Output");
        output->SetAttribute("valueReference", var.valueReference);
        modelStructure->InsertEndChild(output);
    }

    // Add continuous state derivatives
    for (const auto& var : info.getDerivatives()) {
        XMLElement* derivative = doc.NewElement("ContinuousStateDerivative");
        derivative->SetAttribute("valueReference", var.valueReference);
        modelStructure->InsertEndChild(derivative);
    }

    root->InsertEndChild(modelStructure);

    // Save to file
    XMLError result = doc.SaveFile(xmlPath.c_str());
    if (result != XML_SUCCESS) {
        std::cerr << "Error saving XML file: " << xmlPath << std::endl;
        return false;
    }

    std::cout << "Generated: " << xmlPath << std::endl;
    return true;
}

std::string FMUGenerator::generateGUID() {
    // Generate a simple GUID-like string
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    ss << "{";
    for (int i = 0; i < 32; i++) {
        if (i == 8 || i == 12 || i == 16 || i == 20) {
            ss << "-";
        }
        ss << std::hex << dis(gen);
    }
    ss << "}";

    return ss.str();
}

} // namespace lacemodelica
