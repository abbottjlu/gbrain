(function(){function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s}return e})()({1:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.KERNEL_DIR = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _graphUtil = require("./graphUtil");

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var KERNEL_DIR = exports.KERNEL_DIR = function () {
    function KERNEL_DIR() {
        _classCallCheck(this, KERNEL_DIR);
    }

    _createClass(KERNEL_DIR, null, [{
        key: "getSrc",
        value: function getSrc(customCode, geometryLength, efferentStart, efferentNodesCount, _enableNeuronalNetwork) {
            var outputArr = null;
            var returnStr = null;
            if (_enableNeuronalNetwork === true) {
                outputArr = ["dir", "posXYZW", "dataB", "dataF", "dataG", "dataH"];
                returnStr = 'return [vec4(currentDir, 1.0), vec4(currentPos.x, currentPos.y, currentPos.z, 1.0), currentDataB, currentDataF, currentDataG, currentDataH];';
            } else {
                outputArr = ["dir", "posXYZW"];
                returnStr = 'return [vec4(currentDir, 1.0), vec4(currentPos.x, currentPos.y, currentPos.z, 1.0)];';
            }

            return ["x", outputArr,
            // head
            _graphUtil.GraphUtils.adjMatrix_ForceLayout_GLSLFunctionString(geometryLength, efferentStart, efferentNodesCount),

            // source
            "float nodeId = data[x].x;\n                    float numOfConnections = data[x].y;\n                    vec2 xGeometry = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n                    vec3 currentPos = posXYZW[xGeometry].xyz;\n\n                    float bornDate = dataB[xGeometry].x;\n                    float dieDate = dataB[xGeometry].y;\n\n                    vec3 currentDir = dir[xGeometry].xyz;\n\n\n                    vec4 currentDataB = dataB[xGeometry];\n                    vec4 currentDataF = dataF[xGeometry];\n                    vec4 currentDataG = dataG[xGeometry];\n                    vec4 currentDataH = dataH[xGeometry];\n\n                    currentDir = vec3(0.0, 0.0, 0.0);\n\n                    if(enableForceLayout == 1.0) {\n                        idAdjMatrixResponse adjM = idAdjMatrix_ForceLayout(nodeId, currentPos, currentDir, numOfConnections, currentTimestamp, bornDate, dieDate, enableNeuronalNetwork);\n                        currentDir = (adjM.collisionExists == 1.0) ? adjM.force : currentDir+(adjM.force*1.0);\n\n                        if(enableNeuronalNetwork == 1.0 && currentTrainLayer == -3.0) {\n                            currentDataB = vec4(currentDataB.x, currentDataB.y, adjM.netFOutputA, adjM.netErrorWeightA);\n                            currentDataF = vec4(adjM.netFOutputB, adjM.netErrorWeightB, adjM.netFOutputC, adjM.netErrorWeightC);\n                            currentDataG = vec4(adjM.netFOutputD, adjM.netErrorWeightD, adjM.netFOutputE, adjM.netErrorWeightE);\n                            currentDataH = vec4(adjM.netFOutputF, adjM.netErrorWeightF, adjM.netFOutputG, adjM.netErrorWeightG);\n                        }\n                    }\n\n                    " + (customCode !== undefined ? customCode : '') + "\n\n                    if(enableDrag == 1.0) {\n                        if(nodeId == idToDrag) {\n                            currentPos = vec3(MouseDragTranslationX, MouseDragTranslationY, MouseDragTranslationZ);\n                        }\n                    }\n\n                    currentPos += currentDir;\n                    if(only2d == 1.0) {\n                        currentPos.y = 0.0;\n                    }\n\n                    " + returnStr];
        }
    }]);

    return KERNEL_DIR;
}();

global.KERNEL_DIR = KERNEL_DIR;
module.exports.KERNEL_DIR = KERNEL_DIR;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./graphUtil":2}],2:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var GraphUtils = exports.GraphUtils = function () {
    function GraphUtils() {
        _classCallCheck(this, GraphUtils);
    }

    _createClass(GraphUtils, null, [{
        key: "nodesDrawMode",
        value: function nodesDrawMode(geometryLength) {
            if (geometryLength === 1) return "vec4(color.rgb, 1.0)";else return "vec4(tex.rgb*color.rgb, tex.a)";
        }
    }, {
        key: "adjMatrix_ForceLayout_GLSLFunctionString",
        value: function adjMatrix_ForceLayout_GLSLFunctionString(geometryLength, efferentStart, efferentNodesCount) {
            return '' + "vec3 sphericalColl(vec3 currentDir, vec3 currentDirB, vec3 dirToBN) {\n            vec3 currentDirN = normalize(currentDir);\n            float pPoint = abs(dot(currentDirN, dirToBN));\n            vec3 reflectV = reflect(currentDirN*-1.0, dirToBN);\n\n            vec3 currentDirBN = normalize(currentDirB);\n            float pPointB = abs(dot(currentDirBN, dirToBN));\n\n            vec3 repulsionForce = (reflectV*-1.0)* (((1.0-pPoint)*length(currentDir))+((pPointB)*length(currentDirB)));\n\n            return (repulsionForce.x > 0.0 && repulsionForce.y > 0.0 && repulsionForce.z > 0.0) ? repulsionForce : dirToBN*-0.1;\n        }\n\n        struct CalculationResponse {\n            vec3 atraction;\n            float acumAtraction;\n            vec3 repulsion;\n            float collisionExists;\n            float netChildInputSumA;\n            float netParentErrorWeightA;\n            float netChildInputSumB;\n            float netParentErrorWeightB;\n            float netChildInputSumC;\n            float netParentErrorWeightC;\n            float netChildInputSumD;\n            float netParentErrorWeightD;\n            float netChildInputSumE;\n            float netParentErrorWeightE;\n            float netChildInputSumF;\n            float netParentErrorWeightF;\n            float netChildInputSumG;\n            float netParentErrorWeightG;\n        };" +

            // pixAdjMatA (bornDate, dieDate, weight (parent:-2;child:w), isParent (1.0:parent;0.0:child))
            // pixAdjMatA (linkMultiplier, activationFunction)
            "CalculationResponse calculate(float nodeId,\n                                        vec4 pixAdjMatACurrent, vec4 pixAdjMatAOpposite,\n                                        vec4 pixAdjMatBCurrent, vec4 pixAdjMatBOpposite,\n                                        vec2 xGeomCurrent, vec2 xGeomOpposite,\n                                        vec3 currentPos, vec3 currentDir,\n                                        vec3 atraction, float acumAtraction, vec3 repulsion,\n                                        float enableNeuronalNetwork,\n                                        float netChildInputSumA, float netParentErrorWeightA,\n                                        float netChildInputSumB, float netParentErrorWeightB,\n                                        float netChildInputSumC, float netParentErrorWeightC,\n                                        float netChildInputSumD, float netParentErrorWeightD,\n                                        float netChildInputSumE, float netParentErrorWeightE,\n                                        float netChildInputSumF, float netParentErrorWeightF,\n                                        float netChildInputSumG, float netParentErrorWeightG) {" +
            // pixAdjMatACurrent
            "float currentBornDate = pixAdjMatACurrent.x;\n            float currentDieDate = pixAdjMatACurrent.y;\n            float currentWeight = pixAdjMatACurrent.z;\n            float currentIsParent = pixAdjMatACurrent.w;" +

            // pixAdjMatAOpposite
            "float oppositeBornDate = pixAdjMatAOpposite.x;\n            float oppositeDieDate = pixAdjMatAOpposite.y;\n            float oppositeWeight = pixAdjMatAOpposite.z;\n            float oppositeIsParent = pixAdjMatAOpposite.w;" +

            // pixAdjMatBCurrent
            "float currentLinkMultiplier = pixAdjMatBCurrent.x;\n            float currentActivationFn = pixAdjMatBCurrent.y;" +

            // pixAdjMatBOpposite
            "float oppositeLinkMultiplier = pixAdjMatBOpposite.x;\n            float oppositeActivationFn = pixAdjMatBOpposite.y;" +

            // dataB Current
            //'float currentBornDate = dataB[xGeomCurrent].x;'+
            //'float currentDieDate = dataB[xGeomCurrent].y;'+
            //'float currentNetOutput = dataB[xGeomCurrent].z;'+
            //'float currentNetError = dataB[xGeomCurrent].w;'+

            // dataB Opposite
            //'float oppositeBornDate = dataB[xGeomOpposite].x;'+
            //'float oppositeDieDate = dataB[xGeomOpposite].y;'+
            "float oppositeNetOutputA = dataB[xGeomOpposite].z;\n            float oppositeNetErrorA = dataB[xGeomOpposite].w;\n\n            float oppositeNetOutputB = dataF[xGeomOpposite].x;\n            float oppositeNetErrorB = dataF[xGeomOpposite].y;\n        \n            float oppositeNetOutputC = dataF[xGeomOpposite].z;\n            float oppositeNetErrorC = dataF[xGeomOpposite].w;\n        \n            float oppositeNetOutputD = dataG[xGeomOpposite].x;\n            float oppositeNetErrorD = dataG[xGeomOpposite].y;\n        \n            float oppositeNetOutputE = dataG[xGeomOpposite].z;\n            float oppositeNetErrorE = dataG[xGeomOpposite].w;\n        \n            float oppositeNetOutputF = dataH[xGeomOpposite].x;\n            float oppositeNetErrorF = dataH[xGeomOpposite].y;\n        \n            float oppositeNetOutputG = dataH[xGeomOpposite].z;\n            float oppositeNetErrorG = dataH[xGeomOpposite].w;" +

            // pos & dir Current
            //'vec3 currentPos = posXYZW[xGeomCurrent].xyz;\n'+
            //'vec3 currentDir = dir[xGeomCurrent].xyz;\n'+

            // pos & dir Opposite
            "vec3 oppositePos = posXYZW[xGeomOpposite].xyz;\n            vec3 oppositeDir = dir[xGeomOpposite].xyz;" +

            // dir / dist to opposite
            'vec3 dirToOpposite = (oppositePos-currentPos);\n' + 'vec3 dirToOppositeN = normalize(dirToOpposite);\n' + 'float dist = distance(oppositePos, currentPos);\n' + // near=0.0 ; far=1.0
            'float distN = max(0.0,dist)/100000.0;' + 'float p = 1.0;' + 'if(currentDieDate != 0.0 && (currentTimestamp < currentBornDate || currentTimestamp > currentDieDate)) ' + 'p = 0.0;' + 'if(oppositeDieDate != 0.0 && (currentTimestamp < oppositeBornDate || currentTimestamp > oppositeDieDate)) ' + 'p = 0.0;' + 'if(p == 1.0) {' + 'float m1 = (enableNeuronalNetwork == 1.0) ? 0.0 : 400000.0;' + 'float m2 = (enableNeuronalNetwork == 1.0) ? 0.0 : 48.0;' + 'if(currentIsParent == 1.0) {' +
            //'if(enableNeuronalNetwork == 1.0) '+
            'netChildInputSumA += oppositeNetOutputA*oppositeWeight;' + 'netChildInputSumB += oppositeNetOutputB*oppositeWeight;' + 'netChildInputSumC += oppositeNetOutputC*oppositeWeight;' + 'netChildInputSumD += oppositeNetOutputD*oppositeWeight;' + 'netChildInputSumE += oppositeNetOutputE*oppositeWeight;' + 'netChildInputSumF += oppositeNetOutputF*oppositeWeight;' + 'netChildInputSumG += oppositeNetOutputG*oppositeWeight;' +
            //'else {'+
            'atraction += dirToOppositeN*max(1.0, distN*abs(oppositeWeight)*(m1/2.0));\n' + 'repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(oppositeWeight)*(m2/2.0));\n' + 'acumAtraction += 1.0;\n' +
            //'}'+
            '} else if(currentIsParent == 0.5) {' +
            //'if(enableNeuronalNetwork == 1.0) '+
            'netParentErrorWeightA += oppositeNetErrorA*currentWeight;' + 'netParentErrorWeightB += oppositeNetErrorB*currentWeight;' + 'netParentErrorWeightC += oppositeNetErrorC*currentWeight;' + 'netParentErrorWeightD += oppositeNetErrorD*currentWeight;' + 'netParentErrorWeightE += oppositeNetErrorE*currentWeight;' + 'netParentErrorWeightF += oppositeNetErrorF*currentWeight;' + 'netParentErrorWeightG += oppositeNetErrorG*currentWeight;' +
            //'else {'+
            'atraction += dirToOppositeN*max(1.0, distN*abs(currentWeight)*m1);\n' + 'repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*m2);\n' + 'acumAtraction += 1.0;\n' +
            //'}'+
            '}' +

            //'if(enableNeuronalNetwork == 0.0) {'+
            'repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*(m2/8.0));\n' + 'acumAtraction += 1.0;\n' +
            //'}'+
            '}' + ("float collisionExists = 0.0;\n            if(enableForceLayoutCollision == 1.0 && dist < 4.0) {\n                collisionExists = 1.0;\n                atraction = sphericalColl(currentDir, oppositeDir, dirToOppositeN);\n            }\n\n            return CalculationResponse(atraction, acumAtraction, repulsion, collisionExists,\n                                        netChildInputSumA, netParentErrorWeightA,\n                                        netChildInputSumB, netParentErrorWeightB,\n                                        netChildInputSumC, netParentErrorWeightC,\n                                        netChildInputSumD, netParentErrorWeightD,\n                                        netChildInputSumE, netParentErrorWeightE,\n                                        netChildInputSumF, netParentErrorWeightF,\n                                        netChildInputSumG, netParentErrorWeightG);\n        }\n        struct idAdjMatrixResponse {\n            vec3 force;\n            float collisionExists;\n            float netFOutputA;\n            float netErrorWeightA;\n            float netFOutputB;\n            float netErrorWeightB;\n            float netFOutputC;\n            float netErrorWeightC;\n            float netFOutputD;\n            float netErrorWeightD;\n            float netFOutputE;\n            float netErrorWeightE;\n            float netFOutputF;\n            float netErrorWeightF;\n            float netFOutputG;\n            float netErrorWeightG;\n        };\n        float tanh(float val) {\n            float tmp = exp(val);\n            float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n            return tanH;\n        }\n        float sigm(float val) {\n            return (1.0 / (1.0 + exp(-val)));\n        }\n        idAdjMatrixResponse idAdjMatrix_ForceLayout(float nodeId, vec3 currentPos, vec3 currentDir, float numOfConnections, float currentTimestamp, float bornDate, float dieDate, float enableNeuronalNetwork) {\n            vec3 atraction = vec3(0.0, 0.0, 0.0);\n            float acumAtraction = 1.0;\n            vec3 repulsion = vec3(0.0, 0.0, 0.0);\n\n            float collisionExists = 0.0;\n            vec3 force = vec3(0.0, 0.0, 0.0);\n\n\n            float netChildInputSumA = 0.0;\n            float foutputA = 0.0;\n            float netParentErrorWeightA = 0.0;\n            \n            float netChildInputSumB = 0.0;\n            float foutputB = 0.0;\n            float netParentErrorWeightB = 0.0;\n            \n            float netChildInputSumC = 0.0;\n            float foutputC = 0.0;\n            float netParentErrorWeightC = 0.0;\n            \n            float netChildInputSumD = 0.0;\n            float foutputD = 0.0;\n            float netParentErrorWeightD = 0.0;\n            \n            float netChildInputSumE = 0.0;\n            float foutputE = 0.0;\n            float netParentErrorWeightE = 0.0;\n            \n            float netChildInputSumF = 0.0;\n            float foutputF = 0.0;\n            float netParentErrorWeightF = 0.0;\n            \n            float netChildInputSumG = 0.0;\n            float foutputG = 0.0;\n            float netParentErrorWeightG = 0.0;\n            \n\n            if(nodeId < nodesCount) {\n                float currentActivationFn = 0.0;\n                vec2 xGeomCurrent = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n                for(int n=0; n < 4096; n++) {\n                    if(float(n) >= nodesCount) {break;}\n                    if(float(n) != nodeId) {\n                        vec2 xGeomOpposite = get_global_id(float(n), uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n                        vec2 xAdjMatCurrent = get_global_id(vec2(float(n), nodeId), widthAdjMatrix);\n                        vec2 xAdjMatOpposite = get_global_id(vec2(nodeId, float(n)), widthAdjMatrix);\n\n                        vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n                        vec4 pixAdjMatAOpposite = adjacencyMatrix[xAdjMatOpposite];\n\n                        vec4 pixAdjMatBCurrent = adjacencyMatrixB[xAdjMatCurrent];\n                        vec4 pixAdjMatBOpposite = adjacencyMatrixB[xAdjMatOpposite];\n\n\n                        CalculationResponse calcResponse = calculate(nodeId,\n                                                                    pixAdjMatACurrent, pixAdjMatAOpposite,\n                                                                    pixAdjMatBCurrent, pixAdjMatBOpposite,\n                                                                    xGeomCurrent, xGeomOpposite,\n                                                                    currentPos, currentDir,\n                                                                    atraction, acumAtraction, repulsion,\n                                                                    enableNeuronalNetwork,\n                                                                    netChildInputSumA, netParentErrorWeightA,\n                                                                    netChildInputSumB, netParentErrorWeightB,\n                                                                    netChildInputSumC, netParentErrorWeightC,\n                                                                    netChildInputSumD, netParentErrorWeightD,\n                                                                    netChildInputSumE, netParentErrorWeightE,\n                                                                    netChildInputSumF, netParentErrorWeightF,\n                                                                    netChildInputSumG, netParentErrorWeightG);\n                        atraction = calcResponse.atraction;\n                        acumAtraction = calcResponse.acumAtraction;\n                        repulsion = calcResponse.repulsion;\n                        \n                        \n                        netChildInputSumA = calcResponse.netChildInputSumA;\n                        netParentErrorWeightA = calcResponse.netParentErrorWeightA;\n                        \n                        netChildInputSumB = calcResponse.netChildInputSumB;\n                        netParentErrorWeightB = calcResponse.netParentErrorWeightB;\n                        \n                        netChildInputSumC = calcResponse.netChildInputSumC;\n                        netParentErrorWeightC = calcResponse.netParentErrorWeightC;\n                        \n                        netChildInputSumD = calcResponse.netChildInputSumD;\n                        netParentErrorWeightD = calcResponse.netParentErrorWeightD;\n                        \n                        netChildInputSumE = calcResponse.netChildInputSumE;\n                        netParentErrorWeightE = calcResponse.netParentErrorWeightE;\n                        \n                        netChildInputSumF = calcResponse.netChildInputSumF;\n                        netParentErrorWeightF = calcResponse.netParentErrorWeightF;\n                        \n                        netChildInputSumG = calcResponse.netChildInputSumG;\n                        netParentErrorWeightG = calcResponse.netParentErrorWeightG;\n\n\n                        if(calcResponse.collisionExists == 1.0) {\n                            collisionExists = 1.0;\n                            force = calcResponse.atraction;\n                            break;\n                        }\n\n                        if(dieDate != 0.0) {\n                            if(currentTimestamp < bornDate || currentTimestamp > dieDate) {\n                                force = vec3(0.0, 0.0, 0.0);\n                                break;\n                            }\n                        }\n                    }\n                }\n\n                if(collisionExists == 0.0) {\n                    force += (atraction/acumAtraction)*1.0;\n                    force += (repulsion/acumAtraction)*1.0;\n                }\n\n                if(enableNeuronalNetwork == 1.0) {\n                    " + GraphUtils.efferentNodesStr(efferentStart, efferentNodesCount) + "\n                }\n            }\n\n            return idAdjMatrixResponse(vec3(force), collisionExists,\n                                        foutputA, netParentErrorWeightA,\n                                        foutputB, netParentErrorWeightB,\n                                        foutputC, netParentErrorWeightC,\n                                        foutputD, netParentErrorWeightD,\n                                        foutputE, netParentErrorWeightE,\n                                        foutputF, netParentErrorWeightF,\n                                        foutputG, netParentErrorWeightG);\n        }");
        }
    }, {
        key: "efferentNodesStr",
        value: function efferentNodesStr(efferentStart, efferentNodesCount) {
            var str = "\n            if(nodeId < afferentNodesCount) {\n                for(float n=0.0; n < 1024.0; n+=1.0) {\n                    if(n >= afferentNodesCount) {\n                        break;\n                    }\n                    if(nodeId == n) {\n                        foutputA = afferentNodesA[int(n)];\n                        foutputB = afferentNodesB[int(n)];\n                        foutputC = afferentNodesC[int(n)];\n                        foutputD = afferentNodesD[int(n)];\n                        foutputE = afferentNodesE[int(n)];\n                        foutputF = afferentNodesF[int(n)];\n                        foutputG = afferentNodesG[int(n)];\n                        break;\n                    }\n                }\n            } else {\n                foutputA = max(0.0, netChildInputSumA); " + "\n                foutputB = max(0.0, netChildInputSumB);\n                foutputC = max(0.0, netChildInputSumC);\n                foutputD = max(0.0, netChildInputSumD);\n                foutputE = max(0.0, netChildInputSumE);\n                foutputF = max(0.0, netChildInputSumF);\n                foutputG = max(0.0, netChildInputSumG);\n            }";

            str += "\n        if(nodeId == " + efferentStart.toFixed(1) + (") {\n            netParentErrorWeightA = (efferentNodesA[0] != 0.0) ? netChildInputSumA-efferentNodesA[0] : 0.0;\n            " + "\n            netParentErrorWeightB = (efferentNodesB[0] != 0.0) ? netChildInputSumB-efferentNodesB[0] : 0.0;\n            " + "\n            netParentErrorWeightC = (efferentNodesC[0] != 0.0) ? netChildInputSumC-efferentNodesC[0] : 0.0;\n            " + "\n            netParentErrorWeightD = (efferentNodesD[0] != 0.0) ? netChildInputSumD-efferentNodesD[0] : 0.0;\n            " + "\n            netParentErrorWeightE = (efferentNodesE[0] != 0.0) ? netChildInputSumE-efferentNodesE[0] : 0.0;\n            " + "\n            netParentErrorWeightF = (efferentNodesF[0] != 0.0) ? netChildInputSumF-efferentNodesF[0] : 0.0;\n            " + "\n            netParentErrorWeightG = (efferentNodesG[0] != 0.0) ? netChildInputSumG-efferentNodesG[0] : 0.0;\n            " + "\n        }");
            for (var n = efferentStart + 1; n < efferentStart + efferentNodesCount; n++) {
                str += "\n            else if(nodeId == " + n.toFixed(1) + ") {\n                netParentErrorWeightA = (efferentNodesA[" + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumA-efferentNodesA[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightB = (efferentNodesB[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumB-efferentNodesB[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightC = (efferentNodesC[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumC-efferentNodesC[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightD = (efferentNodesD[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumD-efferentNodesD[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightE = (efferentNodesE[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumE-efferentNodesE[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightF = (efferentNodesF[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumF-efferentNodesF[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n                netParentErrorWeightG = (efferentNodesG[") + Math.round(n - efferentStart) + "] != 0.0) ? netChildInputSumG-efferentNodesG[" + Math.round(n - efferentStart) + ("] : 0.0;\n                " + "\n            }");
            }str += "\n        else {\n            if(foutputA <= 0.0) {\n                netParentErrorWeightA = 0.0;\n            }\n            if(foutputB <= 0.0) {\n                netParentErrorWeightB = 0.0;\n            }\n            if(foutputC <= 0.0) {\n                netParentErrorWeightC = 0.0;\n            }\n            if(foutputD <= 0.0) {\n                netParentErrorWeightD = 0.0;\n            }\n            if(foutputE <= 0.0) {\n                netParentErrorWeightE = 0.0;\n            }\n            if(foutputF <= 0.0) {\n                netParentErrorWeightF = 0.0;\n            }\n            if(foutputG <= 0.0) {\n                netParentErrorWeightG = 0.0;\n            }\n        }";

            return str;
        }
    }, {
        key: "adjMatrix_Autolink_GLSLFunctionString",
        value: function adjMatrix_Autolink_GLSLFunctionString(geometryLength) {
            return '' + 'float GetAngle(vec3 A, vec3 B) {' + // from -180.0 to 180.0
            'vec3 cr = cross(A, B);' + 'float d = dot(A, B);' + 'if(cr.y < 0.0) {' + 'if(d > 0.0) {' + 'd =        (1.0-d)*90.0;' + '} else {' + 'd = 90.0+  (abs(d)*90.0);' + '}' + '} else {' + 'if(d > 0.0) {' + 'd = 270.0+ (d*90.0);' + '} else {' + 'd = 180.0+ ((1.0-abs(d))*90.0);' + '}' + '}' + 'return d;' + '}' + 'vec4 idAdjMatrix_Autolink(float nodeId, vec3 currentPos) {\n' +
            // INIT VARS
            'vec2 totalIDrelation = vec2(0.0, 0.0);' + 'float totalAngleRelations = 0.0;' +
            // END INIT VARS

            'if(nodeId < nodesCount) {\n' + 'for(int n=0; n < 4096; n++) {\n' + 'if(float(n) >= nodesCount) break;\n' + 'if(float(n) != nodeId) {' + 'vec2 xAdjMatCurrent = get_global_id(vec2(float(n), nodeId), widthAdjMatrix);' + 'vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n' +

            // RELATION FOUND
            'if(pixAdjMatACurrent.x > 0.0) {' + 'vec2 xGeomOpposite = get_global_id(float(n), uBufferWidth, ' + geometryLength.toFixed(1) + ');\n' + 'vec3 currentPosB = posXYZW[xGeomOpposite].xyz;\n' + 'vec3 dirToBN = normalize(currentPosB-currentPos);\n' + 'vec2 IDrelation = vec2(0.0, 0.0);' + 'float angleRelations = 360.0;' + 'if(nodeId < nodesCount) {\n' + 'for(int nB=0; nB < 4096; nB++) {\n' + 'if(float(nB) >= nodesCount) break;\n' + 'if(float(nB) != float(n) && float(nB) != nodeId) {' + 'vec2 xAdjMatCurrentB = get_global_id(vec2(float(nB), nodeId), widthAdjMatrix);' + 'vec4 pixAdjMatACurrent_B = adjacencyMatrix[xAdjMatCurrentB];\n' + 'if(pixAdjMatACurrent_B.x > 0.0) {' + 'vec2 xGeom_oppoB = get_global_id(float(nB), uBufferWidth, ' + geometryLength.toFixed(1) + ');\n' + 'vec3 currentPosBB = posXYZW[xGeom_oppoB].xyz;\n' + 'vec3 dirToBBN = normalize(currentPosBB-currentPos);\n' + 'float angle = GetAngle(dirToBN,dirToBBN);' + 'if(angle > 0.0 && angle < angleRelations) {' + 'IDrelation = xGeom_oppoB;' + 'angleRelations = angle;' + '}' + '}' + '}' + '}' + '}' + 'if(angleRelations < 360.0 && angleRelations > totalAngleRelations) {' + 'totalIDrelation = IDrelation;' + 'totalAngleRelations = angleRelations;' + '}' + '}' +
            // END RELATION FOUND

            '}' + '}' +
            // SUMMATION
            // END SUMMATION

            '}' + 'return vec4(totalIDrelation, totalAngleRelations, 0.0);' + '}';
        }
    }]);

    return GraphUtils;
}();

global.GraphUtils = GraphUtils;
module.exports.GraphUtils = GraphUtils;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}]},{},[1]);
