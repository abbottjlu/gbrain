(function(){function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s}return e})()({1:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var KERNEL_DIR = exports.KERNEL_DIR = function () {
    function KERNEL_DIR() {
        _classCallCheck(this, KERNEL_DIR);
    }

    _createClass(KERNEL_DIR, null, [{
        key: "getSrc",
        value: function getSrc(customCode, geometryLength, afferentNodesCount, efferentStart, efferentNodesCount) {
            var outputArr = ["dir", "posXYZW", "dataB", "dataF", "dataG", "dataH"];
            var returnStr = 'return [vec4(currentDir, 1.0), vec4(currentPos.x, currentPos.y, currentPos.z, 1.0), currentDataB, currentDataF, currentDataG, currentDataH];';

            return ["x", outputArr,
            // head
            "float tanh(float val) {\n                float tmp = exp(val);\n                float tanH = (tmp - 1.0 / tmp) / (tmp + 1.0 / tmp);\n                return tanH;\n            }\n            float sigm(float val) {\n                return (1.0 / (1.0 + exp(-val)));\n            }",

            // source
            "float nodeId = data[x].x;\n            vec2 xGeometry = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n            vec3 currentPos = posXYZW[xGeometry].xyz;\n            vec3 currentDir = dir[xGeometry].xyz;\n\n\n            vec4 currentDataB = dataB[xGeometry];\n            vec4 currentDataF = dataF[xGeometry];\n            vec4 currentDataG = dataG[xGeometry];\n            vec4 currentDataH = dataH[xGeometry];\n\n            currentDir = vec3(0.0, 0.0, 0.0);\n\n            \n            vec3 atraction = vec3(0.0, 0.0, 0.0);\n            float acumAtraction = 1.0;\n            vec3 repulsion = vec3(0.0, 0.0, 0.0);\n\n            vec3 force = vec3(0.0, 0.0, 0.0);\n\n\n            float netChildInputSumA = 0.0;\n            float foutputA = 0.0;\n            float netParentErrorWeightA = 0.0;\n            \n            float netChildInputSumB = 0.0;\n            float foutputB = 0.0;\n            float netParentErrorWeightB = 0.0;\n            \n            float netChildInputSumC = 0.0;\n            float foutputC = 0.0;\n            float netParentErrorWeightC = 0.0;\n            \n            float netChildInputSumD = 0.0;\n            float foutputD = 0.0;\n            float netParentErrorWeightD = 0.0;\n            \n            float netChildInputSumE = 0.0;\n            float foutputE = 0.0;\n            float netParentErrorWeightE = 0.0;\n            \n\n            if(nodeId < nodesCount && enableTrain == 0.0) {\n                float currentActivationFn = 0.0;\n                vec2 xGeomCurrent = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n                for(int n=0; n < 4096; n++) {\n                    if(float(n) >= nodesCount) {break;}\n                    if(float(n) != nodeId) {\n                        vec2 xGeomOpposite = get_global_id(float(n), uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n                        vec2 xAdjMatCurrent = get_global_id(vec2(float(n), nodeId), widthAdjMatrix);\n                        vec2 xAdjMatOpposite = get_global_id(vec2(nodeId, float(n)), widthAdjMatrix);\n\n                        vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n                        vec4 pixAdjMatAOpposite = adjacencyMatrix[xAdjMatOpposite];\n\n                        vec4 pixAdjMatBCurrent = adjacencyMatrixB[xAdjMatCurrent];\n                        vec4 pixAdjMatBOpposite = adjacencyMatrixB[xAdjMatOpposite];\n\n\n                                                                    \n                        " + "\n                        float currentLayerNum = pixAdjMatACurrent.x;\n                        float currentWeight = pixAdjMatACurrent.z;\n                        float currentIsParent = pixAdjMatACurrent.w;\n            \n                        " + "\n                        float oppositeLayerNum = pixAdjMatAOpposite.x;\n                        float oppositeWeight = pixAdjMatAOpposite.z;\n                        float oppositeIsParent = pixAdjMatAOpposite.w;\n            \n            \n                        " + "\n                        float currentLinkMultiplier = pixAdjMatBCurrent.x;\n                        float currentActivationFn = pixAdjMatBCurrent.y;\n            \n                        " + "\n                        float oppositeLinkMultiplier = pixAdjMatBOpposite.x;\n                        float oppositeActivationFn = pixAdjMatBOpposite.y;\n            \n            \n            \n                        " + "\n                        float currentBiasNode = dataB[xGeomCurrent].x;\n                        " + "\n            \n                        " + "\n                        float oppositeBiasNode = dataB[xGeomOpposite].x;\n                        \n                        float oppositeNetErrorA = dataB[xGeomOpposite].y;\n                        float oppositeNetOutputA = dataB[xGeomOpposite].z;\n                        float oppositeInputsumA = dataB[xGeomOpposite].w;\n                        \n                        float oppositeNetErrorB = dataF[xGeomOpposite].x;\n                        float oppositeNetOutputB = dataF[xGeomOpposite].y;\n                        float oppositeInputsumB = dataF[xGeomOpposite].z;\n                    \n                        float oppositeNetErrorC = dataF[xGeomOpposite].w;\n                        float oppositeNetOutputC = dataG[xGeomOpposite].x;\n                        float oppositeInputsumC = dataG[xGeomOpposite].y;\n                    \n                        float oppositeNetErrorD = dataG[xGeomOpposite].z;\n                        float oppositeNetOutputD = dataG[xGeomOpposite].w;\n                        float oppositeInputsumD = dataH[xGeomOpposite].x;\n                    \n                        float oppositeNetErrorE = dataH[xGeomOpposite].y;\n                        float oppositeNetOutputE = dataH[xGeomOpposite].z;\n                        float oppositeInputsumE = dataH[xGeomOpposite].w;\n            \n            \n                        " + "\n                        " + "\n                        " + "\n            \n                        " + "\n                        vec3 oppositePos = posXYZW[xGeomOpposite].xyz;\n                        vec3 oppositeDir = dir[xGeomOpposite].xyz;\n            \n                        " + "\n                        vec3 dirToOpposite = (oppositePos-currentPos);\n                        vec3 dirToOppositeN = normalize(dirToOpposite);\n            \n                        float dist = distance(oppositePos, currentPos); " + "\n                        float distN = max(0.0,dist)/100000.0;\n            \n                        float mm = 10000000.0;\n                        float m1 = 400000.0/mm;\n                        float m2 = 48.0/mm;\n                        if(currentIsParent == 1.0) {\n                            netChildInputSumA += oppositeNetOutputA*oppositeWeight;\n                            netChildInputSumB += oppositeNetOutputB*oppositeWeight;\n                            netChildInputSumC += oppositeNetOutputC*oppositeWeight;\n                            netChildInputSumD += oppositeNetOutputD*oppositeWeight;\n                            netChildInputSumE += oppositeNetOutputE*oppositeWeight;\n                            \n                            atraction += dirToOppositeN*max(1.0, distN*abs(oppositeWeight)*(m1/2.0));\n                            repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(oppositeWeight)*(m2/2.0));\n                            acumAtraction += 1.0;\n                        } else if(currentIsParent == 0.5) {\n                            float parentGOutputDerivA = 1.0;                    \n                            float parentGOutputDerivB = 1.0;\n                            float parentGOutputDerivC = 1.0;\n                            float parentGOutputDerivD = 1.0;\n                            float parentGOutputDerivE = 1.0;\n                            if(currentLayerNum < layerCount-1.0) { \n                                parentGOutputDerivA = (oppositeInputsumA <= 0.0) ? 0.01 : 1.0;                    \n                                parentGOutputDerivB = (oppositeInputsumB <= 0.0) ? 0.01 : 1.0;\n                                parentGOutputDerivC = (oppositeInputsumC <= 0.0) ? 0.01 : 1.0;\n                                parentGOutputDerivD = (oppositeInputsumD <= 0.0) ? 0.01 : 1.0;\n                                parentGOutputDerivE = (oppositeInputsumE <= 0.0) ? 0.01 : 1.0;\n                            }\n                            \n                            if(currentBiasNode == 0.0) {\n                                netParentErrorWeightA += oppositeNetErrorA*parentGOutputDerivA*currentWeight;\n                                netParentErrorWeightB += oppositeNetErrorB*parentGOutputDerivB*currentWeight;\n                                netParentErrorWeightC += oppositeNetErrorC*parentGOutputDerivC*currentWeight;\n                                netParentErrorWeightD += oppositeNetErrorD*parentGOutputDerivD*currentWeight;\n                                netParentErrorWeightE += oppositeNetErrorE*parentGOutputDerivE*currentWeight;\n                            } else {\n                                netParentErrorWeightA += oppositeNetErrorA*parentGOutputDerivA;\n                                netParentErrorWeightB += oppositeNetErrorB*parentGOutputDerivB;\n                                netParentErrorWeightC += oppositeNetErrorC*parentGOutputDerivC;\n                                netParentErrorWeightD += oppositeNetErrorD*parentGOutputDerivD;\n                                netParentErrorWeightE += oppositeNetErrorE*parentGOutputDerivE;\n                            }\n                            atraction += dirToOppositeN*max(1.0, distN*abs(currentWeight)*m1);\n                            repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*m2);\n                            acumAtraction += 1.0;\n                        }\n            \n                        repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*(m2/8.0));\n                        acumAtraction += 1.0;\n                    }\n                }\n                \n                float vndm = (viewNeuronDynamics == 1.0) ? netChildInputSumA : 1.0;\n                force += (atraction/acumAtraction)*abs(vndm);\n                force += (repulsion/acumAtraction)*abs(vndm);\n                currentDir += force;\n                \n                \n                float currentBiasNode = dataB[xGeomCurrent].x;\n                \n                " + KERNEL_DIR.efferentNodesStr(afferentNodesCount, efferentStart, efferentNodesCount) + "\n                \n                currentDataB = vec4(currentDataB.x, netParentErrorWeightA, foutputA, netChildInputSumA);\n                currentDataF = vec4(netParentErrorWeightB, foutputB, netChildInputSumB, netParentErrorWeightC);\n                currentDataG = vec4(foutputC, netChildInputSumC, netParentErrorWeightD, foutputD);\n                currentDataH = vec4(netChildInputSumD, netParentErrorWeightE, foutputE, netChildInputSumE);\n            }\n\n            " + (customCode !== undefined ? customCode : '') + "\n\n            if(enableDrag == 1.0) {\n                if(nodeId == idToDrag) {\n                    currentPos = vec3(MouseDragTranslationX, MouseDragTranslationY, MouseDragTranslationZ);\n                }\n            }\n\n            currentPos += currentDir;\n            if(only2d == 1.0) {\n                currentPos.y = 0.0;\n            }\n\n            " + returnStr];
        }
    }, {
        key: "efferentNodesStr",
        value: function efferentNodesStr(afferentNodesCount, efferentStart, efferentNodesCount) {
            /////////////////////////////////////////////////
            // OUTPUT
            /////////////////////////////////////////////////
            var str = "\n            if(nodeId < afferentNodesCount) {\n                for(float n=0.0; n < 1024.0; n+=1.0) {\n                    if(n >= afferentNodesCount) {\n                        break;\n                    }\n                    if(nodeId == n) {\n                        foutputA = afferentNodesA[int(n)];\n                        foutputB = afferentNodesB[int(n)];\n                        foutputC = afferentNodesC[int(n)];\n                        foutputD = afferentNodesD[int(n)];\n                        foutputE = afferentNodesE[int(n)];\n                        break;\n                    }\n                }\n            } else {\n                if(currentBiasNode == 0.0) {                                     \n                    foutputA = (netChildInputSumA <= 0.0) ? 0.01*netChildInputSumA : netChildInputSumA; " + "\n                    foutputB = (netChildInputSumB <= 0.0) ? 0.01*netChildInputSumB : netChildInputSumB;\n                    foutputC = (netChildInputSumC <= 0.0) ? 0.01*netChildInputSumC : netChildInputSumC;\n                    foutputD = (netChildInputSumD <= 0.0) ? 0.01*netChildInputSumD : netChildInputSumD;\n                    foutputE = (netChildInputSumE <= 0.0) ? 0.01*netChildInputSumE : netChildInputSumE;\n                } else {\n                    foutputA = 1.0;\n                    foutputB = 1.0;\n                    foutputC = 1.0;\n                    foutputD = 1.0;\n                    foutputE = 1.0;\n                }\n            }";

            /////////////////////////////////////////////////
            // ERROR
            /////////////////////////////////////////////////
            for (var n = efferentStart; n < efferentStart + efferentNodesCount; n++) {
                var cond = n === efferentStart ? "if" : "else if";
                str += "\n            " + cond + "(nodeId == " + n.toFixed(1) + (") {\n                foutputA = netChildInputSumA; " + " \n                foutputB = netChildInputSumB;\n                foutputC = netChildInputSumC;\n                foutputD = netChildInputSumD;\n                foutputE = netChildInputSumE;\n                    \n                netParentErrorWeightA = efferentNodesA[") + Math.round(n - efferentStart) + "];\n                netParentErrorWeightB = efferentNodesB[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightC = efferentNodesC[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightD = efferentNodesD[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightE = efferentNodesE[" + Math.round(n - efferentStart) + "];\n            }";
            }
            /*str += `
            else {
                netParentErrorWeightA *= dataB[xGeometry].z;
                netParentErrorWeightB *= dataF[xGeometry].y;
                netParentErrorWeightC *= dataG[xGeometry].x;
                netParentErrorWeightD *= dataG[xGeometry].w;
                netParentErrorWeightE *= dataH[xGeometry].z;
            }`;*/
            return str;
        }
    }]);

    return KERNEL_DIR;
}();

global.KERNEL_DIR = KERNEL_DIR;
module.exports.KERNEL_DIR = KERNEL_DIR;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})

},{}]},{},[1])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvZ2JyYWluL0tFUk5FTF9ESVIuY2xhc3MuanMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7O0FDQUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24oKXtmdW5jdGlvbiBlKHQsbixyKXtmdW5jdGlvbiBzKG8sdSl7aWYoIW5bb10pe2lmKCF0W29dKXt2YXIgYT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2lmKCF1JiZhKXJldHVybiBhKG8sITApO2lmKGkpcmV0dXJuIGkobywhMCk7dmFyIGY9bmV3IEVycm9yKFwiQ2Fubm90IGZpbmQgbW9kdWxlICdcIitvK1wiJ1wiKTt0aHJvdyBmLmNvZGU9XCJNT0RVTEVfTk9UX0ZPVU5EXCIsZn12YXIgbD1uW29dPXtleHBvcnRzOnt9fTt0W29dWzBdLmNhbGwobC5leHBvcnRzLGZ1bmN0aW9uKGUpe3ZhciBuPXRbb11bMV1bZV07cmV0dXJuIHMobj9uOmUpfSxsLGwuZXhwb3J0cyxlLHQsbixyKX1yZXR1cm4gbltvXS5leHBvcnRzfXZhciBpPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7Zm9yKHZhciBvPTA7bzxyLmxlbmd0aDtvKyspcyhyW29dKTtyZXR1cm4gc31yZXR1cm4gZX0pKCkiLCJcInVzZSBzdHJpY3RcIjtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gICAgdmFsdWU6IHRydWVcbn0pO1xuXG52YXIgX2NyZWF0ZUNsYXNzID0gZnVuY3Rpb24gKCkgeyBmdW5jdGlvbiBkZWZpbmVQcm9wZXJ0aWVzKHRhcmdldCwgcHJvcHMpIHsgZm9yICh2YXIgaSA9IDA7IGkgPCBwcm9wcy5sZW5ndGg7IGkrKykgeyB2YXIgZGVzY3JpcHRvciA9IHByb3BzW2ldOyBkZXNjcmlwdG9yLmVudW1lcmFibGUgPSBkZXNjcmlwdG9yLmVudW1lcmFibGUgfHwgZmFsc2U7IGRlc2NyaXB0b3IuY29uZmlndXJhYmxlID0gdHJ1ZTsgaWYgKFwidmFsdWVcIiBpbiBkZXNjcmlwdG9yKSBkZXNjcmlwdG9yLndyaXRhYmxlID0gdHJ1ZTsgT2JqZWN0LmRlZmluZVByb3BlcnR5KHRhcmdldCwgZGVzY3JpcHRvci5rZXksIGRlc2NyaXB0b3IpOyB9IH0gcmV0dXJuIGZ1bmN0aW9uIChDb25zdHJ1Y3RvciwgcHJvdG9Qcm9wcywgc3RhdGljUHJvcHMpIHsgaWYgKHByb3RvUHJvcHMpIGRlZmluZVByb3BlcnRpZXMoQ29uc3RydWN0b3IucHJvdG90eXBlLCBwcm90b1Byb3BzKTsgaWYgKHN0YXRpY1Byb3BzKSBkZWZpbmVQcm9wZXJ0aWVzKENvbnN0cnVjdG9yLCBzdGF0aWNQcm9wcyk7IHJldHVybiBDb25zdHJ1Y3RvcjsgfTsgfSgpO1xuXG5mdW5jdGlvbiBfY2xhc3NDYWxsQ2hlY2soaW5zdGFuY2UsIENvbnN0cnVjdG9yKSB7IGlmICghKGluc3RhbmNlIGluc3RhbmNlb2YgQ29uc3RydWN0b3IpKSB7IHRocm93IG5ldyBUeXBlRXJyb3IoXCJDYW5ub3QgY2FsbCBhIGNsYXNzIGFzIGEgZnVuY3Rpb25cIik7IH0gfVxuXG52YXIgS0VSTkVMX0RJUiA9IGV4cG9ydHMuS0VSTkVMX0RJUiA9IGZ1bmN0aW9uICgpIHtcbiAgICBmdW5jdGlvbiBLRVJORUxfRElSKCkge1xuICAgICAgICBfY2xhc3NDYWxsQ2hlY2sodGhpcywgS0VSTkVMX0RJUik7XG4gICAgfVxuXG4gICAgX2NyZWF0ZUNsYXNzKEtFUk5FTF9ESVIsIG51bGwsIFt7XG4gICAgICAgIGtleTogXCJnZXRTcmNcIixcbiAgICAgICAgdmFsdWU6IGZ1bmN0aW9uIGdldFNyYyhjdXN0b21Db2RlLCBnZW9tZXRyeUxlbmd0aCwgYWZmZXJlbnROb2Rlc0NvdW50LCBlZmZlcmVudFN0YXJ0LCBlZmZlcmVudE5vZGVzQ291bnQpIHtcbiAgICAgICAgICAgIHZhciBvdXRwdXRBcnIgPSBbXCJkaXJcIiwgXCJwb3NYWVpXXCIsIFwiZGF0YUJcIiwgXCJkYXRhRlwiLCBcImRhdGFHXCIsIFwiZGF0YUhcIl07XG4gICAgICAgICAgICB2YXIgcmV0dXJuU3RyID0gJ3JldHVybiBbdmVjNChjdXJyZW50RGlyLCAxLjApLCB2ZWM0KGN1cnJlbnRQb3MueCwgY3VycmVudFBvcy55LCBjdXJyZW50UG9zLnosIDEuMCksIGN1cnJlbnREYXRhQiwgY3VycmVudERhdGFGLCBjdXJyZW50RGF0YUcsIGN1cnJlbnREYXRhSF07JztcblxuICAgICAgICAgICAgcmV0dXJuIFtcInhcIiwgb3V0cHV0QXJyLFxuICAgICAgICAgICAgLy8gaGVhZFxuICAgICAgICAgICAgXCJmbG9hdCB0YW5oKGZsb2F0IHZhbCkge1xcbiAgICAgICAgICAgICAgICBmbG9hdCB0bXAgPSBleHAodmFsKTtcXG4gICAgICAgICAgICAgICAgZmxvYXQgdGFuSCA9ICh0bXAgLSAxLjAgLyB0bXApIC8gKHRtcCArIDEuMCAvIHRtcCk7XFxuICAgICAgICAgICAgICAgIHJldHVybiB0YW5IO1xcbiAgICAgICAgICAgIH1cXG4gICAgICAgICAgICBmbG9hdCBzaWdtKGZsb2F0IHZhbCkge1xcbiAgICAgICAgICAgICAgICByZXR1cm4gKDEuMCAvICgxLjAgKyBleHAoLXZhbCkpKTtcXG4gICAgICAgICAgICB9XCIsXG5cbiAgICAgICAgICAgIC8vIHNvdXJjZVxuICAgICAgICAgICAgXCJmbG9hdCBub2RlSWQgPSBkYXRhW3hdLng7XFxuICAgICAgICAgICAgdmVjMiB4R2VvbWV0cnkgPSBnZXRfZ2xvYmFsX2lkKG5vZGVJZCwgdUJ1ZmZlcldpZHRoLCBcIiArIGdlb21ldHJ5TGVuZ3RoLnRvRml4ZWQoMSkgKyBcIik7XFxuXFxuXFxuICAgICAgICAgICAgdmVjMyBjdXJyZW50UG9zID0gcG9zWFlaV1t4R2VvbWV0cnldLnh5ejtcXG4gICAgICAgICAgICB2ZWMzIGN1cnJlbnREaXIgPSBkaXJbeEdlb21ldHJ5XS54eXo7XFxuXFxuXFxuICAgICAgICAgICAgdmVjNCBjdXJyZW50RGF0YUIgPSBkYXRhQlt4R2VvbWV0cnldO1xcbiAgICAgICAgICAgIHZlYzQgY3VycmVudERhdGFGID0gZGF0YUZbeEdlb21ldHJ5XTtcXG4gICAgICAgICAgICB2ZWM0IGN1cnJlbnREYXRhRyA9IGRhdGFHW3hHZW9tZXRyeV07XFxuICAgICAgICAgICAgdmVjNCBjdXJyZW50RGF0YUggPSBkYXRhSFt4R2VvbWV0cnldO1xcblxcbiAgICAgICAgICAgIGN1cnJlbnREaXIgPSB2ZWMzKDAuMCwgMC4wLCAwLjApO1xcblxcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgIHZlYzMgYXRyYWN0aW9uID0gdmVjMygwLjAsIDAuMCwgMC4wKTtcXG4gICAgICAgICAgICBmbG9hdCBhY3VtQXRyYWN0aW9uID0gMS4wO1xcbiAgICAgICAgICAgIHZlYzMgcmVwdWxzaW9uID0gdmVjMygwLjAsIDAuMCwgMC4wKTtcXG5cXG4gICAgICAgICAgICB2ZWMzIGZvcmNlID0gdmVjMygwLjAsIDAuMCwgMC4wKTtcXG5cXG5cXG4gICAgICAgICAgICBmbG9hdCBuZXRDaGlsZElucHV0U3VtQSA9IDAuMDtcXG4gICAgICAgICAgICBmbG9hdCBmb3V0cHV0QSA9IDAuMDtcXG4gICAgICAgICAgICBmbG9hdCBuZXRQYXJlbnRFcnJvcldlaWdodEEgPSAwLjA7XFxuICAgICAgICAgICAgXFxuICAgICAgICAgICAgZmxvYXQgbmV0Q2hpbGRJbnB1dFN1bUIgPSAwLjA7XFxuICAgICAgICAgICAgZmxvYXQgZm91dHB1dEIgPSAwLjA7XFxuICAgICAgICAgICAgZmxvYXQgbmV0UGFyZW50RXJyb3JXZWlnaHRCID0gMC4wO1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgIGZsb2F0IG5ldENoaWxkSW5wdXRTdW1DID0gMC4wO1xcbiAgICAgICAgICAgIGZsb2F0IGZvdXRwdXRDID0gMC4wO1xcbiAgICAgICAgICAgIGZsb2F0IG5ldFBhcmVudEVycm9yV2VpZ2h0QyA9IDAuMDtcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICBmbG9hdCBuZXRDaGlsZElucHV0U3VtRCA9IDAuMDtcXG4gICAgICAgICAgICBmbG9hdCBmb3V0cHV0RCA9IDAuMDtcXG4gICAgICAgICAgICBmbG9hdCBuZXRQYXJlbnRFcnJvcldlaWdodEQgPSAwLjA7XFxuICAgICAgICAgICAgXFxuICAgICAgICAgICAgZmxvYXQgbmV0Q2hpbGRJbnB1dFN1bUUgPSAwLjA7XFxuICAgICAgICAgICAgZmxvYXQgZm91dHB1dEUgPSAwLjA7XFxuICAgICAgICAgICAgZmxvYXQgbmV0UGFyZW50RXJyb3JXZWlnaHRFID0gMC4wO1xcbiAgICAgICAgICAgIFxcblxcbiAgICAgICAgICAgIGlmKG5vZGVJZCA8IG5vZGVzQ291bnQgJiYgZW5hYmxlVHJhaW4gPT0gMC4wKSB7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGN1cnJlbnRBY3RpdmF0aW9uRm4gPSAwLjA7XFxuICAgICAgICAgICAgICAgIHZlYzIgeEdlb21DdXJyZW50ID0gZ2V0X2dsb2JhbF9pZChub2RlSWQsIHVCdWZmZXJXaWR0aCwgXCIgKyBnZW9tZXRyeUxlbmd0aC50b0ZpeGVkKDEpICsgXCIpO1xcbiAgICAgICAgICAgICAgICBmb3IoaW50IG49MDsgbiA8IDQwOTY7IG4rKykge1xcbiAgICAgICAgICAgICAgICAgICAgaWYoZmxvYXQobikgPj0gbm9kZXNDb3VudCkge2JyZWFrO31cXG4gICAgICAgICAgICAgICAgICAgIGlmKGZsb2F0KG4pICE9IG5vZGVJZCkge1xcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIgeEdlb21PcHBvc2l0ZSA9IGdldF9nbG9iYWxfaWQoZmxvYXQobiksIHVCdWZmZXJXaWR0aCwgXCIgKyBnZW9tZXRyeUxlbmd0aC50b0ZpeGVkKDEpICsgXCIpO1xcblxcblxcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIgeEFkak1hdEN1cnJlbnQgPSBnZXRfZ2xvYmFsX2lkKHZlYzIoZmxvYXQobiksIG5vZGVJZCksIHdpZHRoQWRqTWF0cml4KTtcXG4gICAgICAgICAgICAgICAgICAgICAgICB2ZWMyIHhBZGpNYXRPcHBvc2l0ZSA9IGdldF9nbG9iYWxfaWQodmVjMihub2RlSWQsIGZsb2F0KG4pKSwgd2lkdGhBZGpNYXRyaXgpO1xcblxcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzQgcGl4QWRqTWF0QUN1cnJlbnQgPSBhZGphY2VuY3lNYXRyaXhbeEFkak1hdEN1cnJlbnRdO1xcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzQgcGl4QWRqTWF0QU9wcG9zaXRlID0gYWRqYWNlbmN5TWF0cml4W3hBZGpNYXRPcHBvc2l0ZV07XFxuXFxuICAgICAgICAgICAgICAgICAgICAgICAgdmVjNCBwaXhBZGpNYXRCQ3VycmVudCA9IGFkamFjZW5jeU1hdHJpeEJbeEFkak1hdEN1cnJlbnRdO1xcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzQgcGl4QWRqTWF0Qk9wcG9zaXRlID0gYWRqYWNlbmN5TWF0cml4Qlt4QWRqTWF0T3Bwb3NpdGVdO1xcblxcblxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgICAgICAgICAgXCIgKyBcIlxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IGN1cnJlbnRMYXllck51bSA9IHBpeEFkak1hdEFDdXJyZW50Lng7XFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgY3VycmVudFdlaWdodCA9IHBpeEFkak1hdEFDdXJyZW50Lno7XFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgY3VycmVudElzUGFyZW50ID0gcGl4QWRqTWF0QUN1cnJlbnQudztcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVMYXllck51bSA9IHBpeEFkak1hdEFPcHBvc2l0ZS54O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlV2VpZ2h0ID0gcGl4QWRqTWF0QU9wcG9zaXRlLno7XFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVJc1BhcmVudCA9IHBpeEFkak1hdEFPcHBvc2l0ZS53O1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiICsgXCJcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBjdXJyZW50TGlua011bHRpcGxpZXIgPSBwaXhBZGpNYXRCQ3VycmVudC54O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IGN1cnJlbnRBY3RpdmF0aW9uRm4gPSBwaXhBZGpNYXRCQ3VycmVudC55O1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiICsgXCJcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBvcHBvc2l0ZUxpbmtNdWx0aXBsaWVyID0gcGl4QWRqTWF0Qk9wcG9zaXRlLng7XFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVBY3RpdmF0aW9uRm4gPSBwaXhBZGpNYXRCT3Bwb3NpdGUueTtcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgY3VycmVudEJpYXNOb2RlID0gZGF0YUJbeEdlb21DdXJyZW50XS54O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiICsgXCJcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVCaWFzTm9kZSA9IGRhdGFCW3hHZW9tT3Bwb3NpdGVdLng7XFxuICAgICAgICAgICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVOZXRFcnJvckEgPSBkYXRhQlt4R2VvbU9wcG9zaXRlXS55O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlTmV0T3V0cHV0QSA9IGRhdGFCW3hHZW9tT3Bwb3NpdGVdLno7XFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgb3Bwb3NpdGVJbnB1dHN1bUEgPSBkYXRhQlt4R2VvbU9wcG9zaXRlXS53O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlTmV0RXJyb3JCID0gZGF0YUZbeEdlb21PcHBvc2l0ZV0ueDtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBvcHBvc2l0ZU5ldE91dHB1dEIgPSBkYXRhRlt4R2VvbU9wcG9zaXRlXS55O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlSW5wdXRzdW1CID0gZGF0YUZbeEdlb21PcHBvc2l0ZV0uejtcXG4gICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlTmV0RXJyb3JDID0gZGF0YUZbeEdlb21PcHBvc2l0ZV0udztcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBvcHBvc2l0ZU5ldE91dHB1dEMgPSBkYXRhR1t4R2VvbU9wcG9zaXRlXS54O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlSW5wdXRzdW1DID0gZGF0YUdbeEdlb21PcHBvc2l0ZV0ueTtcXG4gICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlTmV0RXJyb3JEID0gZGF0YUdbeEdlb21PcHBvc2l0ZV0uejtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBvcHBvc2l0ZU5ldE91dHB1dEQgPSBkYXRhR1t4R2VvbU9wcG9zaXRlXS53O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlSW5wdXRzdW1EID0gZGF0YUhbeEdlb21PcHBvc2l0ZV0ueDtcXG4gICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlTmV0RXJyb3JFID0gZGF0YUhbeEdlb21PcHBvc2l0ZV0ueTtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBvcHBvc2l0ZU5ldE91dHB1dEUgPSBkYXRhSFt4R2VvbU9wcG9zaXRlXS56O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG9wcG9zaXRlSW5wdXRzdW1FID0gZGF0YUhbeEdlb21PcHBvc2l0ZV0udztcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgXCIgKyBcIlxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiICsgXCJcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgdmVjMyBvcHBvc2l0ZVBvcyA9IHBvc1hZWldbeEdlb21PcHBvc2l0ZV0ueHl6O1xcbiAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMgb3Bwb3NpdGVEaXIgPSBkaXJbeEdlb21PcHBvc2l0ZV0ueHl6O1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIFwiICsgXCJcXG4gICAgICAgICAgICAgICAgICAgICAgICB2ZWMzIGRpclRvT3Bwb3NpdGUgPSAob3Bwb3NpdGVQb3MtY3VycmVudFBvcyk7XFxuICAgICAgICAgICAgICAgICAgICAgICAgdmVjMyBkaXJUb09wcG9zaXRlTiA9IG5vcm1hbGl6ZShkaXJUb09wcG9zaXRlKTtcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBkaXN0ID0gZGlzdGFuY2Uob3Bwb3NpdGVQb3MsIGN1cnJlbnRQb3MpOyBcIiArIFwiXFxuICAgICAgICAgICAgICAgICAgICAgICAgZmxvYXQgZGlzdE4gPSBtYXgoMC4wLGRpc3QpLzEwMDAwMC4wO1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG1tID0gMTAwMDAwMDAuMDtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBtMSA9IDQwMDAwMC4wL21tO1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IG0yID0gNDguMC9tbTtcXG4gICAgICAgICAgICAgICAgICAgICAgICBpZihjdXJyZW50SXNQYXJlbnQgPT0gMS4wKSB7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldENoaWxkSW5wdXRTdW1BICs9IG9wcG9zaXRlTmV0T3V0cHV0QSpvcHBvc2l0ZVdlaWdodDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV0Q2hpbGRJbnB1dFN1bUIgKz0gb3Bwb3NpdGVOZXRPdXRwdXRCKm9wcG9zaXRlV2VpZ2h0O1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXRDaGlsZElucHV0U3VtQyArPSBvcHBvc2l0ZU5ldE91dHB1dEMqb3Bwb3NpdGVXZWlnaHQ7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldENoaWxkSW5wdXRTdW1EICs9IG9wcG9zaXRlTmV0T3V0cHV0RCpvcHBvc2l0ZVdlaWdodDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV0Q2hpbGRJbnB1dFN1bUUgKz0gb3Bwb3NpdGVOZXRPdXRwdXRFKm9wcG9zaXRlV2VpZ2h0O1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgYXRyYWN0aW9uICs9IGRpclRvT3Bwb3NpdGVOKm1heCgxLjAsIGRpc3ROKmFicyhvcHBvc2l0ZVdlaWdodCkqKG0xLzIuMCkpO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXB1bHNpb24gKz0gLWRpclRvT3Bwb3NpdGVOKm1heCgxLjAsICgxLjAtZGlzdE4pKmFicyhvcHBvc2l0ZVdlaWdodCkqKG0yLzIuMCkpO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBhY3VtQXRyYWN0aW9uICs9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2UgaWYoY3VycmVudElzUGFyZW50ID09IDAuNSkge1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmbG9hdCBwYXJlbnRHT3V0cHV0RGVyaXZBID0gMS4wOyAgICAgICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdPdXRwdXREZXJpdkIgPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdPdXRwdXREZXJpdkMgPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdPdXRwdXREZXJpdkQgPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdPdXRwdXREZXJpdkUgPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmKGN1cnJlbnRMYXllck51bSA8IGxheWVyQ291bnQtMS4wKSB7IFxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50R091dHB1dERlcml2QSA9IChvcHBvc2l0ZUlucHV0c3VtQSA8PSAwLjApID8gMC4wMSA6IDEuMDsgICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50R091dHB1dERlcml2QiA9IChvcHBvc2l0ZUlucHV0c3VtQiA8PSAwLjApID8gMC4wMSA6IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHBhcmVudEdPdXRwdXREZXJpdkMgPSAob3Bwb3NpdGVJbnB1dHN1bUMgPD0gMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwYXJlbnRHT3V0cHV0RGVyaXZEID0gKG9wcG9zaXRlSW5wdXRzdW1EIDw9IDAuMCkgPyAwLjAxIDogMS4wO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgcGFyZW50R091dHB1dERlcml2RSA9IChvcHBvc2l0ZUlucHV0c3VtRSA8PSAwLjApID8gMC4wMSA6IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYoY3VycmVudEJpYXNOb2RlID09IDAuMCkge1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV0UGFyZW50RXJyb3JXZWlnaHRBICs9IG9wcG9zaXRlTmV0RXJyb3JBKnBhcmVudEdPdXRwdXREZXJpdkEqY3VycmVudFdlaWdodDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0QiArPSBvcHBvc2l0ZU5ldEVycm9yQipwYXJlbnRHT3V0cHV0RGVyaXZCKmN1cnJlbnRXZWlnaHQ7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEMgKz0gb3Bwb3NpdGVOZXRFcnJvckMqcGFyZW50R091dHB1dERlcml2QypjdXJyZW50V2VpZ2h0O1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV0UGFyZW50RXJyb3JXZWlnaHREICs9IG9wcG9zaXRlTmV0RXJyb3JEKnBhcmVudEdPdXRwdXREZXJpdkQqY3VycmVudFdlaWdodDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0RSArPSBvcHBvc2l0ZU5ldEVycm9yRSpwYXJlbnRHT3V0cHV0RGVyaXZFKmN1cnJlbnRXZWlnaHQ7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEEgKz0gb3Bwb3NpdGVOZXRFcnJvckEqcGFyZW50R091dHB1dERlcml2QTtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0QiArPSBvcHBvc2l0ZU5ldEVycm9yQipwYXJlbnRHT3V0cHV0RGVyaXZCO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbmV0UGFyZW50RXJyb3JXZWlnaHRDICs9IG9wcG9zaXRlTmV0RXJyb3JDKnBhcmVudEdPdXRwdXREZXJpdkM7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEQgKz0gb3Bwb3NpdGVOZXRFcnJvckQqcGFyZW50R091dHB1dERlcml2RDtcXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0RSArPSBvcHBvc2l0ZU5ldEVycm9yRSpwYXJlbnRHT3V0cHV0RGVyaXZFO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGF0cmFjdGlvbiArPSBkaXJUb09wcG9zaXRlTiptYXgoMS4wLCBkaXN0TiphYnMoY3VycmVudFdlaWdodCkqbTEpO1xcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICByZXB1bHNpb24gKz0gLWRpclRvT3Bwb3NpdGVOKm1heCgxLjAsICgxLjAtZGlzdE4pKmFicyhjdXJyZW50V2VpZ2h0KSptMik7XFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGFjdW1BdHJhY3Rpb24gKz0gMS4wO1xcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgICAgICAgICByZXB1bHNpb24gKz0gLWRpclRvT3Bwb3NpdGVOKm1heCgxLjAsICgxLjAtZGlzdE4pKmFicyhjdXJyZW50V2VpZ2h0KSoobTIvOC4wKSk7XFxuICAgICAgICAgICAgICAgICAgICAgICAgYWN1bUF0cmFjdGlvbiArPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICB9XFxuICAgICAgICAgICAgICAgIH1cXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIGZsb2F0IHZuZG0gPSAodmlld05ldXJvbkR5bmFtaWNzID09IDEuMCkgPyBuZXRDaGlsZElucHV0U3VtQSA6IDEuMDtcXG4gICAgICAgICAgICAgICAgZm9yY2UgKz0gKGF0cmFjdGlvbi9hY3VtQXRyYWN0aW9uKSphYnModm5kbSk7XFxuICAgICAgICAgICAgICAgIGZvcmNlICs9IChyZXB1bHNpb24vYWN1bUF0cmFjdGlvbikqYWJzKHZuZG0pO1xcbiAgICAgICAgICAgICAgICBjdXJyZW50RGlyICs9IGZvcmNlO1xcbiAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIGZsb2F0IGN1cnJlbnRCaWFzTm9kZSA9IGRhdGFCW3hHZW9tQ3VycmVudF0ueDtcXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIFwiICsgS0VSTkVMX0RJUi5lZmZlcmVudE5vZGVzU3RyKGFmZmVyZW50Tm9kZXNDb3VudCwgZWZmZXJlbnRTdGFydCwgZWZmZXJlbnROb2Rlc0NvdW50KSArIFwiXFxuICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICBjdXJyZW50RGF0YUIgPSB2ZWM0KGN1cnJlbnREYXRhQi54LCBuZXRQYXJlbnRFcnJvcldlaWdodEEsIGZvdXRwdXRBLCBuZXRDaGlsZElucHV0U3VtQSk7XFxuICAgICAgICAgICAgICAgIGN1cnJlbnREYXRhRiA9IHZlYzQobmV0UGFyZW50RXJyb3JXZWlnaHRCLCBmb3V0cHV0QiwgbmV0Q2hpbGRJbnB1dFN1bUIsIG5ldFBhcmVudEVycm9yV2VpZ2h0Qyk7XFxuICAgICAgICAgICAgICAgIGN1cnJlbnREYXRhRyA9IHZlYzQoZm91dHB1dEMsIG5ldENoaWxkSW5wdXRTdW1DLCBuZXRQYXJlbnRFcnJvcldlaWdodEQsIGZvdXRwdXREKTtcXG4gICAgICAgICAgICAgICAgY3VycmVudERhdGFIID0gdmVjNChuZXRDaGlsZElucHV0U3VtRCwgbmV0UGFyZW50RXJyb3JXZWlnaHRFLCBmb3V0cHV0RSwgbmV0Q2hpbGRJbnB1dFN1bUUpO1xcbiAgICAgICAgICAgIH1cXG5cXG4gICAgICAgICAgICBcIiArIChjdXN0b21Db2RlICE9PSB1bmRlZmluZWQgPyBjdXN0b21Db2RlIDogJycpICsgXCJcXG5cXG4gICAgICAgICAgICBpZihlbmFibGVEcmFnID09IDEuMCkge1xcbiAgICAgICAgICAgICAgICBpZihub2RlSWQgPT0gaWRUb0RyYWcpIHtcXG4gICAgICAgICAgICAgICAgICAgIGN1cnJlbnRQb3MgPSB2ZWMzKE1vdXNlRHJhZ1RyYW5zbGF0aW9uWCwgTW91c2VEcmFnVHJhbnNsYXRpb25ZLCBNb3VzZURyYWdUcmFuc2xhdGlvblopO1xcbiAgICAgICAgICAgICAgICB9XFxuICAgICAgICAgICAgfVxcblxcbiAgICAgICAgICAgIGN1cnJlbnRQb3MgKz0gY3VycmVudERpcjtcXG4gICAgICAgICAgICBpZihvbmx5MmQgPT0gMS4wKSB7XFxuICAgICAgICAgICAgICAgIGN1cnJlbnRQb3MueSA9IDAuMDtcXG4gICAgICAgICAgICB9XFxuXFxuICAgICAgICAgICAgXCIgKyByZXR1cm5TdHJdO1xuICAgICAgICB9XG4gICAgfSwge1xuICAgICAgICBrZXk6IFwiZWZmZXJlbnROb2Rlc1N0clwiLFxuICAgICAgICB2YWx1ZTogZnVuY3Rpb24gZWZmZXJlbnROb2Rlc1N0cihhZmZlcmVudE5vZGVzQ291bnQsIGVmZmVyZW50U3RhcnQsIGVmZmVyZW50Tm9kZXNDb3VudCkge1xuICAgICAgICAgICAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAgICAgICAgICAgLy8gT1VUUFVUXG4gICAgICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4gICAgICAgICAgICB2YXIgc3RyID0gXCJcXG4gICAgICAgICAgICBpZihub2RlSWQgPCBhZmZlcmVudE5vZGVzQ291bnQpIHtcXG4gICAgICAgICAgICAgICAgZm9yKGZsb2F0IG49MC4wOyBuIDwgMTAyNC4wOyBuKz0xLjApIHtcXG4gICAgICAgICAgICAgICAgICAgIGlmKG4gPj0gYWZmZXJlbnROb2Rlc0NvdW50KSB7XFxuICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7XFxuICAgICAgICAgICAgICAgICAgICB9XFxuICAgICAgICAgICAgICAgICAgICBpZihub2RlSWQgPT0gbikge1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZvdXRwdXRBID0gYWZmZXJlbnROb2Rlc0FbaW50KG4pXTtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmb3V0cHV0QiA9IGFmZmVyZW50Tm9kZXNCW2ludChuKV07XFxuICAgICAgICAgICAgICAgICAgICAgICAgZm91dHB1dEMgPSBhZmZlcmVudE5vZGVzQ1tpbnQobildO1xcbiAgICAgICAgICAgICAgICAgICAgICAgIGZvdXRwdXREID0gYWZmZXJlbnROb2Rlc0RbaW50KG4pXTtcXG4gICAgICAgICAgICAgICAgICAgICAgICBmb3V0cHV0RSA9IGFmZmVyZW50Tm9kZXNFW2ludChuKV07XFxuICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWs7XFxuICAgICAgICAgICAgICAgICAgICB9XFxuICAgICAgICAgICAgICAgIH1cXG4gICAgICAgICAgICB9IGVsc2Uge1xcbiAgICAgICAgICAgICAgICBpZihjdXJyZW50Qmlhc05vZGUgPT0gMC4wKSB7ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICAgICAgZm91dHB1dEEgPSAobmV0Q2hpbGRJbnB1dFN1bUEgPD0gMC4wKSA/IDAuMDEqbmV0Q2hpbGRJbnB1dFN1bUEgOiBuZXRDaGlsZElucHV0U3VtQTsgXCIgKyBcIlxcbiAgICAgICAgICAgICAgICAgICAgZm91dHB1dEIgPSAobmV0Q2hpbGRJbnB1dFN1bUIgPD0gMC4wKSA/IDAuMDEqbmV0Q2hpbGRJbnB1dFN1bUIgOiBuZXRDaGlsZElucHV0U3VtQjtcXG4gICAgICAgICAgICAgICAgICAgIGZvdXRwdXRDID0gKG5ldENoaWxkSW5wdXRTdW1DIDw9IDAuMCkgPyAwLjAxKm5ldENoaWxkSW5wdXRTdW1DIDogbmV0Q2hpbGRJbnB1dFN1bUM7XFxuICAgICAgICAgICAgICAgICAgICBmb3V0cHV0RCA9IChuZXRDaGlsZElucHV0U3VtRCA8PSAwLjApID8gMC4wMSpuZXRDaGlsZElucHV0U3VtRCA6IG5ldENoaWxkSW5wdXRTdW1EO1xcbiAgICAgICAgICAgICAgICAgICAgZm91dHB1dEUgPSAobmV0Q2hpbGRJbnB1dFN1bUUgPD0gMC4wKSA/IDAuMDEqbmV0Q2hpbGRJbnB1dFN1bUUgOiBuZXRDaGlsZElucHV0U3VtRTtcXG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcXG4gICAgICAgICAgICAgICAgICAgIGZvdXRwdXRBID0gMS4wO1xcbiAgICAgICAgICAgICAgICAgICAgZm91dHB1dEIgPSAxLjA7XFxuICAgICAgICAgICAgICAgICAgICBmb3V0cHV0QyA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGZvdXRwdXREID0gMS4wO1xcbiAgICAgICAgICAgICAgICAgICAgZm91dHB1dEUgPSAxLjA7XFxuICAgICAgICAgICAgICAgIH1cXG4gICAgICAgICAgICB9XCI7XG5cbiAgICAgICAgICAgIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgICAgICAgICAgIC8vIEVSUk9SXG4gICAgICAgICAgICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4gICAgICAgICAgICBmb3IgKHZhciBuID0gZWZmZXJlbnRTdGFydDsgbiA8IGVmZmVyZW50U3RhcnQgKyBlZmZlcmVudE5vZGVzQ291bnQ7IG4rKykge1xuICAgICAgICAgICAgICAgIHZhciBjb25kID0gbiA9PT0gZWZmZXJlbnRTdGFydCA/IFwiaWZcIiA6IFwiZWxzZSBpZlwiO1xuICAgICAgICAgICAgICAgIHN0ciArPSBcIlxcbiAgICAgICAgICAgIFwiICsgY29uZCArIFwiKG5vZGVJZCA9PSBcIiArIG4udG9GaXhlZCgxKSArIChcIikge1xcbiAgICAgICAgICAgICAgICBmb3V0cHV0QSA9IG5ldENoaWxkSW5wdXRTdW1BOyBcIiArIFwiIFxcbiAgICAgICAgICAgICAgICBmb3V0cHV0QiA9IG5ldENoaWxkSW5wdXRTdW1CO1xcbiAgICAgICAgICAgICAgICBmb3V0cHV0QyA9IG5ldENoaWxkSW5wdXRTdW1DO1xcbiAgICAgICAgICAgICAgICBmb3V0cHV0RCA9IG5ldENoaWxkSW5wdXRTdW1EO1xcbiAgICAgICAgICAgICAgICBmb3V0cHV0RSA9IG5ldENoaWxkSW5wdXRTdW1FO1xcbiAgICAgICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0QSA9IGVmZmVyZW50Tm9kZXNBW1wiKSArIE1hdGgucm91bmQobiAtIGVmZmVyZW50U3RhcnQpICsgXCJdO1xcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEIgPSBlZmZlcmVudE5vZGVzQltcIiArIE1hdGgucm91bmQobiAtIGVmZmVyZW50U3RhcnQpICsgXCJdO1xcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEMgPSBlZmZlcmVudE5vZGVzQ1tcIiArIE1hdGgucm91bmQobiAtIGVmZmVyZW50U3RhcnQpICsgXCJdO1xcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEQgPSBlZmZlcmVudE5vZGVzRFtcIiArIE1hdGgucm91bmQobiAtIGVmZmVyZW50U3RhcnQpICsgXCJdO1xcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEUgPSBlZmZlcmVudE5vZGVzRVtcIiArIE1hdGgucm91bmQobiAtIGVmZmVyZW50U3RhcnQpICsgXCJdO1xcbiAgICAgICAgICAgIH1cIjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIC8qc3RyICs9IGBcclxuICAgICAgICAgICAgZWxzZSB7XHJcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEEgKj0gZGF0YUJbeEdlb21ldHJ5XS56O1xyXG4gICAgICAgICAgICAgICAgbmV0UGFyZW50RXJyb3JXZWlnaHRCICo9IGRhdGFGW3hHZW9tZXRyeV0ueTtcclxuICAgICAgICAgICAgICAgIG5ldFBhcmVudEVycm9yV2VpZ2h0QyAqPSBkYXRhR1t4R2VvbWV0cnldLng7XHJcbiAgICAgICAgICAgICAgICBuZXRQYXJlbnRFcnJvcldlaWdodEQgKj0gZGF0YUdbeEdlb21ldHJ5XS53O1xyXG4gICAgICAgICAgICAgICAgbmV0UGFyZW50RXJyb3JXZWlnaHRFICo9IGRhdGFIW3hHZW9tZXRyeV0uejtcclxuICAgICAgICAgICAgfWA7Ki9cbiAgICAgICAgICAgIHJldHVybiBzdHI7XG4gICAgICAgIH1cbiAgICB9XSk7XG5cbiAgICByZXR1cm4gS0VSTkVMX0RJUjtcbn0oKTtcblxuZ2xvYmFsLktFUk5FTF9ESVIgPSBLRVJORUxfRElSO1xubW9kdWxlLmV4cG9ydHMuS0VSTkVMX0RJUiA9IEtFUk5FTF9ESVI7Il19
