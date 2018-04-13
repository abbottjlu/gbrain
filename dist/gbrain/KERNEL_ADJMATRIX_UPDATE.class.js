(function(){function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s}return e})()({1:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var KERNEL_ADJMATRIX_UPDATE = exports.KERNEL_ADJMATRIX_UPDATE = function () {
    function KERNEL_ADJMATRIX_UPDATE() {
        _classCallCheck(this, KERNEL_ADJMATRIX_UPDATE);
    }

    _createClass(KERNEL_ADJMATRIX_UPDATE, null, [{
        key: "getSrc",
        value: function getSrc(geometryLength) {
            return ["x", ["adjacencyMatrix"],
            // head
            "",

            // source
            "vec4 adjMat = adjacencyMatrix[x]; \n            vec4 adjMatB = adjacencyMatrixB[x];\n\n            float linkLayerNum = adjMat.x;\n            float linkWeight = adjMat.z;\n            float linkTypeParent = adjMat.w;\n            \n            if(linkTypeParent == 0.5 && linkLayerNum > 0.0) {\n                float id = adjMatB.z;\n                float idInv = adjMatB.w;\n            \n                vec2 xGeometryCurrentChild = get_global_id(id, bufferNodesWidth, " + geometryLength.toFixed(1) + ");\n                vec2 xGeometryParent = get_global_id(idInv, bufferNodesWidth, " + geometryLength.toFixed(1) + ");\n\n                float childGOutputA = dataB[xGeometryCurrentChild].z;\n                float parentGErrorA = dataB[xGeometryParent].w;\n                \n                float childGOutputB = dataF[xGeometryCurrentChild].x;\n                float parentGErrorB = dataF[xGeometryParent].y;\n                \n                float childGOutputC = dataF[xGeometryCurrentChild].z;\n                float parentGErrorC = dataF[xGeometryParent].w;\n                \n                float childGOutputD = dataG[xGeometryCurrentChild].x;\n                float parentGErrorD = dataG[xGeometryParent].y;\n                \n                float childGOutputE = dataG[xGeometryCurrentChild].z;\n                float parentGErrorE = dataG[xGeometryParent].w;\n                \n                float childGOutputF = dataH[xGeometryCurrentChild].x;\n                float parentGErrorF = dataH[xGeometryParent].y;\n                \n                float childGOutputG = dataH[xGeometryCurrentChild].z;\n                float parentGErrorG = dataH[xGeometryParent].w;\n            \n                float lr = learningRate;\n                float l2_decay = 0.01;\n                float gpu_batch_size = 7.0;\n                float br = gpu_batch_repeats;\n                \n                float derivA = (childGOutputA < 0.0) ? 0.01 : 1.0;\n                float derivB = (childGOutputB < 0.0) ? 0.01 : 1.0;\n                float derivC = (childGOutputC < 0.0) ? 0.01 : 1.0;\n                float derivD = (childGOutputD < 0.0) ? 0.01 : 1.0;\n                float derivE = (childGOutputE < 0.0) ? 0.01 : 1.0;\n                float derivF = (childGOutputF < 0.0) ? 0.01 : 1.0;\n                float derivG = (childGOutputG < 0.0) ? 0.01 : 1.0;\n                \n                if(linkLayerNum == layerCount-1.0) {\n                    derivA = 1.0;\n                    derivB = 1.0;\n                    derivC = 1.0;\n                    derivD = 1.0;\n                    derivE = 1.0;\n                    derivF = 1.0;\n                    derivG = 1.0;\n                }\n                \n                float bsm = 0.0;\n                bsm = (childGOutputA != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputB != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputC != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputD != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputE != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputF != 0.0) ? bsm+1. : bsm;\n                bsm = (childGOutputG != 0.0) ? bsm+1. : bsm;\n                \n                linkWeight += (-lr*parentGErrorA*derivA)/(bsm*br);\n                if(parentGErrorB != 0.0) {linkWeight += (-lr*parentGErrorB*derivB)/(bsm*br);}\n                if(parentGErrorC != 0.0) {linkWeight += (-lr*parentGErrorC*derivC)/(bsm*br);}\n                if(parentGErrorD != 0.0) {linkWeight += (-lr*parentGErrorD*derivD)/(bsm*br);}\n                if(parentGErrorE != 0.0) {linkWeight += (-lr*parentGErrorE*derivE)/(bsm*br);}\n                if(parentGErrorF != 0.0) {linkWeight += (-lr*parentGErrorF*derivF)/(bsm*br);}\n                if(parentGErrorG != 0.0) {linkWeight += (-lr*parentGErrorG*derivG)/(bsm*br);}\n            }\n            \n            return [vec4(linkLayerNum, 0.0, linkWeight, linkTypeParent)];\n            "];
        }
    }]);

    return KERNEL_ADJMATRIX_UPDATE;
}();

global.KERNEL_ADJMATRIX_UPDATE = KERNEL_ADJMATRIX_UPDATE;
module.exports.KERNEL_ADJMATRIX_UPDATE = KERNEL_ADJMATRIX_UPDATE;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})

},{}]},{},[1])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvZ2JyYWluL0tFUk5FTF9BREpNQVRSSVhfVVBEQVRFLmNsYXNzLmpzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOztBQ0FBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EiLCJmaWxlIjoiZ2VuZXJhdGVkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXNDb250ZW50IjpbIihmdW5jdGlvbigpe2Z1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfXJldHVybiBlfSkoKSIsIlwidXNlIHN0cmljdFwiO1xuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHtcbiAgICB2YWx1ZTogdHJ1ZVxufSk7XG5cbnZhciBfY3JlYXRlQ2xhc3MgPSBmdW5jdGlvbiAoKSB7IGZ1bmN0aW9uIGRlZmluZVByb3BlcnRpZXModGFyZ2V0LCBwcm9wcykgeyBmb3IgKHZhciBpID0gMDsgaSA8IHByb3BzLmxlbmd0aDsgaSsrKSB7IHZhciBkZXNjcmlwdG9yID0gcHJvcHNbaV07IGRlc2NyaXB0b3IuZW51bWVyYWJsZSA9IGRlc2NyaXB0b3IuZW51bWVyYWJsZSB8fCBmYWxzZTsgZGVzY3JpcHRvci5jb25maWd1cmFibGUgPSB0cnVlOyBpZiAoXCJ2YWx1ZVwiIGluIGRlc2NyaXB0b3IpIGRlc2NyaXB0b3Iud3JpdGFibGUgPSB0cnVlOyBPYmplY3QuZGVmaW5lUHJvcGVydHkodGFyZ2V0LCBkZXNjcmlwdG9yLmtleSwgZGVzY3JpcHRvcik7IH0gfSByZXR1cm4gZnVuY3Rpb24gKENvbnN0cnVjdG9yLCBwcm90b1Byb3BzLCBzdGF0aWNQcm9wcykgeyBpZiAocHJvdG9Qcm9wcykgZGVmaW5lUHJvcGVydGllcyhDb25zdHJ1Y3Rvci5wcm90b3R5cGUsIHByb3RvUHJvcHMpOyBpZiAoc3RhdGljUHJvcHMpIGRlZmluZVByb3BlcnRpZXMoQ29uc3RydWN0b3IsIHN0YXRpY1Byb3BzKTsgcmV0dXJuIENvbnN0cnVjdG9yOyB9OyB9KCk7XG5cbmZ1bmN0aW9uIF9jbGFzc0NhbGxDaGVjayhpbnN0YW5jZSwgQ29uc3RydWN0b3IpIHsgaWYgKCEoaW5zdGFuY2UgaW5zdGFuY2VvZiBDb25zdHJ1Y3RvcikpIHsgdGhyb3cgbmV3IFR5cGVFcnJvcihcIkNhbm5vdCBjYWxsIGEgY2xhc3MgYXMgYSBmdW5jdGlvblwiKTsgfSB9XG5cbnZhciBLRVJORUxfQURKTUFUUklYX1VQREFURSA9IGV4cG9ydHMuS0VSTkVMX0FESk1BVFJJWF9VUERBVEUgPSBmdW5jdGlvbiAoKSB7XG4gICAgZnVuY3Rpb24gS0VSTkVMX0FESk1BVFJJWF9VUERBVEUoKSB7XG4gICAgICAgIF9jbGFzc0NhbGxDaGVjayh0aGlzLCBLRVJORUxfQURKTUFUUklYX1VQREFURSk7XG4gICAgfVxuXG4gICAgX2NyZWF0ZUNsYXNzKEtFUk5FTF9BREpNQVRSSVhfVVBEQVRFLCBudWxsLCBbe1xuICAgICAgICBrZXk6IFwiZ2V0U3JjXCIsXG4gICAgICAgIHZhbHVlOiBmdW5jdGlvbiBnZXRTcmMoZ2VvbWV0cnlMZW5ndGgpIHtcbiAgICAgICAgICAgIHJldHVybiBbXCJ4XCIsIFtcImFkamFjZW5jeU1hdHJpeFwiXSxcbiAgICAgICAgICAgIC8vIGhlYWRcbiAgICAgICAgICAgIFwiXCIsXG5cbiAgICAgICAgICAgIC8vIHNvdXJjZVxuICAgICAgICAgICAgXCJ2ZWM0IGFkak1hdCA9IGFkamFjZW5jeU1hdHJpeFt4XTsgXFxuICAgICAgICAgICAgdmVjNCBhZGpNYXRCID0gYWRqYWNlbmN5TWF0cml4Qlt4XTtcXG5cXG4gICAgICAgICAgICBmbG9hdCBsaW5rTGF5ZXJOdW0gPSBhZGpNYXQueDtcXG4gICAgICAgICAgICBmbG9hdCBsaW5rV2VpZ2h0ID0gYWRqTWF0Lno7XFxuICAgICAgICAgICAgZmxvYXQgbGlua1R5cGVQYXJlbnQgPSBhZGpNYXQudztcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICBpZihsaW5rVHlwZVBhcmVudCA9PSAwLjUgJiYgbGlua0xheWVyTnVtID4gMC4wKSB7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGlkID0gYWRqTWF0Qi56O1xcbiAgICAgICAgICAgICAgICBmbG9hdCBpZEludiA9IGFkak1hdEIudztcXG4gICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgdmVjMiB4R2VvbWV0cnlDdXJyZW50Q2hpbGQgPSBnZXRfZ2xvYmFsX2lkKGlkLCBidWZmZXJOb2Rlc1dpZHRoLCBcIiArIGdlb21ldHJ5TGVuZ3RoLnRvRml4ZWQoMSkgKyBcIik7XFxuICAgICAgICAgICAgICAgIHZlYzIgeEdlb21ldHJ5UGFyZW50ID0gZ2V0X2dsb2JhbF9pZChpZEludiwgYnVmZmVyTm9kZXNXaWR0aCwgXCIgKyBnZW9tZXRyeUxlbmd0aC50b0ZpeGVkKDEpICsgXCIpO1xcblxcbiAgICAgICAgICAgICAgICBmbG9hdCBjaGlsZEdPdXRwdXRBID0gZGF0YUJbeEdlb21ldHJ5Q3VycmVudENoaWxkXS56O1xcbiAgICAgICAgICAgICAgICBmbG9hdCBwYXJlbnRHRXJyb3JBID0gZGF0YUJbeEdlb21ldHJ5UGFyZW50XS53O1xcbiAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgZmxvYXQgY2hpbGRHT3V0cHV0QiA9IGRhdGFGW3hHZW9tZXRyeUN1cnJlbnRDaGlsZF0ueDtcXG4gICAgICAgICAgICAgICAgZmxvYXQgcGFyZW50R0Vycm9yQiA9IGRhdGFGW3hHZW9tZXRyeVBhcmVudF0ueTtcXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIGZsb2F0IGNoaWxkR091dHB1dEMgPSBkYXRhRlt4R2VvbWV0cnlDdXJyZW50Q2hpbGRdLno7XFxuICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdFcnJvckMgPSBkYXRhRlt4R2VvbWV0cnlQYXJlbnRdLnc7XFxuICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICBmbG9hdCBjaGlsZEdPdXRwdXREID0gZGF0YUdbeEdlb21ldHJ5Q3VycmVudENoaWxkXS54O1xcbiAgICAgICAgICAgICAgICBmbG9hdCBwYXJlbnRHRXJyb3JEID0gZGF0YUdbeEdlb21ldHJ5UGFyZW50XS55O1xcbiAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgZmxvYXQgY2hpbGRHT3V0cHV0RSA9IGRhdGFHW3hHZW9tZXRyeUN1cnJlbnRDaGlsZF0uejtcXG4gICAgICAgICAgICAgICAgZmxvYXQgcGFyZW50R0Vycm9yRSA9IGRhdGFHW3hHZW9tZXRyeVBhcmVudF0udztcXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIGZsb2F0IGNoaWxkR091dHB1dEYgPSBkYXRhSFt4R2VvbWV0cnlDdXJyZW50Q2hpbGRdLng7XFxuICAgICAgICAgICAgICAgIGZsb2F0IHBhcmVudEdFcnJvckYgPSBkYXRhSFt4R2VvbWV0cnlQYXJlbnRdLnk7XFxuICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICBmbG9hdCBjaGlsZEdPdXRwdXRHID0gZGF0YUhbeEdlb21ldHJ5Q3VycmVudENoaWxkXS56O1xcbiAgICAgICAgICAgICAgICBmbG9hdCBwYXJlbnRHRXJyb3JHID0gZGF0YUhbeEdlb21ldHJ5UGFyZW50XS53O1xcbiAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICBmbG9hdCBsciA9IGxlYXJuaW5nUmF0ZTtcXG4gICAgICAgICAgICAgICAgZmxvYXQgbDJfZGVjYXkgPSAwLjAxO1xcbiAgICAgICAgICAgICAgICBmbG9hdCBncHVfYmF0Y2hfc2l6ZSA9IDcuMDtcXG4gICAgICAgICAgICAgICAgZmxvYXQgYnIgPSBncHVfYmF0Y2hfcmVwZWF0cztcXG4gICAgICAgICAgICAgICAgXFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2QSA9IChjaGlsZEdPdXRwdXRBIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2QiA9IChjaGlsZEdPdXRwdXRCIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2QyA9IChjaGlsZEdPdXRwdXRDIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2RCA9IChjaGlsZEdPdXRwdXREIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2RSA9IChjaGlsZEdPdXRwdXRFIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2RiA9IChjaGlsZEdPdXRwdXRGIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIGZsb2F0IGRlcml2RyA9IChjaGlsZEdPdXRwdXRHIDwgMC4wKSA/IDAuMDEgOiAxLjA7XFxuICAgICAgICAgICAgICAgIFxcbiAgICAgICAgICAgICAgICBpZihsaW5rTGF5ZXJOdW0gPT0gbGF5ZXJDb3VudC0xLjApIHtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2QSA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2QiA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2QyA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2RCA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2RSA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2RiA9IDEuMDtcXG4gICAgICAgICAgICAgICAgICAgIGRlcml2RyA9IDEuMDtcXG4gICAgICAgICAgICAgICAgfVxcbiAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgZmxvYXQgYnNtID0gMC4wO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0QSAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0QiAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0QyAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0RCAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0RSAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0RiAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBic20gPSAoY2hpbGRHT3V0cHV0RyAhPSAwLjApID8gYnNtKzEuIDogYnNtO1xcbiAgICAgICAgICAgICAgICBcXG4gICAgICAgICAgICAgICAgbGlua1dlaWdodCArPSAoLWxyKnBhcmVudEdFcnJvckEqZGVyaXZBKS8oYnNtKmJyKTtcXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yQiAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yQipkZXJpdkIpLyhic20qYnIpO31cXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yQyAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yQypkZXJpdkMpLyhic20qYnIpO31cXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yRCAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yRCpkZXJpdkQpLyhic20qYnIpO31cXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yRSAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yRSpkZXJpdkUpLyhic20qYnIpO31cXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yRiAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yRipkZXJpdkYpLyhic20qYnIpO31cXG4gICAgICAgICAgICAgICAgaWYocGFyZW50R0Vycm9yRyAhPSAwLjApIHtsaW5rV2VpZ2h0ICs9ICgtbHIqcGFyZW50R0Vycm9yRypkZXJpdkcpLyhic20qYnIpO31cXG4gICAgICAgICAgICB9XFxuICAgICAgICAgICAgXFxuICAgICAgICAgICAgcmV0dXJuIFt2ZWM0KGxpbmtMYXllck51bSwgMC4wLCBsaW5rV2VpZ2h0LCBsaW5rVHlwZVBhcmVudCldO1xcbiAgICAgICAgICAgIFwiXTtcbiAgICAgICAgfVxuICAgIH1dKTtcblxuICAgIHJldHVybiBLRVJORUxfQURKTUFUUklYX1VQREFURTtcbn0oKTtcblxuZ2xvYmFsLktFUk5FTF9BREpNQVRSSVhfVVBEQVRFID0gS0VSTkVMX0FESk1BVFJJWF9VUERBVEU7XG5tb2R1bGUuZXhwb3J0cy5LRVJORUxfQURKTUFUUklYX1VQREFURSA9IEtFUk5FTF9BREpNQVRSSVhfVVBEQVRFOyJdfQ==
