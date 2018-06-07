import "scejs";

export class VFP_NODE {
    constructor() {

    }

	static getSrc(customCode, geometryLength) {
        return [["RGB"],

            //██╗   ██╗███████╗██████╗ ████████╗███████╗██╗  ██╗    ██╗  ██╗███████╗ █████╗ ██████╗
            //██║   ██║██╔════╝██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██║  ██║██╔════╝██╔══██╗██╔══██╗
            //██║   ██║█████╗  ██████╔╝   ██║   █████╗   ╚███╔╝     ███████║█████╗  ███████║██║  ██║
            //╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██╔══╝   ██╔██╗     ██╔══██║██╔══╝  ██╔══██║██║  ██║
            // ╚████╔╝ ███████╗██║  ██║   ██║   ███████╗██╔╝ ██╗    ██║  ██║███████╗██║  ██║██████╔╝
            //  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝
       		'varying vec4 vVertexColor;\n'+
            'varying vec4 vVertexColorNetError;\n'+
            'varying vec4 vUV;\n'+
       		'varying vec2 vVertexUV;\n'+
       		'varying float vUseTex;\n'+
       		'varying vec4 vWNMatrix;\n'+
            'varying float vNodeId;\n'+
            'varying float vNodeIdOpposite;\n'+
       		'varying float vDist;\n'+
       		'varying float vIsSelected;\n'+
            'varying float vIsHover;\n'+
       		'varying float vUseCrosshair;\n'+
       		'varying float vIstarget;\n'+
       		
       		'vec2 get2Dfrom1D(float idx, float columns) {'+
       			'float n = idx/columns;'+
       			'float row = float(int(n));'+
       			'float col = fract(n)*columns;'+

       			'float ts = 1.0/columns;'+
       			'return vec2(ts*col, ts*row);'+
       		'}'+
       		`float getDigit(float planeId, float value) {           		
       		    if(value >= 0.0 && planeId == 1.0) return 39.0;
       		    
           		if(value < 0.0 && planeId == 0.0) return 38.0;
       		    if(value < 0.0 && planeId == 2.0) return 39.0;
       		     
       		    float val = value;                         
                float selectedDigit = 0.0;
                for(int n=0; n <= 12; n++) {
                    float num = floor(abs(val));
                    if(num == 0.0) {                                
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                        }
                    } else if(num == 1.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -10.0 : 10.0;
                        }                    
                    } else if(num == 2.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -20.0 : 20.0;
                        }                   
                    } else if(num == 3.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -30.0 : 30.0;
                        }                    
                    } else if(num == 4.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -40.0 : 40.0;
                        }                     
                    } else if(num == 5.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -50.0 : 50.0;
                        }                    
                    } else if(num == 6.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -60.0 : 60.0;
                        }                   
                    } else if(num == 7.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -70.0 : 70.0;
                        }                  
                    } else if(num == 8.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -80.0 : 80.0;
                        }                 
                    } else if(num == 9.0) {
                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {
                            selectedDigit = num; break;
                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {
                            selectedDigit = num; break;
                        } else {
                            val *= 10.0;
                            val += (val >= 0.0) ? -90.0 : 90.0;
                        }               
                    }
                }
                selectedDigit = abs(floor(selectedDigit));
                if(selectedDigit == 0.0) {
                    return 28.0;
                } else if(selectedDigit == 1.0) {
                    return 29.0;
                } else if(selectedDigit == 2.0) {
                    return 30.0;
                } else if(selectedDigit == 3.0) {
                    return 31.0;
                } else if(selectedDigit == 4.0) {
                    return 32.0;
                } else if(selectedDigit == 5.0) {
                    return 33.0;
                } else if(selectedDigit == 6.0) {
                    return 34.0;
                } else if(selectedDigit == 7.0) {
                    return 35.0;
                } else if(selectedDigit == 8.0) {
                    return 36.0;
                } else if(selectedDigit == 9.0) {
                    return 37.0;
                }
            }`+
       		 'mat4 lookAt(vec3 eye, vec3 center, vec3 up) {'+
       		     'vec3 zaxis = normalize(center - eye);'+
       		     'vec3 xaxis = normalize(cross(up, zaxis));'+
       		     'vec3 yaxis = cross(zaxis, xaxis);'+

       		     'mat4 matrix;'+
       		     //Column Major
       		     'matrix[0][0] = xaxis.x;'+
       		     'matrix[1][0] = yaxis.x;'+
       		     'matrix[2][0] = zaxis.x;'+
       		     'matrix[3][0] = 0.0;'+

       		     'matrix[0][1] = xaxis.y;'+
       		     'matrix[1][1] = yaxis.y;'+
       		     'matrix[2][1] = zaxis.y;'+
       		     'matrix[3][1] = 0.0;'+

       		     'matrix[0][2] = xaxis.z;'+
       		     'matrix[1][2] = yaxis.z;'+
       		     'matrix[2][2] = zaxis.z;'+
       		     'matrix[3][2] = 0.0;'+

       		     'matrix[0][3] = -dot(xaxis, eye);'+
       		     'matrix[1][3] = -dot(yaxis, eye);'+
       		     'matrix[2][3] = -dot(zaxis, eye);'+
       		     'matrix[3][3] = 1.0;'+

       		     'return matrix;'+
       		 '}'+
       		 'mat4 transpose(mat4 m) {'+
       			  'return mat4(  m[0][0], m[1][0], m[2][0], m[3][0],'+
       			                'm[0][1], m[1][1], m[2][1], m[3][1],'+
       			  				'm[0][2], m[1][2], m[2][2], m[3][2],'+
       			  				'm[0][3], m[1][3], m[2][3], m[3][3]);'+
       		'}'+
       		'mat4 rotationMatrix(vec3 axis, float angle) {'+
       			'axis = normalize(axis);'+
       			'float s = sin(angle);'+
       			'float c = cos(angle);'+
       			'float oc = 1.0 - c;'+

       			'return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,'+
       			'oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,'+
       			'oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,'+
       			'0.0,                                0.0,                                0.0,                                1.0);'+
       		'}'+

            Utils.degToRadGLSLFunctionString()+
            Utils.radToDegGLSLFunctionString()+
            Utils.cartesianToSphericalGLSLFunctionString()+
            Utils.sphericalToCartesianGLSLFunctionString()+
            Utils.getVectorGLSLFunctionString(),



            //██╗   ██╗███████╗██████╗ ████████╗███████╗██╗  ██╗    ███████╗ ██████╗ ██╗   ██╗██████╗  ██████╗███████╗
            //██║   ██║██╔════╝██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝
            //██║   ██║█████╗  ██████╔╝   ██║   █████╗   ╚███╔╝     ███████╗██║   ██║██║   ██║██████╔╝██║     █████╗
            //╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██╔══╝   ██╔██╗     ╚════██║██║   ██║██║   ██║██╔══██╗██║     ██╔══╝
            // ╚████╔╝ ███████╗██║  ██║   ██║   ███████╗██╔╝ ██╗    ███████║╚██████╔╝╚██████╔╝██║  ██║╚██████╗███████╗
            //  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚══════╝
            `
            mat4 nodepos = nodeWMatrix;
            
            vec4 nodeVertexPosition = nodeVertexPos[];
            vec4 nodeVertexTex = nodeVertexTexture[];
            
            float nodeId = data[].x;            
            vec2 xGeometryNode = get_global_id(nodeId, bufferNodesWidth, `+geometryLength.toFixed(1)+`); ${/* bufferWidth6, geometryLength */''}

            vec4 currentPosition = posXYZW[xGeometryNode];
            vec3 oppositePosition = vec3(0.0, 0.0, 0.0);
            

            float currentLineVertex = data[].z; ${/* this is isTarget for arrows */''}
            float vertexCount = 1.0;
                
            vec4 nodeVertexColor = vec4(1.0, 1.0, 1.0, 1.0);
            vec4 nodeVertexColorNetError = vec4(1.0, 1.0, 1.0, 1.0);
            vVertexUV = vec2(-1.0, -1.0);

            if(isNode == 1.0) {
                float foutput = dataB[xGeometryNode].z;
                float error = dataB[xGeometryNode].y;
            
                currentPosition += vec4(0.0, 0.1, 0.0, 1.0);

                mat4 mm = rotationMatrix(vec3(1.0,0.0,0.0), (3.1416/2.0)*3.0);
                nodepos = nodepos*mm;
                float nodeImgId = nodeImgId[];

                if(nodeImgId != -1.0) {
                    vUseTex = 1.0;
                    vVertexUV = get2Dfrom1D(nodeImgId, nodeImgColumns)+vec2(nodeVertexTexture.x/nodeImgColumns,nodeVertexTexture.y/nodeImgColumns);
                }

                nodeVertexColor = vec4(0.0, 0.0, 0.0, 1.0);
                if(foutput > 0.0) {
                    nodeVertexColor = vec4(abs(foutput), abs(foutput), abs(foutput), 1.0);
                } else if(foutput < 0.0) {
                    nodeVertexColor = vec4(abs(foutput), 0.0, 0.0, 1.0);
                }

                if(error > 0.0) {
                    nodeVertexColorNetError = vec4(0.0, abs(error), 0.0, 1.0);
                } else if(error < 0.0) {
                    nodeVertexColorNetError = vec4(abs(error), 0.0, 0.0, 1.0);
                } else {
                    nodeVertexColorNetError = vec4(0.0,0.0,0.0, 0.0);
                }

                vIsSelected = (idToDrag == data[].x) ? 1.0 : 0.0;
                vIsHover = (idToHover == data[].x) ? 1.0 : 0.0;
            }
            if(isLink == 1.0) {        
                float foutput = dataB[xGeometryNode].z;
                float error = dataB[xGeometryNode].w;
                
                float nodeIdOpposite = data[].y;
                
                vec2 xGeometryNode_opposite = get_global_id(nodeIdOpposite, bufferNodesWidth, `+geometryLength.toFixed(1)+`);
                vec3 oppositePosition = posXYZW[xGeometryNode_opposite].xyz;
                
                ${VFP_NODE.linkStr()}
            
                currentPosition += vec4(dirN*(lineIncrements*currentLineVertex), 1.0); ${/* displacing from center to first point */''}

                vIsSelected = (idToDrag == data[].x || idToDrag == data[].y) ? 1.0 : 0.0;
                vIsHover = (idToHover == data[].x || idToHover == data[].y) ? 1.0 : 0.0;
                
                nodeVertexColor = vec4(abs(foutput), abs(foutput), abs(foutput), 1.0);

                if(error > 0.0) {
                    nodeVertexColorNetError = vec4(0.0, abs(error), 0.0, 1.0);
                } else if(error < 0.0) {
                    nodeVertexColorNetError = vec4(abs(error), 0.0, 0.0, 1.0);
                } else {
                    nodeVertexColorNetError = vec4(0.0,0.0,0.0, 0.0);
                }
                        
                vNodeIdOpposite = nodeIdOpposite;
            }
            if(isArrow == 1.0) {
                float nodeIdOpposite = data[].y;
                
                vec2 xGeometryNode_opposite = get_global_id(nodeIdOpposite, bufferNodesWidth, `+geometryLength.toFixed(1)+`);
                vec3 oppositePosition = posXYZW[xGeometryNode_opposite].xyz;
                
                ${VFP_NODE.linkStr()}                
            
                vec3 currentPositionTMP;
                ${/* displacing from center to first point */''}
                float currentLineVertexU = vertexCount-1.0;
                currentPositionTMP = oppositePosition+((dirN*-1.0)*(lineIncrements*currentLineVertexU));

                mat4 pp = lookAt(currentPositionTMP, vec3(currentPosition.x, currentPosition.y, currentPosition.z), vec3(0.0, 1.0, 0.0));
                pp = transpose(pp);
                nodepos[0][0] = pp[0][0];
                nodepos[0][1] = pp[1][0];
                nodepos[0][2] = pp[2][0];

                nodepos[1][0] = pp[0][1];
                nodepos[1][1] = pp[1][1];
                nodepos[1][2] = pp[2][1];

                nodepos[2][0] = pp[0][2];
                nodepos[2][1] = pp[1][2];
                nodepos[2][2] = pp[2][2];

                ${/* displace from center to node border */''}
                dir = currentPositionTMP-vec3(currentPosition.x, currentPosition.y, currentPosition.z);
                currentPosition += vec4(normalize(dir),1.0)*2.0;

                vIsSelected = (idToDrag == data[].x || idToDrag == data[].y) ? 1.0 : 0.0;
                vIsHover = (idToHover == data[].x || idToHover == data[].y) ? 1.0 : 0.0;
                
                vNodeIdOpposite = nodeIdOpposite;
            }
            if(isNodeText == 1.0) {
                float foutput = dataB[xGeometryNode].z;
                
                float letId = (showValues == 1.0) ? getDigit(data[].y, foutput) : letterId[];
                mat4 mm = rotationMatrix(vec3(1.0,0.0,0.0), (3.1416/2.0)*3.0);
                nodepos = nodepos*mm;

                vVertexUV = get2Dfrom1D(letId, fontImgColumns)+vec2(nodeVertexTexture.x/fontImgColumns,nodeVertexTexture.y/fontImgColumns);
                nodeVertexPosition = vec4(nodeVertexPosition.x*0.1, nodeVertexPosition.y*0.1, nodeVertexPosition.z*0.1, 1.0);
                currentPosition.x += 3.5;
                currentPosition.y += 0.5;

                vIsSelected = (idToDrag == data[].x) ? 1.0 : 0.0;
                vIsHover = (idToHover == data[].x) ? 1.0 : 0.0;
                
                vIstarget = (currentLineVertex == 1.0) ? 1.0 : 0.0;
            }
            nodepos[3][0] = currentPosition.x;
            nodepos[3][1] = currentPosition.y;
            nodepos[3][2] = currentPosition.z;

            mat4 nodeposG = nodeWMatrix;
            vWNMatrix = nodeposG * nodeVertexNormal[];

            vUseCrosshair = 0.0;
            

            ${customCode}
            
            vDist = max(0.3, 1.0-(distance(currentPosition.xyz, oppositePosition)*0.01)); ${/* dist/acum */''}
            vVertexColor = nodeVertexColor;
            vVertexColorNetError = nodeVertexColorNetError;
            vUV = nodeVertexTexture;
            vNodeId = nodeId;


            gl_Position = PMatrix * cameraWMatrix * nodepos * nodeVertexPosition;`,


            //███████╗██████╗  █████╗  ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗    ██╗  ██╗███████╗ █████╗ ██████╗
            //██╔════╝██╔══██╗██╔══██╗██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝    ██║  ██║██╔════╝██╔══██╗██╔══██╗
            //█████╗  ██████╔╝███████║██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║       ███████║█████╗  ███████║██║  ██║
            //██╔══╝  ██╔══██╗██╔══██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║       ██╔══██║██╔══╝  ██╔══██║██║  ██║
            //██║     ██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║       ██║  ██║███████╗██║  ██║██████╔╝
            //╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝
       		`varying vec4 vVertexColor;
            varying vec4 vVertexColorNetError;
            varying vec4 vUV;
       		varying vec2 vVertexUV;
       		varying float vUseTex;
       		varying vec4 vWNMatrix;
            varying float vNodeId;
            varying float vNodeIdOpposite;
       		varying float vDist;
       		varying float vIsSelected;
            varying float vIsHover;
       		varying float vUseCrosshair;
       		varying float vIstarget;`,

            //███████╗██████╗  █████╗  ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗    ███████╗ ██████╗ ██╗   ██╗██████╗  ██████╗███████╗
            //██╔════╝██╔══██╗██╔══██╗██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝    ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝
            //█████╗  ██████╔╝███████║██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║       ███████╗██║   ██║██║   ██║██████╔╝██║     █████╗
            //██╔══╝  ██╔══██╗██╔══██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║       ╚════██║██║   ██║██║   ██║██╔══██╗██║     ██╔══╝
            //██║     ██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║       ███████║╚██████╔╝╚██████╔╝██║  ██║╚██████╗███████╗
            //╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚══════╝
            'vec4 color = vVertexColor;\n'+
            'vec4 colorB = vVertexColorNetError;\n'+
            'vec4 colorOrange = vec4(255.0/255.0, 131.0/255.0, 0.0/255.0, 1.0);'+
            'vec4 colorOrangeDark = vec4(255.0/255.0, 80.0/255.0, 0.0/255.0, 1.0);'+
            'vec4 colorPurple = vec4(132.0/255.0, 0.0/255.0, 255.0/255.0, 1.0);'+

            'vec4 fcolor = vec4(0.0, 0.0, 0.0, 1.0);'+
            'if(isNode == 1.0) {'+
                'if(vUseTex == 1.0) {'+
                    'vec4 tex;'+
                    'if(vUseCrosshair == 1.0) {'+
                        'tex = nodesImgCrosshair[vVertexUV.xy];'+
                    '} else if(vIsSelected == 1.0) {'+
                        'color = colorOrangeDark;'+
                        'tex = nodesImgCrosshair[vVertexUV.xy];'+
                    '} else if(vIsHover == 1.0) {'+
                        'color = colorPurple;'+
                        'tex = nodesImg[vVertexUV.xy];'+
                    '} else {'+
                        'tex = nodesImg[vVertexUV.xy];'+
                    '}'+
                    'color = '+VFP_NODE.nodesDrawMode(geometryLength)+';\n'+
                '}'+
                'if(color.a < 0.1) discard;'+

                'fcolor = color;\n'+
                // half up: ouput; half down: error
                'if(vUV.y > (0.7)) '+
                    'fcolor = colorB;\n'+ // 0.75-(colorB.w*1.0)   'fcolor = (vUV.y < (0.5)) ? color : ((colorB.w == 0.0)?color:colorB);\n'+
            '} else if(isLink == 1.0) {'+
                'if(vIsSelected == 1.0) '+
                    'color = colorOrange;'+
                'else if(vIsHover == 1.0) '+
                    'color = colorPurple;'+

                // weight color
                'vec2 xAdjMatCurrent = get_global_id(vec2(vNodeIdOpposite, vNodeId), widthAdjMatrix);'+
                'vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n'+

                // x weight
                'if(pixAdjMatACurrent.z > 0.0) '+
                    'fcolor = vec4(0.0, pixAdjMatACurrent.z, 0.0, abs(pixAdjMatACurrent.z));\n'+
                'else '+
                    'fcolor = vec4(abs(pixAdjMatACurrent.z), 0.0, 0.0, abs(pixAdjMatACurrent.z));\n'+

                // x output
                'if(multiplyOutput == 1.0) '+
                    'fcolor *= vec4(color.xyz, 1.0);\n'+
            '} else if(isArrow == 1.0) {'+
                'if(vIstarget == 1.0) {'+
                    'if(vIsSelected == 1.0) {'+
                        'color = vec4(colorOrange.rgb*vDist, 1.0);\n'+
                    '} else if(vIsHover == 1.0) {'+
                        'color = vec4(colorPurple.rgb*vDist, 1.0);\n'+
                    '}'+
                '} else {'+
                    'color = vec4(1.0, 0.0, 0.0, 0.0);'+
                '}'+
                'fcolor = color;\n'+
            '} else if(isNodeText == 1.0) {'+
                'fcolor = fontsImg[vVertexUV.xy];\n'+
            '}'+

            'return [fcolor];'
        ];
	};

	static linkStr() {
	    return `
	    vec3 dir = oppositePosition-vec3(currentPosition.x, currentPosition.y, currentPosition.z);
        float dist = length(dir);
        float lineIncrements = dist/vertexCount;
        vec3 dirN = normalize(dir);

        float repeatId = data[].w;
        vec3 cr = cross(dirN, vec3(0.0, 1.0, 0.0));
        float currentLineVertexSQRT = abs( currentLineVertex-(vertexCount/2.0) )/(vertexCount/2.0);
        currentLineVertexSQRT = sqrt(1.0-currentLineVertexSQRT);`;
    }

    static nodesDrawMode(geometryLength) {
        if(geometryLength === 1)
            return "vec4(color.rgb, 1.0)";
        else
            return "vec4(tex.rgb*color.rgb, tex.a)";
    }
}
global.VFP_NODE = VFP_NODE;
module.exports.VFP_NODE = VFP_NODE;