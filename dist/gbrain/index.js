(function(){function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s}return e})()({1:[function(require,module,exports){
/*!
 * Draggabilly v2.2.0
 * Make that shiz draggable
 * https://draggabilly.desandro.com
 * MIT license
 */

/*jshint browser: true, strict: true, undef: true, unused: true */

( function( window, factory ) {
  // universal module definition
  /* jshint strict: false */ /*globals define, module, require */
  if ( typeof define == 'function' && define.amd ) {
    // AMD
    define( [
        'get-size/get-size',
        'unidragger/unidragger'
      ],
      function( getSize, Unidragger ) {
        return factory( window, getSize, Unidragger );
      });
  } else if ( typeof module == 'object' && module.exports ) {
    // CommonJS
    module.exports = factory(
      window,
      require('get-size'),
      require('unidragger')
    );
  } else {
    // browser global
    window.Draggabilly = factory(
      window,
      window.getSize,
      window.Unidragger
    );
  }

}( window, function factory( window, getSize, Unidragger ) {

'use strict';

// -------------------------- helpers & variables -------------------------- //

// extend objects
function extend( a, b ) {
  for ( var prop in b ) {
    a[ prop ] = b[ prop ];
  }
  return a;
}

function noop() {}

var jQuery = window.jQuery;

// --------------------------  -------------------------- //

function Draggabilly( element, options ) {
  // querySelector if string
  this.element = typeof element == 'string' ?
    document.querySelector( element ) : element;

  if ( jQuery ) {
    this.$element = jQuery( this.element );
  }

  // options
  this.options = extend( {}, this.constructor.defaults );
  this.option( options );

  this._create();
}

// inherit Unidragger methods
var proto = Draggabilly.prototype = Object.create( Unidragger.prototype );

Draggabilly.defaults = {
};

/**
 * set options
 * @param {Object} opts
 */
proto.option = function( opts ) {
  extend( this.options, opts );
};

// css position values that don't need to be set
var positionValues = {
  relative: true,
  absolute: true,
  fixed: true
};

proto._create = function() {
  // properties
  this.position = {};
  this._getPosition();

  this.startPoint = { x: 0, y: 0 };
  this.dragPoint = { x: 0, y: 0 };

  this.startPosition = extend( {}, this.position );

  // set relative positioning
  var style = getComputedStyle( this.element );
  if ( !positionValues[ style.position ] ) {
    this.element.style.position = 'relative';
  }

  // events, bridge jQuery events from vanilla
  this.on( 'pointerDown', this.onPointerDown );
  this.on( 'pointerMove', this.onPointerMove );
  this.on( 'pointerUp', this.onPointerUp );

  this.enable();
  this.setHandles();
};

/**
 * set this.handles and bind start events to 'em
 */
proto.setHandles = function() {
  this.handles = this.options.handle ?
    this.element.querySelectorAll( this.options.handle ) : [ this.element ];

  this.bindHandles();
};

/**
 * emits events via EvEmitter and jQuery events
 * @param {String} type - name of event
 * @param {Event} event - original event
 * @param {Array} args - extra arguments
 */
proto.dispatchEvent = function( type, event, args ) {
  var emitArgs = [ event ].concat( args );
  this.emitEvent( type, emitArgs );
  this.dispatchJQueryEvent( type, event, args );
};

proto.dispatchJQueryEvent = function( type, event, args ) {
  var jQuery = window.jQuery;
  // trigger jQuery event
  if ( !jQuery || !this.$element ) {
    return;
  }
  // create jQuery event
  var $event = jQuery.Event( event );
  $event.type = type;
  this.$element.trigger( $event, args );
};

// -------------------------- position -------------------------- //

// get x/y position from style
proto._getPosition = function() {
  var style = getComputedStyle( this.element );
  var x = this._getPositionCoord( style.left, 'width' );
  var y = this._getPositionCoord( style.top, 'height' );
  // clean up 'auto' or other non-integer values
  this.position.x = isNaN( x ) ? 0 : x;
  this.position.y = isNaN( y ) ? 0 : y;

  this._addTransformPosition( style );
};

proto._getPositionCoord = function( styleSide, measure ) {
  if ( styleSide.indexOf('%') != -1 ) {
    // convert percent into pixel for Safari, #75
    var parentSize = getSize( this.element.parentNode );
    // prevent not-in-DOM element throwing bug, #131
    return !parentSize ? 0 :
      ( parseFloat( styleSide ) / 100 ) * parentSize[ measure ];
  }
  return parseInt( styleSide, 10 );
};

// add transform: translate( x, y ) to position
proto._addTransformPosition = function( style ) {
  var transform = style.transform;
  // bail out if value is 'none'
  if ( transform.indexOf('matrix') !== 0 ) {
    return;
  }
  // split matrix(1, 0, 0, 1, x, y)
  var matrixValues = transform.split(',');
  // translate X value is in 12th or 4th position
  var xIndex = transform.indexOf('matrix3d') === 0 ? 12 : 4;
  var translateX = parseInt( matrixValues[ xIndex ], 10 );
  // translate Y value is in 13th or 5th position
  var translateY = parseInt( matrixValues[ xIndex + 1 ], 10 );
  this.position.x += translateX;
  this.position.y += translateY;
};

// -------------------------- events -------------------------- //

proto.onPointerDown = function( event, pointer ) {
  this.element.classList.add('is-pointer-down');
  this.dispatchJQueryEvent( 'pointerDown', event, [ pointer ] );
};

/**
 * drag start
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.dragStart = function( event, pointer ) {
  if ( !this.isEnabled ) {
    return;
  }
  this._getPosition();
  this.measureContainment();
  // position _when_ drag began
  this.startPosition.x = this.position.x;
  this.startPosition.y = this.position.y;
  // reset left/top style
  this.setLeftTop();

  this.dragPoint.x = 0;
  this.dragPoint.y = 0;

  this.element.classList.add('is-dragging');
  this.dispatchEvent( 'dragStart', event, [ pointer ] );
  // start animation
  this.animate();
};

proto.measureContainment = function() {
  var container = this.getContainer();
  if ( !container ) {
    return;
  }

  var elemSize = getSize( this.element );
  var containerSize = getSize( container );
  var elemRect = this.element.getBoundingClientRect();
  var containerRect = container.getBoundingClientRect();

  var borderSizeX = containerSize.borderLeftWidth + containerSize.borderRightWidth;
  var borderSizeY = containerSize.borderTopWidth + containerSize.borderBottomWidth;

  var position = this.relativeStartPosition = {
    x: elemRect.left - ( containerRect.left + containerSize.borderLeftWidth ),
    y: elemRect.top - ( containerRect.top + containerSize.borderTopWidth )
  };

  this.containSize = {
    width: ( containerSize.width - borderSizeX ) - position.x - elemSize.width,
    height: ( containerSize.height - borderSizeY ) - position.y - elemSize.height
  };
};

proto.getContainer = function() {
  var containment = this.options.containment;
  if ( !containment ) {
    return;
  }
  var isElement = containment instanceof HTMLElement;
  // use as element
  if ( isElement ) {
    return containment;
  }
  // querySelector if string
  if ( typeof containment == 'string' ) {
    return document.querySelector( containment );
  }
  // fallback to parent element
  return this.element.parentNode;
};

// ----- move event ----- //

proto.onPointerMove = function( event, pointer, moveVector ) {
  this.dispatchJQueryEvent( 'pointerMove', event, [ pointer, moveVector ] );
};

/**
 * drag move
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.dragMove = function( event, pointer, moveVector ) {
  if ( !this.isEnabled ) {
    return;
  }
  var dragX = moveVector.x;
  var dragY = moveVector.y;

  var grid = this.options.grid;
  var gridX = grid && grid[0];
  var gridY = grid && grid[1];

  dragX = applyGrid( dragX, gridX );
  dragY = applyGrid( dragY, gridY );

  dragX = this.containDrag( 'x', dragX, gridX );
  dragY = this.containDrag( 'y', dragY, gridY );

  // constrain to axis
  dragX = this.options.axis == 'y' ? 0 : dragX;
  dragY = this.options.axis == 'x' ? 0 : dragY;

  this.position.x = this.startPosition.x + dragX;
  this.position.y = this.startPosition.y + dragY;
  // set dragPoint properties
  this.dragPoint.x = dragX;
  this.dragPoint.y = dragY;

  this.dispatchEvent( 'dragMove', event, [ pointer, moveVector ] );
};

function applyGrid( value, grid, method ) {
  method = method || 'round';
  return grid ? Math[ method ]( value / grid ) * grid : value;
}

proto.containDrag = function( axis, drag, grid ) {
  if ( !this.options.containment ) {
    return drag;
  }
  var measure = axis == 'x' ? 'width' : 'height';

  var rel = this.relativeStartPosition[ axis ];
  var min = applyGrid( -rel, grid, 'ceil' );
  var max = this.containSize[ measure ];
  max = applyGrid( max, grid, 'floor' );
  return  Math.max( min, Math.min( max, drag ) );
};

// ----- end event ----- //

/**
 * pointer up
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.onPointerUp = function( event, pointer ) {
  this.element.classList.remove('is-pointer-down');
  this.dispatchJQueryEvent( 'pointerUp', event, [ pointer ] );
};

/**
 * drag end
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.dragEnd = function( event, pointer ) {
  if ( !this.isEnabled ) {
    return;
  }
  // use top left position when complete
  this.element.style.transform = '';
  this.setLeftTop();
  this.element.classList.remove('is-dragging');
  this.dispatchEvent( 'dragEnd', event, [ pointer ] );
};

// -------------------------- animation -------------------------- //

proto.animate = function() {
  // only render and animate if dragging
  if ( !this.isDragging ) {
    return;
  }

  this.positionDrag();

  var _this = this;
  requestAnimationFrame( function animateFrame() {
    _this.animate();
  });

};

// left/top positioning
proto.setLeftTop = function() {
  this.element.style.left = this.position.x + 'px';
  this.element.style.top  = this.position.y + 'px';
};

proto.positionDrag = function() {
  this.element.style.transform = 'translate3d( ' + this.dragPoint.x +
    'px, ' + this.dragPoint.y + 'px, 0)';
};

// ----- staticClick ----- //

proto.staticClick = function( event, pointer ) {
  this.dispatchEvent( 'staticClick', event, [ pointer ] );
};

// ----- methods ----- //

/**
 * @param {Number} x
 * @param {Number} y
 */
proto.setPosition = function( x, y ) {
  this.position.x = x;
  this.position.y = y;
  this.setLeftTop();
};

proto.enable = function() {
  this.isEnabled = true;
};

proto.disable = function() {
  this.isEnabled = false;
  if ( this.isDragging ) {
    this.dragEnd();
  }
};

proto.destroy = function() {
  this.disable();
  // reset styles
  this.element.style.transform = '';
  this.element.style.left = '';
  this.element.style.top = '';
  this.element.style.position = '';
  // unbind handles
  this.unbindHandles();
  // remove jQuery data
  if ( this.$element ) {
    this.$element.removeData('draggabilly');
  }
};

// ----- jQuery bridget ----- //

// required for jQuery bridget
proto._init = noop;

if ( jQuery && jQuery.bridget ) {
  jQuery.bridget( 'draggabilly', Draggabilly );
}

// -----  ----- //

return Draggabilly;

}));

},{"get-size":3,"unidragger":5}],2:[function(require,module,exports){
/**
 * EvEmitter v1.1.0
 * Lil' event emitter
 * MIT License
 */

/* jshint unused: true, undef: true, strict: true */

( function( global, factory ) {
  // universal module definition
  /* jshint strict: false */ /* globals define, module, window */
  if ( typeof define == 'function' && define.amd ) {
    // AMD - RequireJS
    define( factory );
  } else if ( typeof module == 'object' && module.exports ) {
    // CommonJS - Browserify, Webpack
    module.exports = factory();
  } else {
    // Browser globals
    global.EvEmitter = factory();
  }

}( typeof window != 'undefined' ? window : this, function() {

"use strict";

function EvEmitter() {}

var proto = EvEmitter.prototype;

proto.on = function( eventName, listener ) {
  if ( !eventName || !listener ) {
    return;
  }
  // set events hash
  var events = this._events = this._events || {};
  // set listeners array
  var listeners = events[ eventName ] = events[ eventName ] || [];
  // only add once
  if ( listeners.indexOf( listener ) == -1 ) {
    listeners.push( listener );
  }

  return this;
};

proto.once = function( eventName, listener ) {
  if ( !eventName || !listener ) {
    return;
  }
  // add event
  this.on( eventName, listener );
  // set once flag
  // set onceEvents hash
  var onceEvents = this._onceEvents = this._onceEvents || {};
  // set onceListeners object
  var onceListeners = onceEvents[ eventName ] = onceEvents[ eventName ] || {};
  // set flag
  onceListeners[ listener ] = true;

  return this;
};

proto.off = function( eventName, listener ) {
  var listeners = this._events && this._events[ eventName ];
  if ( !listeners || !listeners.length ) {
    return;
  }
  var index = listeners.indexOf( listener );
  if ( index != -1 ) {
    listeners.splice( index, 1 );
  }

  return this;
};

proto.emitEvent = function( eventName, args ) {
  var listeners = this._events && this._events[ eventName ];
  if ( !listeners || !listeners.length ) {
    return;
  }
  // copy over to avoid interference if .off() in listener
  listeners = listeners.slice(0);
  args = args || [];
  // once stuff
  var onceListeners = this._onceEvents && this._onceEvents[ eventName ];

  for ( var i=0; i < listeners.length; i++ ) {
    var listener = listeners[i]
    var isOnce = onceListeners && onceListeners[ listener ];
    if ( isOnce ) {
      // remove listener
      // remove before trigger to prevent recursion
      this.off( eventName, listener );
      // unset once flag
      delete onceListeners[ listener ];
    }
    // trigger listener
    listener.apply( this, args );
  }

  return this;
};

proto.allOff = function() {
  delete this._events;
  delete this._onceEvents;
};

return EvEmitter;

}));

},{}],3:[function(require,module,exports){
/*!
 * getSize v2.0.3
 * measure size of elements
 * MIT license
 */

/* jshint browser: true, strict: true, undef: true, unused: true */
/* globals console: false */

( function( window, factory ) {
  /* jshint strict: false */ /* globals define, module */
  if ( typeof define == 'function' && define.amd ) {
    // AMD
    define( factory );
  } else if ( typeof module == 'object' && module.exports ) {
    // CommonJS
    module.exports = factory();
  } else {
    // browser global
    window.getSize = factory();
  }

})( window, function factory() {
'use strict';

// -------------------------- helpers -------------------------- //

// get a number from a string, not a percentage
function getStyleSize( value ) {
  var num = parseFloat( value );
  // not a percent like '100%', and a number
  var isValid = value.indexOf('%') == -1 && !isNaN( num );
  return isValid && num;
}

function noop() {}

var logError = typeof console == 'undefined' ? noop :
  function( message ) {
    console.error( message );
  };

// -------------------------- measurements -------------------------- //

var measurements = [
  'paddingLeft',
  'paddingRight',
  'paddingTop',
  'paddingBottom',
  'marginLeft',
  'marginRight',
  'marginTop',
  'marginBottom',
  'borderLeftWidth',
  'borderRightWidth',
  'borderTopWidth',
  'borderBottomWidth'
];

var measurementsLength = measurements.length;

function getZeroSize() {
  var size = {
    width: 0,
    height: 0,
    innerWidth: 0,
    innerHeight: 0,
    outerWidth: 0,
    outerHeight: 0
  };
  for ( var i=0; i < measurementsLength; i++ ) {
    var measurement = measurements[i];
    size[ measurement ] = 0;
  }
  return size;
}

// -------------------------- getStyle -------------------------- //

/**
 * getStyle, get style of element, check for Firefox bug
 * https://bugzilla.mozilla.org/show_bug.cgi?id=548397
 */
function getStyle( elem ) {
  var style = getComputedStyle( elem );
  if ( !style ) {
    logError( 'Style returned ' + style +
      '. Are you running this code in a hidden iframe on Firefox? ' +
      'See https://bit.ly/getsizebug1' );
  }
  return style;
}

// -------------------------- setup -------------------------- //

var isSetup = false;

var isBoxSizeOuter;

/**
 * setup
 * check isBoxSizerOuter
 * do on first getSize() rather than on page load for Firefox bug
 */
function setup() {
  // setup once
  if ( isSetup ) {
    return;
  }
  isSetup = true;

  // -------------------------- box sizing -------------------------- //

  /**
   * Chrome & Safari measure the outer-width on style.width on border-box elems
   * IE11 & Firefox<29 measures the inner-width
   */
  var div = document.createElement('div');
  div.style.width = '200px';
  div.style.padding = '1px 2px 3px 4px';
  div.style.borderStyle = 'solid';
  div.style.borderWidth = '1px 2px 3px 4px';
  div.style.boxSizing = 'border-box';

  var body = document.body || document.documentElement;
  body.appendChild( div );
  var style = getStyle( div );
  // round value for browser zoom. desandro/masonry#928
  isBoxSizeOuter = Math.round( getStyleSize( style.width ) ) == 200;
  getSize.isBoxSizeOuter = isBoxSizeOuter;

  body.removeChild( div );
}

// -------------------------- getSize -------------------------- //

function getSize( elem ) {
  setup();

  // use querySeletor if elem is string
  if ( typeof elem == 'string' ) {
    elem = document.querySelector( elem );
  }

  // do not proceed on non-objects
  if ( !elem || typeof elem != 'object' || !elem.nodeType ) {
    return;
  }

  var style = getStyle( elem );

  // if hidden, everything is 0
  if ( style.display == 'none' ) {
    return getZeroSize();
  }

  var size = {};
  size.width = elem.offsetWidth;
  size.height = elem.offsetHeight;

  var isBorderBox = size.isBorderBox = style.boxSizing == 'border-box';

  // get all measurements
  for ( var i=0; i < measurementsLength; i++ ) {
    var measurement = measurements[i];
    var value = style[ measurement ];
    var num = parseFloat( value );
    // any 'auto', 'medium' value will be 0
    size[ measurement ] = !isNaN( num ) ? num : 0;
  }

  var paddingWidth = size.paddingLeft + size.paddingRight;
  var paddingHeight = size.paddingTop + size.paddingBottom;
  var marginWidth = size.marginLeft + size.marginRight;
  var marginHeight = size.marginTop + size.marginBottom;
  var borderWidth = size.borderLeftWidth + size.borderRightWidth;
  var borderHeight = size.borderTopWidth + size.borderBottomWidth;

  var isBorderBoxSizeOuter = isBorderBox && isBoxSizeOuter;

  // overwrite width and height if we can get it from style
  var styleWidth = getStyleSize( style.width );
  if ( styleWidth !== false ) {
    size.width = styleWidth +
      // add padding and border unless it's already including it
      ( isBorderBoxSizeOuter ? 0 : paddingWidth + borderWidth );
  }

  var styleHeight = getStyleSize( style.height );
  if ( styleHeight !== false ) {
    size.height = styleHeight +
      // add padding and border unless it's already including it
      ( isBorderBoxSizeOuter ? 0 : paddingHeight + borderHeight );
  }

  size.innerWidth = size.width - ( paddingWidth + borderWidth );
  size.innerHeight = size.height - ( paddingHeight + borderHeight );

  size.outerWidth = size.width + marginWidth;
  size.outerHeight = size.height + marginHeight;

  return size;
}

return getSize;

});

},{}],4:[function(require,module,exports){
(function (global){
!function(){function e(t,n,r){function o(s,a){if(!n[s]){if(!t[s]){var l="function"==typeof require&&require;if(!a&&l)return l(s,!0);if(i)return i(s,!0);var u=new Error("Cannot find module '"+s+"'");throw u.code="MODULE_NOT_FOUND",u}var c=n[s]={exports:{}};t[s][0].call(c.exports,function(e){var n=t[s][1][e];return o(n||e)},c,c.exports,e,t,n,r)}return n[s].exports}for(var i="function"==typeof require&&require,s=0;s<r.length;s++)o(r[s]);return o}return e}()({1:[function(e,t,n){(function(t){!function(){function t(n,r,o){function i(a,l){if(!r[a]){if(!n[a]){var u="function"==typeof e&&e;if(!l&&u)return u(a,!0);if(s)return s(a,!0);var c=new Error("Cannot find module '"+a+"'");throw c.code="MODULE_NOT_FOUND",c}var h=r[a]={exports:{}};n[a][0].call(h.exports,function(e){return i(n[a][1][e]||e)},h,h.exports,t,n,r,o)}return r[a].exports}for(var s="function"==typeof e&&e,a=0;a<o.length;a++)i(o[a]);return i}return t}()({1:[function(e,n,r){(function(t){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(r,"__esModule",{value:!0}),r.WebCLGL=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./WebCLGLBuffer.class"),a=e("./WebCLGLKernel.class"),l=e("./WebCLGLVertexFragmentProgram.class"),u=e("./WebCLGLUtils.class"),c=r.WebCLGL=function(){function e(t){o(this,e),this.utils=new u.WebCLGLUtils,this._gl=null,this.e=null,void 0===t||null===t?(this.e=document.createElement("canvas"),this.e.width=32,this.e.height=32,this._gl=u.WebCLGLUtils.getWebGLContextFromCanvas(this.e,{antialias:!1})):this._gl=t,this._arrExt={OES_texture_float:null,OES_texture_float_linear:null,OES_element_index_uint:null,WEBGL_draw_buffers:null};for(var n in this._arrExt)this._arrExt[n]=this._gl.getExtension(n),null==this._arrExt[n]&&console.error("extension "+n+" not available");this._maxDrawBuffers=null,this._arrExt.hasOwnProperty("WEBGL_draw_buffers")&&null!=this._arrExt.WEBGL_draw_buffers?(this._maxDrawBuffers=this._gl.getParameter(this._arrExt.WEBGL_draw_buffers.MAX_DRAW_BUFFERS_WEBGL),console.log("Max draw buffers: "+this._maxDrawBuffers)):console.log("Max draw buffers: 1");var r=this._gl.getShaderPrecisionFormat(this._gl.FRAGMENT_SHADER,this._gl.HIGH_FLOAT);this.precision=0!==r.precision?"precision highp float;\n\nprecision highp int;\n\n":"precision lowp float;\n\nprecision lowp int;\n\n",this._currentTextureUnit=0,this._bufferWidth=0;var i=this.utils.loadQuad(void 0,1,1);this.vertexBuffer_QUAD=this._gl.createBuffer(),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,this.vertexBuffer_QUAD),this._gl.bufferData(this._gl.ARRAY_BUFFER,new Float32Array(i.vertexArray),this._gl.STATIC_DRAW),this.indexBuffer_QUAD=this._gl.createBuffer(),this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,this.indexBuffer_QUAD),this._gl.bufferData(this._gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(i.indexArray),this._gl.STATIC_DRAW),this.arrayCopyTex=[];var s=this.precision+"attribute vec3 aVertexPosition;\nvarying vec2 vCoord;\nvoid main(void) {\ngl_Position = vec4(aVertexPosition, 1.0);\nvCoord = aVertexPosition.xy*0.5+0.5;\n}\n",a=this.precision+"uniform sampler2D sampler_buffer;\nvarying vec2 vCoord;\nvoid main(void) {\ngl_FragColor = texture2D(sampler_buffer, vCoord);}\n";this.shader_readpixels=this._gl.createProgram(),this.utils.createShader(this._gl,"CLGLREADPIXELS",s,a,this.shader_readpixels),this.attr_VertexPos=this._gl.getAttribLocation(this.shader_readpixels,"aVertexPosition"),this.sampler_buffer=this._gl.getUniformLocation(this.shader_readpixels,"sampler_buffer");var l=function(){return void 0!==this._maxDrawBuffers&&null!==this._maxDrawBuffers?"#extension GL_EXT_draw_buffers : require\n":""}.bind(this),c=(function(){for(var e="",t=0,n=this._maxDrawBuffers;t<n;t++)e+="layout(location = "+t+") out vec4 outCol"+t+";\n";return e}.bind(this),function(){for(var e="",t=0,n=this._maxDrawBuffers;t<n;t++)e+="gl_FragData["+t+"] = texture2D(uArrayCT["+t+"], vCoord);\n";return e}.bind(this));s=this.precision+"attribute vec3 aVertexPosition;\nvarying vec2 vCoord;\nvoid main(void) {\ngl_Position = vec4(aVertexPosition, 1.0);\nvCoord = aVertexPosition.xy*0.5+0.5;\n}",a=l()+this.precision+"uniform sampler2D uArrayCT["+this._maxDrawBuffers+"];\nvarying vec2 vCoord;\nvoid main(void) {\n"+c()+"}",this.shader_copyTexture=this._gl.createProgram(),this.utils.createShader(this._gl,"CLGLCOPYTEXTURE",s,a,this.shader_copyTexture),this.attr_copyTexture_pos=this._gl.getAttribLocation(this.shader_copyTexture,"aVertexPosition");for(var h=0,f=this._maxDrawBuffers;h<f;h++)this.arrayCopyTex[h]=this._gl.getUniformLocation(this.shader_copyTexture,"uArrayCT["+h+"]");this.textureDataAux=this._gl.createTexture(),this._gl.bindTexture(this._gl.TEXTURE_2D,this.textureDataAux),this._gl.texImage2D(this._gl.TEXTURE_2D,0,this._gl.RGBA,2,2,0,this._gl.RGBA,this._gl.FLOAT,new Float32Array([1,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1])),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_MAG_FILTER,this._gl.NEAREST),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_MIN_FILTER,this._gl.NEAREST),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_WRAP_S,this._gl.CLAMP_TO_EDGE),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_WRAP_T,this._gl.CLAMP_TO_EDGE),this._gl.bindTexture(this._gl.TEXTURE_2D,null)}return i(e,[{key:"getContext",value:function(){return this._gl}},{key:"getMaxDrawBuffers",value:function(){return this._maxDrawBuffers}},{key:"checkFramebufferStatus",value:function(){var e=this._gl.checkFramebufferStatus(this._gl.FRAMEBUFFER),t={};return t[this._gl.FRAMEBUFFER_COMPLETE]=!0,t[this._gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT]="FRAMEBUFFER_INCOMPLETE_ATTACHMENT: The attachment types are mismatched or not all framebuffer attachment points are framebuffer attachment complete",t[this._gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT]="FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: There is no attachment",t[this._gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS]="FRAMEBUFFER_INCOMPLETE_DIMENSIONS: Height and width of the attachment are not the same",t[this._gl.FRAMEBUFFER_UNSUPPORTED]="FRAMEBUFFER_UNSUPPORTED: The format of the attachment is not supported or if depth and stencil attachments are not the same renderbuffer",!0===t[e]&&null!==t[e]||(console.log(t[e]),!1)}},{key:"copy",value:function(e,t){if(void 0!==t&&null!==t)if(void 0!==t[0]&&null!==t[0]){this._gl.viewport(0,0,t[0].W,t[0].H),this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,t[0].fBuffer);for(var n=[],r=0,o=t.length;r<o;r++)void 0!==t[r]&&null!==t[r]?(this._gl.framebufferTexture2D(this._gl.FRAMEBUFFER,this._arrExt.WEBGL_draw_buffers["COLOR_ATTACHMENT"+r+"_WEBGL"],this._gl.TEXTURE_2D,t[r].textureData,0),n[r]=this._arrExt.WEBGL_draw_buffers["COLOR_ATTACHMENT"+r+"_WEBGL"]):n[r]=this._gl.NONE;if(this._arrExt.WEBGL_draw_buffers.drawBuffersWEBGL(n),!0===this.checkFramebufferStatus()){this._gl.useProgram(this.shader_copyTexture);for(var i=0,s=t.length;i<s;i++)this._gl.activeTexture(this._gl["TEXTURE"+i]),void 0!==t[i]&&null!==t[i]?this._gl.bindTexture(this._gl.TEXTURE_2D,t[i].textureDataTemp):this._gl.bindTexture(this._gl.TEXTURE_2D,this.textureDataAux),this._gl.uniform1i(this.arrayCopyTex[i],i);this.copyNow(t)}}else this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,null);else this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,null)}},{key:"copyNow",value:function(e){this._gl.enableVertexAttribArray(this.attr_copyTexture_pos),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,this.vertexBuffer_QUAD),this._gl.vertexAttribPointer(this.attr_copyTexture_pos,3,this._gl.FLOAT,!1,0,0),this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,this.indexBuffer_QUAD),this._gl.drawElements(this._gl.TRIANGLES,6,this._gl.UNSIGNED_SHORT,0)}},{key:"createBuffer",value:function(e,t,n){return new s.WebCLGLBuffer(this._gl,e,t,n)}},{key:"createKernel",value:function(e,t){return new a.WebCLGLKernel(this._gl,e,t)}},{key:"createVertexFragmentProgram",value:function(e,t,n,r){return new l.WebCLGLVertexFragmentProgram(this._gl,e,t,n,r)}},{key:"fillBuffer",value:function(e,t,n){this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,n),this._gl.framebufferTexture2D(this._gl.FRAMEBUFFER,this._arrExt.WEBGL_draw_buffers.COLOR_ATTACHMENT0_WEBGL,this._gl.TEXTURE_2D,e,0);var r=[this._arrExt.WEBGL_draw_buffers.COLOR_ATTACHMENT0_WEBGL];this._arrExt.WEBGL_draw_buffers.drawBuffersWEBGL(r),void 0!==t&&null!==t&&this._gl.clearColor(t[0],t[1],t[2],t[3]),this._gl.clear(this._gl.COLOR_BUFFER_BIT)}},{key:"bindAttributeValue",value:function(e,t){void 0!==t&&null!==t?"float4_fromAttr"===e.type?(this._gl.enableVertexAttribArray(e.location[0]),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,t.vertexData0),this._gl.vertexAttribPointer(e.location[0],4,this._gl.FLOAT,!1,0,0)):"float_fromAttr"===e.type&&(this._gl.enableVertexAttribArray(e.location[0]),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,t.vertexData0),this._gl.vertexAttribPointer(e.location[0],1,this._gl.FLOAT,!1,0,0)):this._gl.disableVertexAttribArray(e.location[0])}},{key:"bindSamplerValue",value:function(e,t,n){this._currentTextureUnit<16?this._gl.activeTexture(this._gl["TEXTURE"+this._currentTextureUnit]):this._gl.activeTexture(this._gl.TEXTURE16),void 0!==n&&null!==n?(this._gl.bindTexture(this._gl.TEXTURE_2D,n.textureData),0===this._bufferWidth&&(this._bufferWidth=n.W,this._gl.uniform1f(e,this._bufferWidth))):this._gl.bindTexture(this._gl.TEXTURE_2D,this.textureDataAux),this._gl.uniform1i(t.location[0],this._currentTextureUnit),this._currentTextureUnit++}},{key:"bindUniformValue",value:function(e,t){void 0!==t&&null!==t&&("float"===e.type?t.constructor===Array?this._gl.uniform1fv(e.location[0],t):this._gl.uniform1f(e.location[0],t):"float4"===e.type?this._gl.uniform4f(e.location[0],t[0],t[1],t[2],t[3]):"mat4"===e.type&&this._gl.uniformMatrix4fv(e.location[0],!1,t))}},{key:"bindValue",value:function(e,t,n){switch(t.expectedMode){case"ATTRIBUTE":this.bindAttributeValue(t,n);break;case"SAMPLER":this.bindSamplerValue(e.uBufferWidth,t,n);break;case"UNIFORM":this.bindUniformValue(t,n)}}},{key:"bindFB",value:function(e,t){if(void 0!==e&&null!==e){if(void 0!==e[0]&&null!==e[0]){this._gl.viewport(0,0,e[0].W,e[0].H),this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,!0===t?e[0].fBufferTemp:e[0].fBuffer);for(var n=[],r=0,o=e.length;r<o;r++)if(void 0!==e[r]&&null!==e[r]){var i=!0===t?e[r].textureDataTemp:e[r].textureData;this._gl.framebufferTexture2D(this._gl.FRAMEBUFFER,this._arrExt.WEBGL_draw_buffers["COLOR_ATTACHMENT"+r+"_WEBGL"],this._gl.TEXTURE_2D,i,0),n[r]=this._arrExt.WEBGL_draw_buffers["COLOR_ATTACHMENT"+r+"_WEBGL"]}else n[r]=this._gl.NONE;return this._arrExt.WEBGL_draw_buffers.drawBuffersWEBGL(n),this.checkFramebufferStatus()}return this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,null),!0}return this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,null),!0}},{key:"enqueueNDRangeKernel",value:function(e,t,n,r){if(this._bufferWidth=0,this._gl.useProgram(e.kernel),!0===this.bindFB(t,n)){this._currentTextureUnit=0;for(var o in e.in_values)this.bindValue(e,e.in_values[o],r[o]);this._gl.enableVertexAttribArray(e.attr_VertexPos),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,this.vertexBuffer_QUAD),this._gl.vertexAttribPointer(e.attr_VertexPos,3,this._gl.FLOAT,!1,0,0),this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,this.indexBuffer_QUAD),this._gl.drawElements(this._gl.TRIANGLES,6,this._gl.UNSIGNED_SHORT,0)}}},{key:"enqueueVertexFragmentProgram",value:function(e,t,n,r,o,i){this._bufferWidth=0,this._gl.useProgram(e.vertexFragmentProgram);var s=void 0!==n&&null!==n?n:4;if(!0===this.bindFB(r,o)&&void 0!==t&&null!==t){this._currentTextureUnit=0;for(var a in e.in_vertex_values)this.bindValue(e,e.in_vertex_values[a],i[a]);for(var l in e.in_fragment_values)this.bindValue(e,e.in_fragment_values[l],i[l]);"VERTEX_INDEX"===t.mode?(this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,t.vertexData0),this._gl.drawElements(s,t.length,this._gl.UNSIGNED_SHORT,0)):this._gl.drawArrays(s,0,t.length)}}},{key:"readBuffer",value:function(e){void 0!==this.e&&null!==this.e&&(this.e.width=e.W,this.e.height=e.H),this._gl.useProgram(this.shader_readpixels),this._gl.viewport(0,0,e.W,e.H),this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,e.fBufferTemp),this._gl.framebufferTexture2D(this._gl.FRAMEBUFFER,this._arrExt.WEBGL_draw_buffers.COLOR_ATTACHMENT0_WEBGL,this._gl.TEXTURE_2D,e.textureDataTemp,0);var t=[this._arrExt.WEBGL_draw_buffers.COLOR_ATTACHMENT0_WEBGL];if(this._arrExt.WEBGL_draw_buffers.drawBuffersWEBGL(t),this._gl.activeTexture(this._gl.TEXTURE0),this._gl.bindTexture(this._gl.TEXTURE_2D,e.textureData),this._gl.uniform1i(this.sampler_buffer,0),this._gl.enableVertexAttribArray(this.attr_VertexPos),this._gl.bindBuffer(this._gl.ARRAY_BUFFER,this.vertexBuffer_QUAD),this._gl.vertexAttribPointer(this.attr_VertexPos,3,e._supportFormat,!1,0,0),this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,this.indexBuffer_QUAD),this._gl.drawElements(this._gl.TRIANGLES,6,this._gl.UNSIGNED_SHORT,0),void 0!==e.outArrayFloat&&null!==e.outArrayFloat||(e.outArrayFloat=new Float32Array(e.W*e.H*4)),this._gl.readPixels(0,0,e.W,e.H,this._gl.RGBA,this._gl.FLOAT,e.outArrayFloat),"FLOAT"===e.type){for(var n=new Float32Array(e.outArrayFloat.length/4),r=0,o=e.outArrayFloat.length/4;r<o;r++)n[r]=e.outArrayFloat[4*r];e.outArrayFloat=n}return e.outArrayFloat}}],[{key:"enqueueReadBuffer_WebGLTexture",value:function(e){return e.textureData}}]),e}();t.WebCLGL=c,n.exports.WebCLGL=c}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./WebCLGLBuffer.class":2,"./WebCLGLKernel.class":4,"./WebCLGLUtils.class":5,"./WebCLGLVertexFragmentProgram.class":6}],2:[function(e,n,r){(function(e){"use strict";function t(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(r,"__esModule",{value:!0});var o=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),i=r.WebCLGLBuffer=function(){function e(n,r,o,i){t(this,e),this._gl=n,this.type=void 0!==r||null!==r?r:"FLOAT",this._supportFormat=this._gl.FLOAT,this.linear=void 0===o&&null===o||o,this.mode=void 0!==i||null!==i?i:"SAMPLER",this.W=null,this.H=null,this.textureData=null,this.textureDataTemp=null,this.vertexData0=null,this.fBuffer=null,this.renderBuffer=null,this.fBufferTemp=null,this.renderBufferTemp=null,"SAMPLER"===this.mode&&(this.textureData=this._gl.createTexture(),this.textureDataTemp=this._gl.createTexture()),"SAMPLER"!==this.mode&&"ATTRIBUTE"!==this.mode&&"VERTEX_INDEX"!==this.mode||(this.vertexData0=this._gl.createBuffer())}return o(e,[{key:"createFramebufferAndRenderbuffer",value:function(){var e=function(){var e=this._gl.createRenderbuffer();return this._gl.bindRenderbuffer(this._gl.RENDERBUFFER,e),this._gl.renderbufferStorage(this._gl.RENDERBUFFER,this._gl.DEPTH_COMPONENT16,this.W,this.H),this._gl.bindRenderbuffer(this._gl.RENDERBUFFER,null),e}.bind(this);null!=this.fBuffer&&(this._gl.deleteFramebuffer(this.fBuffer),this._gl.deleteFramebuffer(this.fBufferTemp),this._gl.deleteRenderbuffer(this.renderBuffer),this._gl.deleteRenderbuffer(this.renderBufferTemp)),this.fBuffer=this._gl.createFramebuffer(),this.renderBuffer=e(),this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,this.fBuffer),this._gl.framebufferRenderbuffer(this._gl.FRAMEBUFFER,this._gl.DEPTH_ATTACHMENT,this._gl.RENDERBUFFER,this.renderBuffer),this.fBufferTemp=this._gl.createFramebuffer(),this.renderBufferTemp=e(),this._gl.bindFramebuffer(this._gl.FRAMEBUFFER,this.fBufferTemp),this._gl.framebufferRenderbuffer(this._gl.FRAMEBUFFER,this._gl.DEPTH_ATTACHMENT,this._gl.RENDERBUFFER,this.renderBufferTemp)}},{key:"writeWebGLTextureBuffer",value:function(e,t){var n=function(e,t){!1===t||void 0===t||null===t?this._gl.pixelStorei(this._gl.UNPACK_FLIP_Y_WEBGL,!1):this._gl.pixelStorei(this._gl.UNPACK_FLIP_Y_WEBGL,!0),this._gl.pixelStorei(this._gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),this._gl.bindTexture(this._gl.TEXTURE_2D,e)}.bind(this),r=function(e){e instanceof HTMLImageElement?"FLOAT4"===this.type&&this._gl.texImage2D(this._gl.TEXTURE_2D,0,this._gl.RGBA,this._gl.RGBA,this._supportFormat,e):this._gl.texImage2D(this._gl.TEXTURE_2D,0,this._gl.RGBA,this.W,this.H,0,this._gl.RGBA,this._supportFormat,e)}.bind(this),o=function(){this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_MAG_FILTER,this._gl.NEAREST),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_MIN_FILTER,this._gl.NEAREST),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_WRAP_S,this._gl.CLAMP_TO_EDGE),this._gl.texParameteri(this._gl.TEXTURE_2D,this._gl.TEXTURE_WRAP_T,this._gl.CLAMP_TO_EDGE)}.bind(this);e instanceof WebGLTexture?(this.textureData=e,this.textureDataTemp=e):(n(this.textureData,t),r(e),o(),n(this.textureDataTemp,t),r(e),o()),this._gl.bindTexture(this._gl.TEXTURE_2D,null)}},{key:"writeBuffer",value:function(e,t,n){var r=function(e){if(!(e instanceof HTMLImageElement))if(this.length.constructor===Array?(this.length=this.length[0]*this.length[1],this.W=this.length[0],this.H=this.length[1]):(this.W=Math.ceil(Math.sqrt(this.length)),this.H=this.W),"FLOAT4"===this.type){e=e instanceof Float32Array?e:new Float32Array(e);var t=this.W*this.H*4;if(e.length!==t){for(var n=new Float32Array(t),r=0;r<t;r++)n[r]=null!=e[r]?e[r]:0;e=n}}else if("FLOAT"===this.type){for(var o=this.W*this.H*4,i=new Float32Array(o),s=0,a=this.W*this.H;s<a;s++){var l=4*s;i[l]=null!=e[s]?e[s]:0,i[l+1]=0,i[l+2]=0,i[l+3]=0}e=i}return e}.bind(this);void 0===n||null===n?e instanceof HTMLImageElement?this.length=e.width*e.height:this.length="FLOAT4"===this.type?e.length/4:e.length:this.length=[n[0],n[1]],"SAMPLER"===this.mode&&this.writeWebGLTextureBuffer(r(e),t),"SAMPLER"!==this.mode&&"ATTRIBUTE"!==this.mode||(this._gl.bindBuffer(this._gl.ARRAY_BUFFER,this.vertexData0),this._gl.bufferData(this._gl.ARRAY_BUFFER,e instanceof Float32Array?e:new Float32Array(e),this._gl.STATIC_DRAW)),"VERTEX_INDEX"===this.mode&&(this._gl.bindBuffer(this._gl.ELEMENT_ARRAY_BUFFER,this.vertexData0),this._gl.bufferData(this._gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(e),this._gl.STATIC_DRAW)),this.createFramebufferAndRenderbuffer()}},{key:"remove",value:function(){"SAMPLER"===this.mode&&(this._gl.deleteTexture(this.textureData),this._gl.deleteTexture(this.textureDataTemp)),"SAMPLER"!==this.mode&&"ATTRIBUTE"!==this.mode&&"VERTEX_INDEX"!==this.mode||this._gl.deleteBuffer(this.vertexData0),this._gl.deleteFramebuffer(this.fBuffer),this._gl.deleteFramebuffer(this.fBufferTemp),this._gl.deleteRenderbuffer(this.renderBuffer),this._gl.deleteRenderbuffer(this.renderBufferTemp)}}]),e}();e.WebCLGLBuffer=i,n.exports.WebCLGLBuffer=i}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],3:[function(e,n,r){(function(t){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(){var e=new u,t=null;return arguments[0]instanceof WebGLRenderingContext?(t=arguments[0],e.setCtx(t),e._webCLGL=new a.WebCLGL(t),e.iniG(arguments),e):arguments[0]instanceof HTMLCanvasElement?(t=l.WebCLGLUtils.getWebGLContextFromCanvas(arguments[0]),e.setCtx(t),e._webCLGL=new a.WebCLGL(t),e.iniG(arguments),e):(t=l.WebCLGLUtils.getWebGLContextFromCanvas(document.createElement("canvas"),{antialias:!1}),e.setCtx(t),e._webCLGL=new a.WebCLGL(t),e.ini(arguments))}Object.defineProperty(r,"__esModule",{value:!0}),r.WebCLGLFor=void 0;var s=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}();r.gpufor=i;var a=e("./WebCLGL.class"),l=e("./WebCLGLUtils.class"),u=r.WebCLGLFor=function(){function e(t){o(this,e),this.kernels={},this.vertexFragmentPrograms={},this._args={},this._argsValues={},this.calledArgs={},this._webCLGL=null,this._gl=null}return s(e,[{key:"defineOutputTempModes",value:function(e,t){for(var n=[],r=0;r<e.length;r++)n[r]=null!=e[r]&&function(e,t){var n=!1;for(var r in t)if("indices"!==r){var o=r.split(" ");if(o.length>0&&o[1]===e){n=!0;break}}return n}(e[r],t);return n}},{key:"prepareReturnCode",value:function(e,t){var n=[],r=e.match(new RegExp(/return.*$/gm)),o=(r=r[0].replace("return ","")).match(new RegExp(/\[/gm));if(null!=o&&o.length>=1){r=r.split("[")[1].split("]")[0];for(var i="",s=0,a=0;a<r.length;a++)","===r[a]&&0===s?(n.push(i),i=""):i+=r[a],"("===r[a]&&s++,")"===r[a]&&s--;n.push(i)}else n.push(r.replace(/;$/gm,""));for(var l="",u=0;u<t.length;u++){var c=!1;for(var h in this._args)if("indices"!==h){var f=h.split(" ");if(f[1]===t[u]){var d=f[0].match(new RegExp("float4","gm"));l+=null!=d&&d.length>0?"out"+u+"_float4 = "+n[u]+";\n":"out"+u+"_float = "+n[u]+";\n",c=!0;break}}!1===c&&(l+="out"+u+"_float4 = "+n[u]+";\n")}return l}},{key:"addKernel",value:function(e){var t=e.config,n=t[0],r=t[1]instanceof Array?t[1]:[t[1]],o=t[2],i=t[3],s=this._webCLGL.createKernel(),a=[];for(var l in this._args){var u=l.split(" ")[1];if(void 0!==u&&null!==u){var c=(o+i).match(new RegExp(u.replace(/\[\d.*/,""),"gm"));"indices"!==l&&null!=c&&c.length>0&&(s.in_values[u]={},a.push(l.replace("*attr ","* ").replace(/\[\d.*/,"")))}}i="void main("+a.toString()+") {vec2 "+n+" = get_global_id();"+i.replace(/return.*$/gm,this.prepareReturnCode(i,r))+"}",s.name=e.name,s.viewSource=null!=e.viewSource&&e.viewSource,s.setKernelSource(i,o),s.output=r,s.outputTempModes=this.defineOutputTempModes(r,this._args),s.enabled=!0,s.drawMode=null!=e.drawMode?e.drawMode:4,s.depthTest=null==e.depthTest||e.depthTest,s.blend=null!=e.blend&&e.blend,s.blendEquation=null!=e.blendEquation?e.blendEquation:"FUNC_ADD",s.blendSrcMode=null!=e.blendSrcMode?e.blendSrcMode:"SRC_ALPHA",s.blendDstMode=null!=e.blendDstMode?e.blendDstMode:"ONE_MINUS_SRC_ALPHA",this.kernels[Object.keys(this.kernels).length.toString()]=s}},{key:"addGraphic",value:function(e){var t=e.config,n=[null],r=void 0,o=void 0,i=void 0,s=void 0;5===t.length?(n=t[0]instanceof Array?t[0]:[t[0]],r=t[1],o=t[2],i=t[3],s=t[4]):(r=t[0],o=t[1],i=t[2],s=t[3]);var a=this._webCLGL.createVertexFragmentProgram(),l=[],u=[];for(var c in this._args){var h=c.split(" ")[1];if(void 0!==h&&null!==h){var f=(r+o).match(new RegExp(h.replace(/\[\d.*/,""),"gm"));"indices"!==c&&null!=f&&f.length>0&&(a.in_vertex_values[h]={},l.push(c.replace(/\[\d.*/,"")))}}for(var d in this._args){var g=d.split(" ")[1];if(void 0!==g&&null!==g){var _=(i+s).match(new RegExp(g.replace(/\[\d.*/,""),"gm"));"indices"!==d&&null!=_&&_.length>0&&(a.in_fragment_values[g]={},u.push(d.replace(/\[\d.*/,"")))}}o="void main("+l.toString()+") {"+o+"}",s="void main("+u.toString()+") {"+s.replace(/return.*$/gm,this.prepareReturnCode(s,n))+"}",a.name=e.name,a.viewSource=null!=e.viewSource&&e.viewSource,a.setVertexSource(o,r),a.setFragmentSource(s,i),a.output=n,a.outputTempModes=this.defineOutputTempModes(n,this._args),a.enabled=!0,a.drawMode=null!=e.drawMode?e.drawMode:4,a.depthTest=null==e.depthTest||e.depthTest,a.blend=null==e.blend||e.blend,a.blendEquation=null!=e.blendEquation?e.blendEquation:"FUNC_ADD",a.blendSrcMode=null!=e.blendSrcMode?e.blendSrcMode:"SRC_ALPHA",a.blendDstMode=null!=e.blendDstMode?e.blendDstMode:"ONE_MINUS_SRC_ALPHA",this.vertexFragmentPrograms[Object.keys(this.vertexFragmentPrograms).length.toString()]=a}},{key:"checkArg",value:function(e,t,n){var r=[],o=!1,i=!1;for(var s in t)for(var a in t[s].in_values)if(t[s].in_values[a],a===e){r.push(t[s]);break}for(var l in n){for(var u in n[l].in_vertex_values)if(n[l].in_vertex_values[u],u===e){o=!0;break}for(var c in n[l].in_fragment_values)if(n[l].in_fragment_values[c],c===e){i=!0;break}}return{usedInVertex:o,usedInFragment:i,kernelPr:r}}},{key:"fillArg",value:function(e,t){this._webCLGL.fillBuffer(this._argsValues[e].textureData,t,this._argsValues[e].fBuffer),this._webCLGL.fillBuffer(this._argsValues[e].textureDataTemp,t,this._argsValues[e].fBufferTemp)}},{key:"getAllArgs",value:function(){return this._argsValues}},{key:"addArg",value:function(e){this._args[e]=null}},{key:"getGPUForArg",value:function(e,t){!1===this.calledArgs.hasOwnProperty(e)&&(this.calledArgs[e]=[]),-1===this.calledArgs[e].indexOf(t)&&this.calledArgs[e].push(t),!1===t.calledArgs.hasOwnProperty(e)&&(t.calledArgs[e]=[]),-1===t.calledArgs[e].indexOf(this)&&t.calledArgs[e].push(this);for(var n in t._args){var r=n.split(" ")[1];if(r===e){this._args[n]=t._args[n],this._argsValues[r]=t._argsValues[r];break}}}},{key:"setArg",value:function(e,t,n,r){if("indices"===e)this.setIndices(t);else for(var o in this._args){var i=o.split(" ")[1];if(void 0!==i&&i.replace(/\[\d.*/,"")===e){i!==e&&(e=i);var s=!1;if(null!=o.match(/\*/gm)){var a=this.checkArg(e,this.kernels,this.vertexFragmentPrograms),l="SAMPLER";!0===a.usedInVertex&&0===a.kernelPr.length&&!1===a.usedInFragment&&(l="ATTRIBUTE");var u=o.split("*")[0].toUpperCase();void 0!==r&&null!==r&&(u=r),void 0!==t&&null!==t?((!1===this._argsValues.hasOwnProperty(e)||!0===this._argsValues.hasOwnProperty(e)&&null==this._argsValues[e])&&(this._argsValues[e]=this._webCLGL.createBuffer(u,!1,l),this._argsValues[e].argument=e,s=!0),this._argsValues[e].writeBuffer(t,!1,n)):this._argsValues[e]=null}else void 0!==t&&null!==t&&(this._argsValues[e]=t),s=!0;if(!0===s&&!0===this.calledArgs.hasOwnProperty(e))for(var c=0;c<this.calledArgs[e].length;c++)this.calledArgs[e][c]._argsValues[e]=this._argsValues[e];break}}return t}},{key:"readArg",value:function(e){return this._webCLGL.readBuffer(this._argsValues[e])}},{key:"setIndices",value:function(e){this.CLGL_bufferIndices=this._webCLGL.createBuffer("FLOAT",!1,"VERTEX_INDEX"),this.CLGL_bufferIndices.writeBuffer(e)}},{key:"getCtx",value:function(){return this._webCLGL.getContext()}},{key:"setCtx",value:function(e){this._gl=e}},{key:"getWebCLGL",value:function(){return this._webCLGL}},{key:"onPreProcessKernel",value:function(e,t){this.kernels[e].onpre=t}},{key:"onPostProcessKernel",value:function(e,t){this.kernels[e].onpost=t}},{key:"enableKernel",value:function(e){this.kernels["0"|e.toString()].enabled=!0}},{key:"disableKernel",value:function(e){this.kernels["0"|e.toString()].enabled=!1}},{key:"getKernel",value:function(e){for(var t in this.kernels)if(t===e)return this.kernels[t];return null}},{key:"getAllKernels",value:function(){return this.kernels}},{key:"onPreProcessGraphic",value:function(e,t){this.vertexFragmentPrograms[e].onpre=t}},{key:"onPostProcessGraphic",value:function(e,t){this.vertexFragmentPrograms[e].onpost=t}},{key:"enableGraphic",value:function(e){this.vertexFragmentPrograms["0"|e.toString()].enabled=!0}},{key:"disableGraphic",value:function(e){this.vertexFragmentPrograms["0"|e.toString()].enabled=!1}},{key:"getVertexFragmentProgram",value:function(e){for(var t in this.vertexFragmentPrograms)if(t===e)return this.vertexFragmentPrograms[t];return null}},{key:"getAllVertexFragmentProgram",value:function(){return this.vertexFragmentPrograms}},{key:"processKernel",value:function(e,t,n){if(!0===e.enabled){if(void 0!==n&&null!==n&&!0===n&&(this.arrMakeCopy=[]),!0===e.depthTest?this._gl.enable(this._gl.DEPTH_TEST):this._gl.disable(this._gl.DEPTH_TEST),!0===e.blend?this._gl.enable(this._gl.BLEND):this._gl.disable(this._gl.BLEND),this._gl.blendFunc(this._gl[e.blendSrcMode],this._gl[e.blendDstMode]),this._gl.blendEquation(this._gl[e.blendEquation]),void 0!==e.onpre&&null!==e.onpre&&e.onpre(),void 0===t||null===t||!0===t){for(var r=!1,o=0;o<e.output.length;o++)if(null!=e.output[o]&&!0===e.outputTempModes[o]){r=!0;break}!0===r?(this._webCLGL.enqueueNDRangeKernel(e,l.WebCLGLUtils.getOutputBuffers(e,this._argsValues),!0,this._argsValues),this.arrMakeCopy.push(e)):this._webCLGL.enqueueNDRangeKernel(e,l.WebCLGLUtils.getOutputBuffers(e,this._argsValues),!1,this._argsValues)}else this._webCLGL.enqueueNDRangeKernel(e,l.WebCLGLUtils.getOutputBuffers(e,this._argsValues),!1,this._argsValues);void 0!==e.onpost&&null!==e.onpost&&e.onpost(),void 0!==n&&null!==n&&!0===n&&this.processCopies()}}},{key:"processCopies",value:function(e){for(var t=0;t<this.arrMakeCopy.length;t++)this._webCLGL.copy(this.arrMakeCopy[t],l.WebCLGLUtils.getOutputBuffers(this.arrMakeCopy[t],this._argsValues))}},{key:"processKernels",value:function(e){this.arrMakeCopy=[];for(var t in this.kernels)this.processKernel(this.kernels[t],e);this.processCopies()}},{key:"processGraphic",value:function(e){var t=[];for(var n in this.vertexFragmentPrograms){var r=this.vertexFragmentPrograms[n];if(!0===r.enabled){var o=void 0!==e&&null!==e||void 0===this.CLGL_bufferIndices||null===this.CLGL_bufferIndices?this._argsValues[e]:this.CLGL_bufferIndices;if(void 0!==o&&null!==o&&o.length>0){!0===r.depthTest?this._gl.enable(this._gl.DEPTH_TEST):this._gl.disable(this._gl.DEPTH_TEST),!0===r.blend?this._gl.enable(this._gl.BLEND):this._gl.disable(this._gl.BLEND),this._gl.blendFunc(this._gl[r.blendSrcMode],this._gl[r.blendDstMode]),this._gl.blendEquation(this._gl[r.blendEquation]),void 0!==r.onpre&&null!==r.onpre&&r.onpre();for(var i=!1,s=0;s<r.output.length;s++)if(null!=r.output[s]&&!0===r.outputTempModes[s]){i=!0;break}!0===i?(this._webCLGL.enqueueVertexFragmentProgram(r,o,r.drawMode,l.WebCLGLUtils.getOutputBuffers(r,this._argsValues),!0,this._argsValues),t.push(r)):this._webCLGL.enqueueVertexFragmentProgram(r,o,r.drawMode,l.WebCLGLUtils.getOutputBuffers(r,this._argsValues),!1,this._argsValues),void 0!==r.onpost&&null!==r.onpost&&r.onpost()}}}for(var a=0;a<t.length;a++)this._webCLGL.copy(t[a],l.WebCLGLUtils.getOutputBuffers(t[a],this._argsValues))}},{key:"ini",value:function(){var e=arguments[0],t=void 0,n=void 0,r=void 0;e.length>3?(this._args=e[0],t=e[1],n=e[2],r=e[3]):(this._args=e[0],t=e[1],n="FLOAT",r=e[2]);var o=0;for(var i in this._args){var s=this._args[i];this.setArg(i.split(" ")[1],s),0===o&&(s instanceof Array||s instanceof Float32Array||s instanceof Uint8Array||s instanceof HTMLImageElement)&&(o=s.length)}return"FLOAT"===n?this.addArg("float* result"):this.addArg("float4* result"),this.setArg("result",new Float32Array(o),null,n),this.addKernel({type:"KERNEL",name:"SIMPLE_KERNEL",viewSource:!1,config:[t,["result"],"",r]}),this.processKernels(),this._webCLGL.readBuffer(this._argsValues.result)}},{key:"iniG",value:function(){this._webCLGL.getContext().depthFunc(this._webCLGL.getContext().LEQUAL),this._webCLGL.getContext().clearDepth(1);var e=arguments[0];this._args=e[1];for(var t=2;t<e.length;t++)"KERNEL"===e[t].type?this.addKernel(e[t]):"GRAPHIC"===e[t].type&&this.addGraphic(e[t]);for(var n in this._args){var r=this._args[n];"indices"===n?null!==r&&this.setIndices(r):this.setArg(n.split(" ")[1],r)}}}]),e}();t.WebCLGLFor=u,n.exports.WebCLGLFor=u,t.gpufor=i,n.exports.gpufor=i}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./WebCLGL.class":1,"./WebCLGLUtils.class":5}],4:[function(e,n,r){(function(t){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(r,"__esModule",{value:!0}),r.WebCLGLKernel=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./WebCLGLUtils.class"),a=r.WebCLGLKernel=function(){function e(t,n,r){o(this,e),this._gl=t;var i=this._gl.getShaderPrecisionFormat(this._gl.FRAGMENT_SHADER,this._gl.HIGH_FLOAT);this._precision=0!==i.precision?"precision highp float;\n\nprecision highp int;\n\n":"precision lowp float;\n\nprecision lowp int;\n\n";var s=this._gl.getExtension("WEBGL_draw_buffers");this._maxDrawBuffers=null,null!=s&&(this._maxDrawBuffers=this._gl.getParameter(s.MAX_DRAW_BUFFERS_WEBGL)),this.name="",this.enabled=!0,this.depthTest=null,this.blend=null,this.blendSrcMode=null,this.blendDstMode=null,this.blendEquation=null,this.onpre=null,this.onpost=null,this.viewSource=!1,this.in_values={},this.output=null,this.outputTempModes=null,this.fBuffer=null,this.fBufferTemp=null,this.fBufferLength=0,this.fBufferCount=0,void 0!==n&&null!==n&&this.setKernelSource(n,r)}return i(e,[{key:"setKernelSource",value:function(e,t){for(var n=function(){var e=this._precision+"attribute vec3 aVertexPosition;\nvarying vec2 global_id;\nvoid main(void) {\ngl_Position = vec4(aVertexPosition, 1.0);\nglobal_id = aVertexPosition.xy*0.5+0.5;\n}\n",t="#extension GL_EXT_draw_buffers : require\n"+this._precision+s.WebCLGLUtils.lines_fragment_attrs(this.in_values)+"varying vec2 global_id;\nuniform float uBufferWidth;vec2 get_global_id() {\nreturn global_id;\n}\n"+s.WebCLGLUtils.get_global_id3_GLSLFunctionString()+s.WebCLGLUtils.get_global_id2_GLSLFunctionString()+this._head+"void main(void) {\n"+s.WebCLGLUtils.lines_drawBuffersInit(8)+this._source+s.WebCLGLUtils.lines_drawBuffersWrite(8)+"}\n";this.kernel=this._gl.createProgram(),(new s.WebCLGLUtils).createShader(this._gl,"WEBCLGL",e,t,this.kernel),this.attr_VertexPos=this._gl.getAttribLocation(this.kernel,"aVertexPosition"),this.uBufferWidth=this._gl.getUniformLocation(this.kernel,"uBufferWidth");for(var n in this.in_values){var r={float4_fromSampler:"SAMPLER",float_fromSampler:"SAMPLER",float:"UNIFORM",float4:"UNIFORM",mat4:"UNIFORM"}[this.in_values[n].type];s.WebCLGLUtils.checkArgNameInitialization(this.in_values,n),this.in_values[n].location=[this._gl.getUniformLocation(this.kernel,n.replace(/\[\d.*/,""))],this.in_values[n].expectedMode=r}return"VERTEX PROGRAM\n"+e+"\n FRAGMENT PROGRAM\n"+t}.bind(this),r=e.split(")")[0].split("(")[1].split(","),o=0,i=r.length;o<i;o++)if(null!==r[o].match(/\*/gm)){var a=r[o].split("*")[1].trim();s.WebCLGLUtils.checkArgNameInitialization(this.in_values,a),null!=r[o].match(/float4/gm)?this.in_values[a].type="float4_fromSampler":null!=r[o].match(/float/gm)&&(this.in_values[a].type="float_fromSampler")}else if(""!==r[o]){var l=r[o].split(" ")[1].trim();for(var u in this.in_values)if(u.replace(/\[\d.*/,"")===l){l=u;break}s.WebCLGLUtils.checkArgNameInitialization(this.in_values,l),null!=r[o].match(/float4/gm)?this.in_values[l].type="float4":null!=r[o].match(/float/gm)?this.in_values[l].type="float":null!=r[o].match(/mat4/gm)&&(this.in_values[l].type="mat4")}this._head=void 0!==t&&null!==t?t:"",this._head=this._head.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._head=s.WebCLGLUtils.parseSource(this._head,this.in_values),this._source=e.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._source=this._source.replace(/^\w* \w*\([\w\s\*,]*\) {/gi,"").replace(/}(\s|\t)*$/gi,""),this._source=s.WebCLGLUtils.parseSource(this._source,this.in_values);var c=n();!0===this.viewSource&&(console.log("%c KERNEL: "+this.name,"font-size: 20px; color: blue"),console.log("%c WEBCLGL --------------------------------","color: gray"),console.log("%c "+t+e,"color: gray"),console.log("%c TRANSLATED WEBGL ------------------------------","color: darkgray"),console.log("%c "+c,"color: darkgray"))}}]),e}();t.WebCLGLKernel=a,n.exports.WebCLGLKernel=a}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./WebCLGLUtils.class":5}],5:[function(e,n,r){(function(e){"use strict";function t(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(r,"__esModule",{value:!0});var o=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),i=r.WebCLGLUtils=function(){function e(){t(this,e)}return o(e,[{key:"loadQuad",value:function(e,t,n){var r=void 0===t||null===t?.5:t,o=void 0===n||null===n?.5:n;this.vertexArray=[-r,-o,0,r,-o,0,r,o,0,-r,o,0],this.textureArray=[0,0,0,1,0,0,1,1,0,0,1,0],this.indexArray=[0,1,2,0,2,3];var i={};return i.vertexArray=this.vertexArray,i.textureArray=this.textureArray,i.indexArray=this.indexArray,i}},{key:"createShader",value:function(e,t,n,r,o){var i=!1,s=!1,a=function(t,n){console.log(t);for(var r=[],o=t.split("\n"),i=0,s=o.length;i<s;i++)if(null!=o[i].match(/^ERROR/gim)){var a=o[i].split(":"),l=parseInt(a[2]);r.push([l,o[i]])}var u=e.getShaderSource(n).split("\n");u.unshift("");for(var c=0,h=u.length;c<h;c++){for(var f=!1,d="",g=0,_=r.length;g<_;g++)if(c===r[g][0]){f=!0,d=r[g][1];break}!1===f?console.log("%c"+c+" %c"+u[c],"color:black","color:blue"):console.log("%c►►%c"+c+" %c"+u[c]+"\n%c"+d,"color:red","color:black","color:blue","color:red")}}.bind(this),l=e.createShader(e.VERTEX_SHADER);if(e.shaderSource(l,n),e.compileShader(l),e.getShaderParameter(l,e.COMPILE_STATUS))e.attachShader(o,l),i=!0;else{var u=e.getShaderInfoLog(l);console.log("%c"+t+" ERROR (vertex program)","color:red"),void 0!==u&&null!==u&&a(u,l)}var c=e.createShader(e.FRAGMENT_SHADER);if(e.shaderSource(c,r),e.compileShader(c),e.getShaderParameter(c,e.COMPILE_STATUS))e.attachShader(o,c),s=!0;else{var h=e.getShaderInfoLog(c);console.log("%c"+t+" ERROR (fragment program)","color:red"),void 0!==h&&null!==h&&a(h,c)}if(!0===i&&!0===s){if(e.linkProgram(o),e.getProgramParameter(o,e.LINK_STATUS))return!0;console.log("Error shader program "+t+":\n ");var f=e.getProgramInfoLog(o);return void 0!==f&&null!==f&&console.log(f),!1}return!1}},{key:"pack",value:function(e){var t=[1/255,1/255,1/255,0],n=e,r=this.fract(255*n),o=this.fract(255*r),i=[n,r,o,this.fract(255*o)],s=[i[1]*t[0],i[2]*t[1],i[3]*t[2],i[3]*t[3]];return[i[0]-s[0],i[1]-s[1],i[2]-s[2],i[3]-s[3]]}},{key:"unpack",value:function(e){var t=[1,1/255,1/65025,1/16581375];return this.dot4(e,t)}}],[{key:"getWebGLContextFromCanvas",value:function(e,t){var n=null;if(null==n)try{n=void 0===t||null===t?e.getContext("webgl"):e.getContext("webgl",t),console.log(null==n?"no webgl":"using webgl")}catch(e){n=null}if(null==n)try{n=void 0===t||null===t?e.getContext("experimental-webgl"):e.getContext("experimental-webgl",t),console.log(null==n?"no experimental-webgl":"using experimental-webgl")}catch(e){n=null}return null==n&&(n=!1),n}},{key:"getUint8ArrayFromHTMLImageElement",value:function(e){var t=document.createElement("canvas");t.width=e.width,t.height=e.height;var n=t.getContext("2d");return n.drawImage(e,0,0),n.getImageData(0,0,e.width,e.height).data}},{key:"dot4",value:function(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]+e[3]*t[3]}},{key:"fract",value:function(e){return e>0?e-Math.floor(e):e-Math.ceil(e)}},{key:"packGLSLFunctionString",value:function(){return"vec4 pack (float depth) {\nconst vec4 bias = vec4(1.0 / 255.0,\n1.0 / 255.0,\n1.0 / 255.0,\n0.0);\nfloat r = depth;\nfloat g = fract(r * 255.0);\nfloat b = fract(g * 255.0);\nfloat a = fract(b * 255.0);\nvec4 colour = vec4(r, g, b, a);\nreturn colour - (colour.yzww * bias);\n}\n"}},{key:"unpackGLSLFunctionString",value:function(){return"float unpack (vec4 colour) {\nconst vec4 bitShifts = vec4(1.0,\n1.0 / 255.0,\n1.0 / (255.0 * 255.0),\n1.0 / (255.0 * 255.0 * 255.0));\nreturn dot(colour, bitShifts);\n}\n"}},{key:"getOutputBuffers",value:function(e,t){var n=null;if(void 0!==e.output&&null!==e.output)if(n=[],null!=e.output[0])for(var r=0;r<e.output.length;r++)n[r]=t[e.output[r]];else n=null;return n}},{key:"parseSource",value:function(e,t){for(var n in t){var r=new RegExp(n+"\\[(?!\\d).*?\\]","gm"),o=e.match(r);if(null!=o)for(var i=0,s=o.length;i<s;i++){var a=new RegExp("```(s|\t)*gl.*"+o[i]+".*```[^```(s|\t)*gl]","gm");if(null==e.match(a)){var l=o[i].split("[")[0],u=o[i].split("[")[1].split("]")[0];e={float4_fromSampler:e.replace(l+"["+u+"]","texture2D("+l+","+u+")"),float_fromSampler:e.replace(l+"["+u+"]","texture2D("+l+","+u+").x"),float4_fromAttr:e.replace(l+"["+u+"]",l),float_fromAttr:e.replace(l+"["+u+"]",l)}[t[n].type]}}}return e=e.replace(/```(\s|\t)*gl/gi,"").replace(/```/gi,"").replace(/;/gi,";\n").replace(/}/gi,"}\n").replace(/{/gi,"{\n")}},{key:"lines_vertex_attrs",value:function(e){var t="";for(var n in e)t+={float4_fromSampler:"uniform sampler2D "+n+";",float_fromSampler:"uniform sampler2D "+n+";",float4_fromAttr:"attribute vec4 "+n+";",float_fromAttr:"attribute float "+n+";",float:"uniform float "+n+";",float4:"uniform vec4 "+n+";",mat4:"uniform mat4 "+n+";"}[e[n].type]+"\n";return t}},{key:"lines_fragment_attrs",value:function(e){var t="";for(var n in e)t+={float4_fromSampler:"uniform sampler2D "+n+";",float_fromSampler:"uniform sampler2D "+n+";",float:"uniform float "+n+";",float4:"uniform vec4 "+n+";",mat4:"uniform mat4 "+n+";"}[e[n].type]+"\n";return t}},{key:"lines_drawBuffersInit",value:function(e){for(var t="",n=0,r=e;n<r;n++)t+="float out"+n+"_float = -999.99989;\nvec4 out"+n+"_float4;\n";return t}},{key:"lines_drawBuffersWriteInit",value:function(e){for(var t="",n=0,r=e;n<r;n++)t+="layout(location = "+n+") out vec4 outCol"+n+";\n";return t}},{key:"lines_drawBuffersWrite",value:function(e){for(var t="",n=0,r=e;n<r;n++)t+="if(out"+n+"_float != -999.99989) gl_FragData["+n+"] = vec4(out"+n+"_float,0.0,0.0,1.0);\n else gl_FragData["+n+"] = out"+n+"_float4;\n";return t}},{key:"checkArgNameInitialization",value:function(e,t){!1===e.hasOwnProperty(t)&&(e[t]={type:null,expectedMode:null,location:null})}},{key:"get_global_id3_GLSLFunctionString",value:function(){return"vec2 get_global_id(float id, float bufferWidth, float geometryLength) {\nfloat texelSize = 1.0/bufferWidth;float num = (id*geometryLength)/bufferWidth;float column = fract(num)+(texelSize/2.0);float row = (floor(num)/bufferWidth)+(texelSize/2.0);return vec2(column, row);}\n"}},{key:"get_global_id2_GLSLFunctionString",value:function(){return"vec2 get_global_id(vec2 id, float bufferWidth) {\nfloat texelSize = 1.0/bufferWidth;float column = (id.x/bufferWidth)+(texelSize/2.0);float row = (id.y/bufferWidth)+(texelSize/2.0);return vec2(column, row);}\n"}}]),e}();e.WebCLGLUtils=i,n.exports.WebCLGLUtils=i}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],6:[function(e,n,r){(function(t){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(r,"__esModule",{value:!0}),r.WebCLGLVertexFragmentProgram=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./WebCLGLUtils.class"),a=r.WebCLGLVertexFragmentProgram=function(){function e(t,n,r,i,s){o(this,e),this._gl=t;var a=this._gl.getShaderPrecisionFormat(this._gl.FRAGMENT_SHADER,this._gl.HIGH_FLOAT);this._precision=0!==a.precision?"precision highp float;\n\nprecision highp int;\n\n":"precision lowp float;\n\nprecision lowp int;\n\n";var l=this._gl.getExtension("WEBGL_draw_buffers");this._maxDrawBuffers=null,null!=l&&(this._maxDrawBuffers=this._gl.getParameter(l.MAX_DRAW_BUFFERS_WEBGL)),this.name="",this.viewSource=!1,this.in_vertex_values={},this.in_fragment_values={},this._vertexP_ready=!1,this._fragmentP_ready=!1,this._vertexHead=null,this._vertexSource=null,this._fragmentHead=null,this._fragmentSource=null,this.output=null,this.outputTempModes=null,this.fBuffer=null,this.fBufferTemp=null,this.drawMode=4,void 0!==n&&null!==n&&this.setVertexSource(n,r),void 0!==i&&null!==i&&this.setFragmentSource(i,s)}return i(e,[{key:"compileVertexFragmentSource",value:function(){var e=this._precision+"uniform float uOffset;\nuniform float uBufferWidth;"+s.WebCLGLUtils.lines_vertex_attrs(this.in_vertex_values)+s.WebCLGLUtils.unpackGLSLFunctionString()+s.WebCLGLUtils.get_global_id3_GLSLFunctionString()+s.WebCLGLUtils.get_global_id2_GLSLFunctionString()+this._vertexHead+"void main(void) {\n"+this._vertexSource+"}\n",t="#extension GL_EXT_draw_buffers : require\n"+this._precision+s.WebCLGLUtils.lines_fragment_attrs(this.in_fragment_values)+s.WebCLGLUtils.get_global_id3_GLSLFunctionString()+s.WebCLGLUtils.get_global_id2_GLSLFunctionString()+this._fragmentHead+"void main(void) {\n"+s.WebCLGLUtils.lines_drawBuffersInit(8)+this._fragmentSource+s.WebCLGLUtils.lines_drawBuffersWrite(8)+"}\n";this.vertexFragmentProgram=this._gl.createProgram(),(new s.WebCLGLUtils).createShader(this._gl,"WEBCLGL VERTEX FRAGMENT PROGRAM",e,t,this.vertexFragmentProgram),this.uOffset=this._gl.getUniformLocation(this.vertexFragmentProgram,"uOffset"),this.uBufferWidth=this._gl.getUniformLocation(this.vertexFragmentProgram,"uBufferWidth");for(var n in this.in_vertex_values){var r={float4_fromSampler:"SAMPLER",float_fromSampler:"SAMPLER",float4_fromAttr:"ATTRIBUTE",float_fromAttr:"ATTRIBUTE",float:"UNIFORM",float4:"UNIFORM",mat4:"UNIFORM"}[this.in_vertex_values[n].type];s.WebCLGLUtils.checkArgNameInitialization(this.in_vertex_values,n);var o="ATTRIBUTE"===r?this._gl.getAttribLocation(this.vertexFragmentProgram,n):this._gl.getUniformLocation(this.vertexFragmentProgram,n.replace(/\[\d.*/,""));this.in_vertex_values[n].location=[o],this.in_vertex_values[n].expectedMode=r}for(var i in this.in_fragment_values){var a={float4_fromSampler:"SAMPLER",float_fromSampler:"SAMPLER",float:"UNIFORM",float4:"UNIFORM",mat4:"UNIFORM"}[this.in_fragment_values[i].type];s.WebCLGLUtils.checkArgNameInitialization(this.in_fragment_values,i),this.in_fragment_values[i].location=[this._gl.getUniformLocation(this.vertexFragmentProgram,i.replace(/\[\d.*/,""))],this.in_fragment_values[i].expectedMode=a}return"VERTEX PROGRAM\n"+e+"\n FRAGMENT PROGRAM\n"+t}},{key:"setVertexSource",value:function(e,t){for(var n=e.split(")")[0].split("(")[1].split(","),r=0,o=n.length;r<o;r++)if(null!==n[r].match(/\*attr/gm)){var i=n[r].split("*attr")[1].trim();s.WebCLGLUtils.checkArgNameInitialization(this.in_vertex_values,i),null!=n[r].match(/float4/gm)?this.in_vertex_values[i].type="float4_fromAttr":null!=n[r].match(/float/gm)&&(this.in_vertex_values[i].type="float_fromAttr")}else if(null!==n[r].match(/\*/gm)){var a=n[r].split("*")[1].trim();s.WebCLGLUtils.checkArgNameInitialization(this.in_vertex_values,a),null!=n[r].match(/float4/gm)?this.in_vertex_values[a].type="float4_fromSampler":null!=n[r].match(/float/gm)&&(this.in_vertex_values[a].type="float_fromSampler")}else if(""!==n[r]){var l=n[r].split(" ")[1].trim();for(var u in this.in_vertex_values)if(u.replace(/\[\d.*/,"")===l){l=u;break}s.WebCLGLUtils.checkArgNameInitialization(this.in_vertex_values,l),null!=n[r].match(/float4/gm)?this.in_vertex_values[l].type="float4":null!=n[r].match(/float/gm)?this.in_vertex_values[l].type="float":null!=n[r].match(/mat4/gm)&&(this.in_vertex_values[l].type="mat4")}if(this._vertexHead=void 0!==t&&null!==t?t:"",this._vertexHead=this._vertexHead.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._vertexHead=s.WebCLGLUtils.parseSource(this._vertexHead,this.in_vertex_values),this._vertexSource=e.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._vertexSource=this._vertexSource.replace(/^\w* \w*\([\w\s\*,]*\) {/gi,"").replace(/}(\s|\t)*$/gi,""),this._vertexSource=s.WebCLGLUtils.parseSource(this._vertexSource,this.in_vertex_values),this._vertexP_ready=!0,!0===this._fragmentP_ready){var c=this.compileVertexFragmentSource();!0===this.viewSource&&(console.log("%c VFP: "+this.name,"font-size: 20px; color: green"),console.log("%c WEBCLGL --------------------------------","color: gray"),console.log("%c "+t+e,"color: gray"),console.log("%c TRANSLATED WEBGL ------------------------------","color: darkgray"),console.log("%c "+c,"color: darkgray"))}}},{key:"setFragmentSource",value:function(e,t){for(var n=e.split(")")[0].split("(")[1].split(","),r=0,o=n.length;r<o;r++)if(null!==n[r].match(/\*/gm)){var i=n[r].split("*")[1].trim();s.WebCLGLUtils.checkArgNameInitialization(this.in_fragment_values,i),null!=n[r].match(/float4/gm)?this.in_fragment_values[i].type="float4_fromSampler":null!=n[r].match(/float/gm)&&(this.in_fragment_values[i].type="float_fromSampler")}else if(""!==n[r]){var a=n[r].split(" ")[1].trim();for(var l in this.in_fragment_values)if(l.replace(/\[\d.*/,"")===a){a=l;break}s.WebCLGLUtils.checkArgNameInitialization(this.in_fragment_values,a),null!=n[r].match(/float4/gm)?this.in_fragment_values[a].type="float4":null!=n[r].match(/float/gm)?this.in_fragment_values[a].type="float":null!=n[r].match(/mat4/gm)&&(this.in_fragment_values[a].type="mat4")}if(this._fragmentHead=void 0!==t&&null!==t?t:"",this._fragmentHead=this._fragmentHead.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._fragmentHead=s.WebCLGLUtils.parseSource(this._fragmentHead,this.in_fragment_values),this._fragmentSource=e.replace(/\r\n/gi,"").replace(/\r/gi,"").replace(/\n/gi,""),this._fragmentSource=this._fragmentSource.replace(/^\w* \w*\([\w\s\*,]*\) {/gi,"").replace(/}(\s|\t)*$/gi,""),this._fragmentSource=s.WebCLGLUtils.parseSource(this._fragmentSource,this.in_fragment_values),this._fragmentP_ready=!0,!0===this._vertexP_ready){var u=this.compileVertexFragmentSource();!0===this.viewSource&&(console.log("%c VFP: ","font-size: 20px; color: green"),console.log("%c WEBCLGL --------------------------------","color: gray"),console.log("%c "+t+e,"color: gray"),console.log("%c TRANSLATED WEBGL ------------------------------","color: darkgray"),console.log("%c "+u,"color: darkgray"))}}}]),e}();t.WebCLGLVertexFragmentProgram=a,n.exports.WebCLGLVertexFragmentProgram=a}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./WebCLGLUtils.class":5}],7:[function(e,n,r){(function(t){"use strict";e("./WebCLGL.class"),e("./WebCLGLBuffer.class"),e("./WebCLGLFor.class"),e("./WebCLGLKernel.class"),e("./WebCLGLUtils.class"),e("./WebCLGLVertexFragmentProgram.class"),n.exports.WebCLGL=t.WebCLGL=WebCLGL,n.exports.WebCLGLBuffer=t.WebCLGLBuffer=WebCLGLBuffer,n.exports.WebCLGLFor=t.WebCLGLFor=WebCLGLFor,n.exports.WebCLGLKernel=t.WebCLGLKernel=WebCLGLKernel,n.exports.WebCLGLUtils=t.WebCLGLUtils=WebCLGLUtils,n.exports.WebCLGLVertexFragmentProgram=t.WebCLGLVertexFragmentProgram=WebCLGLVertexFragmentProgram}).call(this,void 0!==t?t:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./WebCLGL.class":1,"./WebCLGLBuffer.class":2,"./WebCLGLFor.class":3,"./WebCLGLKernel.class":4,"./WebCLGLUtils.class":5,"./WebCLGLVertexFragmentProgram.class":6}]},{},[7])}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],2:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.ArrayGenerator=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./Utils.class"),a=n.ArrayGenerator=function(){function e(){o(this,e)}return i(e,[{key:"volumeArray",value:function(e){if(this.arrayNodeDestination=[],this.arrayNodeVertexColor=[],this.vo=e,this.vo instanceof StormVoxelizator==!1)return alert("You must select a voxelizator object with albedo fillmode enabled."),!1;if(void 0===this.vo.image3D_VoxelsColor||null===this.vo.image3D_VoxelsColor)return alert("You must select a voxelizator object with albedo fillmode enabled."),!1;this.data=this.vo.clglBuff_VoxelsColor.items[0].inData;for(var t=0,n=0,r=this.data.length/4;n<r;n++){var o=4*n;this.data[o+3]>0&&t++}var i=(this.currentNodeId-1)/t;this.incremNodesCell=0;var s=-1;this.currentVoxelCell=null,this.CCX=0,this.CCY=0,this.CCZ=0,this.CCXMAX=this.vo.resolution-1,this.CCYMAX=this.vo.resolution-1,this.CCZMAX=this.vo.resolution-1;for(var a=void 0,l=void 0,u=function(){for(;;){if(this.CCX===this.CCXMAX&&this.CCZ===this.CCZMAX&&this.CCY===this.CCYMAX)break;if(this.CCX===this.CCXMAX&&this.CCZ===this.CCZMAX?(this.CCX=0,this.CCZ=0,this.CCY++):this.CCX===this.CCXMAX?(this.CCX=0,this.CCZ++):this.CCX++,this.currentVoxelCell=this.CCY*(this.vo.resolution*this.vo.resolution)+this.CCZ*this.vo.resolution+this.CCX,this.data[4*this.currentVoxelCell+3]>0&&(this.incremNodesCell+=i,this.incremNodesCell>=1)){this.incremNodesCell-=1;break}}}.bind(this),c=0;c<this.arrayNodeId.length;c++)s!==this.arrayNodeId[c]?(s=this.arrayNodeId[c],this.incremNodesCell>=1?this.incremNodesCell-=1:u(),a=(a=(a=$V3([0,0,0]).add($V3([-this.vo.size/2,-this.vo.size/2,-this.vo.size/2]))).add($V3([this.vo.cs*this.CCX*1,this.vo.cs*this.CCY*1,this.vo.cs*(this.CCZMAX-this.CCZ)*1]))).add($V3([this.vo.cs*Math.random(),this.vo.cs*Math.random(),this.vo.cs*Math.random()])),l=$V3([this.data[4*this.currentVoxelCell]/255,this.data[4*this.currentVoxelCell+1]/255,this.data[4*this.currentVoxelCell+2]/255]),this.arrayNodeDestination.push(a.e[0],a.e[1],a.e[2],1),this.arrayNodeVertexColor.push(l.e[0],l.e[1],l.e[2],1)):(this.arrayNodeDestination.push(a.e[0],a.e[1],a.e[2],1),this.arrayNodeVertexColor.push(l.e[0],l.e[1],l.e[2],1));comp_renderer_nodes.setArg("dest",function(){return this.arrayNodeDestination}.bind(this),this.splitNodes),comp_renderer_nodes.setArg("nodeVertexCol",function(){return this.arrayNodeVertexColor}.bind(this),this.splitNodes)}}],[{key:"randomArray",value:function(e){for(var t=[],n=0;n<e.count;n++)"float"===e.type?t.push(-e.offset/2+Math.random()*e.offset):t.push(-e.offset/2+Math.random()*e.offset,-e.offset/2+Math.random()*e.offset,-e.offset/2+Math.random()*e.offset,-e.offset/2+Math.random()*e.offset);return t}},{key:"widthHeightArray",value:function(e){for(var t=[],n=e.width*e.height,r=e.count/n,o=0,i=0,s=0,a=void 0!==e.spacing&&null!==e.spacing?e.spacing:.01,l=0;l<e.count;l++)o>=r&&(++i>e.width-1&&(i=0,s++),o-=r),o+=1,t.push(i*a,0,s*a,1);return t}},{key:"float4Array",value:function(e){for(var t=[],n=0;n<e.count;n++)t.push(e.float4[0],e.float4[1],e.float4[2],e.float4[3]);return t}},{key:"sphericalArray",value:function(e){for(var t=[],n=void 0===e?1:e.radius,r=0;r<e.count;r++){var o=360*Math.random(),i=180*Math.random();t.push(Math.cos(o)*Math.abs(Math.sin(i))*n,Math.cos(i)*n*Math.random(),Math.sin(o)*Math.abs(Math.sin(i))*n,1)}return t}},{key:"hemArray",value:function(e){for(var t=[],n=void 0===e?1:e.radius,r=0;r<e.count;r++){var o=(new s.Utils).getVector(e.normalVector,e.degrees);t.push(o.e[0]*n,o.e[1]*n,o.e[2]*n,1)}return t}},{key:"imageArray",value:function(e){for(var t=s.Utils.getUint8ArrayFromHTMLImageElement(e.image),n=[],r=0;r<e.count;r++){var o=4*r;n.push(parseFloat(t[o]/255),parseFloat(t[o+1]/255),parseFloat(t[o+2]/255),parseFloat(t[o+3]/255))}return n}}]),e}();r.ArrayGenerator=a,t.exports.ArrayGenerator=a}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Utils.class":22}],3:[function(e,t,n){(function(e){"use strict";function r(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0});var o=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),i=n.Component=function(){function e(){r(this,e),this.type=null,this.node=null}return o(e,[{key:"initialize",value:function(e,t){}},{key:"tick",value:function(e){}}]),e}();e.Component=i,t.exports.Component=i}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],4:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentControllerTransformTarget=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./Constants"),c=n.ComponentControllerTransformTarget=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=u.Constants.COMPONENT_TYPES.CONTROLLER_TRANSFORM_TARGET,e.node=null,e.gl=null,e.comp_transformTarget=null,e.comp_projection=null,e._vel=.1,e.forward=0,e.backwardE=0,e.leftE=0,e.rightE=0,e.frontE=0,e.backE=0,e.leftButton=0,e.middleButton=0,e.rightButton=0,e.leftButtonAction="ORBIT",e.middleButtonAction="PAN",e.rightButtonAction="ZOOM",e.lastX=0,e.lastY=0,e.lockRotX=!1,e.lockRotY=!1,e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t,this.comp_transformTarget=this.node.getComponent(u.Constants.COMPONENT_TYPES.TRANSFORM_TARGET),this.comp_projection=this.node.getComponent(u.Constants.COMPONENT_TYPES.PROJECTION)}},{key:"setVelocity",value:function(e){this._vel=e}},{key:"lockRotX",value:function(){this.lockRotX=!0}},{key:"unlockRotX",value:function(){this.lockRotX=!1}},{key:"isLockRotX",value:function(){return this.lockRotX}},{key:"lockRotY",value:function(){this.lockRotY=!0}},{key:"unlockRotY",value:function(){this.lockRotY=!1}},{key:"isLockRotY",value:function(){return this.lockRotY}},{key:"forward",value:function(){this.forward=1}},{key:"backward",value:function(){this.backwardE=1}},{key:"strafeLeft",value:function(){this.leftE=1}},{key:"strafeRight",value:function(){this.rightE=1}},{key:"strafeFront",value:function(){this.frontE=1}},{key:"strafeBack",value:function(){this.backE=1}},{key:"stopForward",value:function(){this.forward=0}},{key:"stopBackward",value:function(){this.backwardE=0}},{key:"stopStrafeLeft",value:function(){this.leftE=0}},{key:"stopStrafeRight",value:function(){this.rightE=0}},{key:"stopStrafeFront",value:function(){this.frontE=0}},{key:"stopStrafeBack",value:function(){this.backE=0}},{key:"mouseDown",value:function(e){this.lastX=e.screenX,this.lastY=e.screenY,0===e.button&&(this.leftButton=1),1===e.button&&(this.middleButton=1),2===e.button&&(this.rightButton=1),this.updateGoal(e)}},{key:"mouseUp",value:function(e){0===e.button&&(this.leftButton=0),1===e.button&&(this.middleButton=0),2===e.button&&(this.rightButton=0)}},{key:"mouseMove",value:function(e){1!==this.leftButton&&1!==this.middleButton||this.updateGoal(e)}},{key:"isLeftBtnActive",value:function(){return 1===this.leftButton}},{key:"isMiddleBtnActive",value:function(){return 1===this.middleButton}},{key:"isRightBtnActive",value:function(){return 1===this.rightButton}},{key:"setLeftButtonAction",value:function(e){this.leftButtonAction=e}},{key:"setMiddleButtonAction",value:function(e){this.middleButtonAction=e}},{key:"setRightButtonAction",value:function(e){this.rightButtonAction=e}},{key:"tick",value:function(e){var t=$V3([0,0,0]);1===this.forward&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getForward().x(-this._vel))),1===this.backwardE&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getForward().x(this._vel))),1===this.leftE&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getLeft().x(-this._vel))),1===this.rightE&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getLeft().x(this._vel))),1===this.backE&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getUp().x(-this._vel))),1===this.frontE&&(t=t.add(this.comp_transformTarget.getMatrix().inverse().getUp().x(this._vel))),this.comp_transformTarget.setPositionTarget(this.comp_transformTarget.getPositionTarget().add(t)),this.comp_transformTarget.setPositionGoal(this.comp_transformTarget.getPositionGoal().add(t))}},{key:"updateGoal",value:function(e){1===this.middleButton?"PAN"===this.middleButtonAction?this.makePan(e):"ORBIT"===this.middleButtonAction&&this.makeOrbit(e):"PAN"===this.leftButtonAction?this.comp_projection.getProjection()===u.Constants.PROJECTION_TYPES.PERSPECTIVE?this.makeOrbit(e):this.makePan(e):"ORBIT"===this.leftButtonAction&&this.makeOrbit(e),this.lastX=e.screenX,this.lastY=e.screenY}},{key:"makePan",value:function(e){e.preventDefault();var t=this.comp_transformTarget.getMatrix().getLeft().x((this.lastX-e.screenX)*(.005*this.comp_projection.getFov())),n=this.comp_transformTarget.getMatrix().getUp().x((this.lastY-e.screenY)*(-.005*this.comp_projection.getFov())),r=t.add(n.x(-1));this.comp_transformTarget.setPositionGoal(this.comp_transformTarget.getPositionGoal().add(r)),this.comp_transformTarget.setPositionTarget(this.comp_transformTarget.getPositionTarget().add(r))}},{key:"makeOrbit",value:function(e){!1===this.lockRotY&&(this.lastX>e.screenX?this.comp_transformTarget.yaw(.5*-(this.lastX-e.screenX)):this.comp_transformTarget.yaw(.5*(e.screenX-this.lastX))),!1===this.lockRotX&&(this.lastY>e.screenY?this.comp_transformTarget.pitch(.5*(this.lastY-e.screenY)):this.comp_transformTarget.pitch(.5*-(e.screenY-this.lastY)))}}]),t}();r.ComponentControllerTransformTarget=c,t.exports.ComponentControllerTransformTarget=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11}],5:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentKeyboardEvents=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./Constants"),c=n.ComponentKeyboardEvents=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=u.Constants.COMPONENT_TYPES.KEYBOARD_EVENTS,e.node=null,e.gl=null,e._onkeydown=null,e._onkeyup=null,e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t}},{key:"onkeydown",value:function(e){this._onkeydown=e}},{key:"onkeyup",value:function(e){this._onkeyup=e}}]),t}();r.ComponentKeyboardEvents=c,t.exports.ComponentKeyboardEvents=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11}],6:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentMouseEvents=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./Constants"),c=n.ComponentMouseEvents=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=u.Constants.COMPONENT_TYPES.MOUSE_EVENTS,e.node=null,e.gl=null,e._onmousedown=null,e._onmouseup=null,e._onmousemove=null,e._onmousewheel=null,e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t}},{key:"onmousedown",value:function(e){this._onmousedown=e}},{key:"onmouseup",value:function(e){this._onmouseup=e}},{key:"onmousemove",value:function(e){this._onmousemove=e}},{key:"onmousewheel",value:function(e){this._onmousewheel=e}}]),t}();r.ComponentMouseEvents=c,t.exports.ComponentMouseEvents=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11}],7:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentProjection=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./StormMath.class"),c=e("./Constants"),h=n.ComponentProjection=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=c.Constants.COMPONENT_TYPES.PROJECTION,e.node=null,e.gl=null,e.mProjectionMatrix=$M16([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),e.proy=c.Constants.PROJECTION_TYPES.PERSPECTIVE,e._width=512,e._height=512,e._fov=45,e._fovOrtho=20,e._near=.1,e._far=1e4,e._nearOrtho=-1e5,e._farOrtho=1e5,e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t,this.updateProjectionMatrix()}},{key:"getMatrix",value:function(){return this.mProjectionMatrix}},{key:"getProjection",value:function(){return this.proy}},{key:"setProjection",value:function(e){this.proy=e,this.updateProjectionMatrix()}},{key:"setResolution",value:function(e,t){this._width=e,this._height=t,this.updateProjectionMatrix()}},{key:"getResolution",value:function(){return{width:this._width,height:this._height}}},{key:"setFov",value:function(e){this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._fov=e:this._fovOrtho=e,this.updateProjectionMatrix()}},{key:"getFov",value:function(){return this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._fov:this._fovOrtho}},{key:"setNear",value:function(e){this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._near=e:this._nearOrtho=e,this.updateProjectionMatrix()}},{key:"getNear",value:function(){return this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._near:this._nearOrtho}},{key:"setFar",value:function(e){this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._far=e:this._farOrtho=e,this.updateProjectionMatrix()}},{key:"getFar",value:function(){return this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._far:this._farOrtho}},{key:"updateProjectionMatrix",value:function(){var e=this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this._fov:this._fovOrtho,t=this._width/this._height;this.proy===c.Constants.PROJECTION_TYPES.PERSPECTIVE?this.mProjectionMatrix=u.StormM16.setPerspectiveProjection(e,t,this._near,this._far):this.mProjectionMatrix=u.StormM16.setOrthographicProjection(-t*e,t*e,-t*e,t*e,this._nearOrtho,this._farOrtho)}}]),t}();r.ComponentProjection=h,t.exports.ComponentProjection=h}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11,"./StormMath.class":20}],8:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentTransform=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./Constants"),c=n.ComponentTransform=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=u.Constants.COMPONENT_TYPES.TRANSFORM,e.node=null,e.gl=null,e.mModelMatrix_Position=$M16([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),e.mModelMatrix_Rotation=$M16([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t}},{key:"getMatrixPosition",value:function(){return this.mModelMatrix_Position}},{key:"getMatrixRotation",value:function(){return this.mModelMatrix_Rotation}}]),t}();r.ComponentTransform=c,t.exports.ComponentTransform=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11}],9:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.ComponentTransformTarget=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),l=e("./Component.class"),u=e("./Utils.class"),c=e("./StormMath.class"),h=e("./Constants"),f=n.ComponentTransformTarget=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=h.Constants.COMPONENT_TYPES.TRANSFORM_TARGET,e.node=null,e.gl=null,e.mModelMatrix=$M16([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),e.targetDistance=5,e.positionGoal=$V3([0,0,e.targetDistance]),e.positionTarget=$V3([0,0,0]),e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t,this.performMatrix()}},{key:"getMatrix",value:function(){return this.mModelMatrix}},{key:"setTargetDistance",value:function(e){this.targetDistance=e}},{key:"getTargetDistance",value:function(){return this.targetDistance}},{key:"setPositionGoal",value:function(e){this.positionGoal=e,this.performMatrix()}},{key:"getPositionGoal",value:function(){return this.positionGoal}},{key:"setPositionTarget",value:function(e){this.positionTarget=e,this.performMatrix()}},{key:"getPositionTarget",value:function(){return this.positionTarget}},{key:"reset",value:function(){this.positionGoal=$V3([0,0,this.targetDistance]),this.positionTarget=$V3([0,0,0]),this.performMatrix()}},{key:"yaw",value:function(e){var t=this.getPositionGoal().subtract(this.getPositionTarget()),n=u.Utils.cartesianToSpherical(t.normalize()),r=u.Utils.sphericalToCartesian(n.radius,n.lat,n.lng+e);this.setPositionGoal(this.getPositionTarget().add(r.x(this.getTargetDistance())))}},{key:"pitch",value:function(e){var t=this.getPositionGoal().subtract(this.getPositionTarget()),n=u.Utils.cartesianToSpherical(t.normalize()),r=u.Utils.sphericalToCartesian(n.radius,n.lat+e,n.lng);this.setPositionGoal(this.getPositionTarget().add(r.x(this.getTargetDistance())))}},{key:"performMatrix",value:function(){this.mModelMatrix=c.StormM16.makeLookAt(this.getPositionGoal().e[0],this.getPositionGoal().e[1],this.getPositionGoal().e[2],this.getPositionTarget().e[0],this.getPositionTarget().e[1],this.getPositionTarget().e[2],0,1,0)}}]),t}();r.ComponentTransformTarget=f,t.exports.ComponentTransformTarget=f}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11,"./StormMath.class":20,"./Utils.class":22}],10:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function i(e,t){if(!e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return!t||"object"!=typeof t&&"function"!=typeof t?e:t}function s(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function, not "+typeof t);e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,enumerable:!1,writable:!0,configurable:!0}}),t&&(Object.setPrototypeOf?Object.setPrototypeOf(e,t):e.__proto__=t)}Object.defineProperty(n,"__esModule",{value:!0}),n.Component_GPU=void 0;var a=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}();e("webclgl");var l=e("./Component.class"),u=e("./Constants"),c=n.Component_GPU=function(e){function t(){o(this,t);var e=i(this,(t.__proto__||Object.getPrototypeOf(t)).call(this));return e.type=u.Constants.COMPONENT_TYPES.GPU,e.node=null,e.gl=null,e.gpufG=null,e.args={},e}return s(t,l.Component),a(t,[{key:"initialize",value:function(e,t){this.node=e,this.gl=t}},{key:"tick",value:function(e){this.tickArguments(),null!=this.gpufG&&this.gpufG.processKernels(),null!=this.gpufG&&this.gpufG.processGraphic(void 0)}},{key:"setGPUFor",value:function(){var e=this,t=arguments;for(var n in arguments[1]){var r=n.split(" ");if(null!=r&&r.length>1){var o=r[1];this.args[o]={fnvalue:arguments[1][n],updatable:null,splits:null,overrideDimensions:null}}arguments[1][n]=arguments[1][n]()}this.gpufG=new function(){return gpufor.apply(e,t)}}},{key:"getWebCLGL",value:function(){return this.gpufG.getWebCLGL()}},{key:"addArgument",value:function(e,t){this.args[e.split(" ")[1]]={fnvalue:t,updatable:null,splits:null,overrideDimensions:null},this.gpufG.addArg(e)}},{key:"setArg",value:function(e,t,n,r){var o=this.gpufG.setArg(e,t(),n,r);return this.args[e]={fnvalue:t,updatable:null,splits:n,overrideDimensions:r},o}},{key:"getComponentBufferArg",value:function(e,t){this.gpufG.getGPUForArg(e,t.gpufG),this.args[e]={fnvalue:null,updatable:null,splits:null,overrideDimensions:null}}},{key:"getArgs",value:function(){return this.args}},{key:"getAllArgs",value:function(){return this.gpufG.getAllArgs()}},{key:"getBuffers",value:function(){return this.gpufG._argsValues}},{key:"setArgUpdatable",value:function(e,t){this.args[e].updatable=t}},{key:"tickArguments",value:function(){for(var e in this.args)if(!0===this.args[e].updatable){var t=this.args[e];this.gpufG.setArg(e,t.fnvalue(),t.splits,t.overrideDimensions)}}}]),t}();r.Component_GPU=c,t.exports.Component_GPU=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Component.class":3,"./Constants":11,webclgl:1}],11:[function(e,t,n){(function(e){"use strict";Object.defineProperty(n,"__esModule",{value:!0});var r=n.Constants={EVENT_TYPES:{KEY_DOWN:0,KEY_UP:1,MOUSE_DOWN:2,MOUSE_UP:3,MOUSE_MOVE:4,MOUSE_WHEEL:5},COMPONENT_TYPES:{KEYBOARD_EVENTS:0,MOUSE_EVENTS:1,TRANSFORM:2,TRANSFORM_TARGET:3,CONTROLLER_TRANSFORM_TARGET:4,PROJECTION:5,GPU:6},PROJECTION_TYPES:{PERSPECTIVE:0,ORTHO:1},VIEW_TYPES:{LEFT:0,RIGHT:1,FRONT:2,BACK:3,TOP:4,BOTTOM:5},BLENDING_MODES:{ZERO:"ZERO",ONE:"ONE",SRC_COLOR:"SRC_COLOR",ONE_MINUS_SRC_COLOR:"ONE_MINUS_SRC_COLOR",DST_COLOR:"DST_COLOR",ONE_MINUS_DST_COLOR:"ONE_MINUS_DST_COLOR",SRC_ALPHA:"SRC_ALPHA",ONE_MINUS_SRC_ALPHA:"ONE_MINUS_SRC_ALPHA",DST_ALPHA:"DST_ALPHA",ONE_MINUS_DST_ALPHA:"ONE_MINUS_DST_ALPHA",SRC_ALPHA_SATURATE:"SRC_ALPHA_SATURATE",CONSTANT_COLOR:"CONSTANT_COLOR",ONE_MINUS_CONSTANT_COLOR:"ONE_MINUS_CONSTANT_COLOR",CONSTANT_ALPHA:"CONSTANT_ALPHA",ONE_MINUS_CONSTANT_ALPHA:"ONE_MINUS_CONSTANT_ALPHA"},BLENDING_EQUATION_TYPES:{FUNC_ADD:"FUNC_ADD",FUNC_SUBTRACT:"FUNC_SUBTRACT",FUNC_REVERSE_SUBTRACT:"FUNC_REVERSE_SUBTRACT"},DRAW_MODES:{POINTS:0,LINES:1,LINE_LOOP:2,LINE_STRIP:3,TRIANGLES:4,TRIANGLE_STRIP:5,TRIANGLE_FAN:6}};e.Constants=r,t.exports.Constants=r}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],12:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.Mesh=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./Utils.class"),a=n.Mesh=function(){function e(){o(this,e),this._obj={},this._textures={},this.objIndex=null,this.indexMax=0}return i(e,[{key:"loadPoint",value:function(){return this._obj.vertexArray=[0,0,0,0],this._obj.normalArray=[0,1,0,0],this._obj.textureArray=[0,0,0,0],this._obj.textureUnitArray=[0],this._obj.indexArray=[0],this._obj}},{key:"loadTriangle",value:function(e){var t=void 0!==e&&void 0!==e.scale?e.scale:1,n=void 0!==e&&void 0!==e.side?e.side:1;return this._obj.vertexArray=[0,0,0,1,n/2*t,0,-1*t,1,-n/2*t,0,-1*t,1],this._obj.normalArray=[0,0,1,1,0,0,1,1,0,0,1,1],this._obj.textureArray=[0,0,0,1,1,0,0,1,1,1,0,1],this._obj.textureUnitArray=[0,0,0],this._obj.indexArray=[0,1,2],this._obj}},{key:"loadQuad",value:function(e,t){var n=void 0===e?.5:e,r=void 0===t?.5:t;return this._obj={},this._obj.vertexArray=[-n,-r,0,1,n,-r,0,1,n,r,0,1,-n,r,0,1],this._obj.normalArray=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],this._obj.textureArray=[0,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1],this._obj.textureUnitArray=[0,0,0,0],this._obj.indexArray=[0,1,2,0,2,3],this._obj}},{key:"loadCircle",value:function(e){this._obj={vertexArray:[],normalArray:[],textureArray:[],textureUnitArray:[],indexArray:[]},this.objIndex=[],this.indexMax=0;for(var t=void 0!==e&&void 0!==e.segments?e.segments:6,n=void 0!==e&&void 0!==e.radius?e.radius:.5,r=360/t,o=360/r,i=function(e){return Math.cos((new s.Utils).degToRad(e))}.bind(this),a=function(e){return Math.sin((new s.Utils).degToRad(e))}.bind(this),l=1,u=o;l<=u;l++){var c=r*l,h=$V3([i(c)*n,0,a(c)*n]),f=$V3([i(c-r)*n,0,a(c-r)*n]),d=$V3([0,0,0]),g=f.subtract(h).cross(d.subtract(h)).normalize(),_=$V3([c/360,0,0]),v=$V3([(c-r)/360,0,0]),p=$V3([0,0,0]),m=this.testIfInIndices(this._obj,h,g,_),y=this.testIfInIndices(this._obj,f,g,v),b=this.testIfInIndices(this._obj,d,g,p);this._obj.indexArray.push(m,y,b)}return this._obj}},{key:"testIfInIndices",value:function(e,t,n,r){for(var o=void 0,i=0,s=this.objIndex.length;i<s;i++)this.objIndex[i].v.e[0]===t.e[0]&&this.objIndex[i].v.e[1]===t.e[1]&&this.objIndex[i].v.e[2]===t.e[2]&&(o=this.objIndex[i].i);return void 0===o&&(o=this.indexMax,this.objIndex.push({i:o,v:$V3([t.e[0],t.e[1],t.e[2]])}),this.indexMax++,e.vertexArray.push(t.e[0],t.e[1],t.e[2],1),e.normalArray.push(n.e[0],n.e[1],n.e[2],1),e.textureArray.push(t.e[0]+.5,t.e[2]+.5,t.e[2]+.5,1),e.textureUnitArray.push(0)),o}},{key:"loadBox",value:function(){this._obj={};var e=new Float32Array([.5,.5,.5]);return this._obj.vertexArray=[-1*e[0],-1*e[1],1*e[2],1,1*e[0],-1*e[1],1*e[2],1,1*e[0],1*e[1],1*e[2],1,-1*e[0],1*e[1],1*e[2],1,-1*e[0],-1*e[1],-1*e[2],1,-1*e[0],1*e[1],-1*e[2],1,1*e[0],1*e[1],-1*e[2],1,1*e[0],-1*e[1],-1*e[2],1,-1*e[0],1*e[1],-1*e[2],1,-1*e[0],1*e[1],1*e[2],1,1*e[0],1*e[1],1*e[2],1,1*e[0],1*e[1],-1*e[2],1,-1*e[0],-1*e[1],-1*e[2],1,1*e[0],-1*e[1],-1*e[2],1,1*e[0],-1*e[1],1*e[2],1,-1*e[0],-1*e[1],1*e[2],1,1*e[0],-1*e[1],-1*e[2],1,1*e[0],1*e[1],-1*e[2],1,1*e[0],1*e[1],1*e[2],1,1*e[0],-1*e[1],1*e[2],1,-1*e[0],-1*e[1],-1*e[2],1,-1*e[0],-1*e[1],1*e[2],1,-1*e[0],1*e[1],1*e[2],1,-1*e[0],1*e[1],-1*e[2],1],this._obj.normalArray=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,-1,1,0,0,-1,1,0,0,-1,1,0,0,-1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1,0,-1,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,-1,0,0,1,-1,0,0,1,-1,0,0,1,-1,0,0,1],this._obj.textureArray=[0,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,0,1,0,1],this._obj.textureUnitArray=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],this._obj.indexArray=[0,1,2,0,2,3,4,5,6,4,6,7,8,9,10,8,10,11,12,13,14,12,14,15,16,17,18,16,18,19,20,21,22,20,22,23],this._obj}},{key:"loadObj",value:function(e,t){for(var n="",r=e.split("/"),o=0,i=r.length-1;o<i;o++)n=n+r[o]+"/";var s=new XMLHttpRequest;s.open("GET",e,!0),s.responseType="blob",s.onload=function(e){var t=new FileReader;t.onload=function(e,t){var r=t.target.result;this.loadObjFromSourceText({sourceText:r,objDirectory:n}),void 0!==e&&"function"==typeof e&&e(this._obj,this._textures)}.bind(this,e),t.readAsText(s.response)}.bind(this,t),s.send(null)}},{key:"loadObjFromSourceText",value:function(e){this._obj={},this._textures={};var t=void 0,n=void 0;t=void 0!==e.sourceText?e.sourceText:void 0,n=void 0!==e.objDirectory?e.objDirectory:void 0;var r=t.split("\r\n");if(1===r.length&&(r=t.split("\n")),null!=r[0].match(/OBJ/gim)){var o=[],i=[],s=[],a=[],l=[],u=[],c=[],h=[],f=[],d=[],g=0,_=[],v=0,p=0,m=!1,y=[],b=[],E=[],C=[],T=[],x=[],P=[],L=[],A=[],w=0,R=0,S=0,k=[],M=0,O={},F=[-1,0];O._unnamed=F;for(var N="",G="",U=0,B=r.length;U<B;U++){var D=r[U].replace(/\t+/gi," ").replace(/\s+$/gi,"").replace(/\s+/gi," ");if("#"!==D[0]){var W=D.split(" ");if("mtllib"===W[0]&&(N=W[1]),"g"===W[0]&&(F=[_.length,0],O[W[1]]=F),"v"===W[0]&&(y[w]=parseFloat(W[1]),b[w]=parseFloat(W[2]),E[w]=parseFloat(W[3]),w++),"vn"===W[0]&&(C[R]=parseFloat(W[1]),T[R]=parseFloat(W[2]),x[R]=parseFloat(W[3]),R++),"vt"===W[0]&&(P[S]=parseFloat(W[1]),L[S]=parseFloat(W[2]),A[S]=parseFloat(W[3]),S++),"usemtl"===W[0]&&(G=W[1],this._textures[g]={fileUrl:n+N,materialName:G},g++),"f"===W[0]){4!==W.length&&console.log("*** Error: face '"+D+"' not handled");for(var V=1;V<4;++V){m=!0;var j=W[V].split("/");if(void 0===k[W[V]]){var I=void 0,Y=void 0,X=void 0;if(1===j.length)Y=I=parseInt(j[0])-1,X=I;else{if(3!==j.length)return null;I=parseInt(j[0])-1,X=parseInt(j[1])-1,Y=parseInt(j[2])-1}d[v]=g-1,o[v]=0,i[v]=0,s[v]=0,I<E.length&&(o[v]=y[I],i[v]=b[I],s[v]=E[I]),c[v]=0,h[v]=0,f[v]=0,X<A.length&&(c[v]=P[X],h[v]=L[X],f[v]=A[X]),a[v]=0,l[v]=0,u[v]=1,Y<x.length&&(a[v]=C[Y],l[v]=T[Y],u[v]=x[Y]),v++,k[W[V]]=M,M++}_[p]=k[W[V]],p++,F[1]++}}}}if(!0===m){this._obj.vertexArray=new Float32Array(4*o.length);for(var H=0,z=o.length;H<z;H++){var K=4*H;this._obj.vertexArray[K]=o[H],this._obj.vertexArray[K+1]=i[H],this._obj.vertexArray[K+2]=s[H],this._obj.vertexArray[K+3]=1}this._obj.normalArray=new Float32Array(4*a.length);for(var q=0,$=a.length;q<$;q++){var Z=4*q;this._obj.normalArray[Z]=a[q],this._obj.normalArray[Z+1]=l[q],this._obj.normalArray[Z+2]=u[q],this._obj.normalArray[Z+3]=1}this._obj.textureArray=new Float32Array(4*c.length);for(var J=0,Q=c.length;J<Q;J++){var ee=4*J;this._obj.textureArray[ee]=c[J],this._obj.textureArray[ee+1]=h[J],this._obj.textureArray[ee+2]=f[J],this._obj.textureArray[ee+3]=1}this._obj.textureUnitArray=d,this._obj.indexArray=_}}else alert("Not OBJ file")}}]),e}();r.Mesh=a,t.exports.Mesh=a}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Utils.class":22}],13:[function(e,t,n){(function(e){"use strict";function r(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0});var o=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),i=n.Node=function(){function e(){r(this,e),this._components={},this._name=null,this._gl=null,this._enabled=!0,this.onTick=null}return o(e,[{key:"initialize",value:function(e,t){this._name=e,this._gl=t}},{key:"addComponent",value:function(e){this._components[e.type]=e,this._components[e.type].initialize(this,this._gl)}},{key:"getComponent",value:function(e){return this._components[e]}},{key:"getComponents",value:function(){return this._components}},{key:"setEnabled",value:function(e){this._enabled=e}},{key:"isEnabled",value:function(){return this._enabled}},{key:"setName",value:function(e){this._name=e}},{key:"getName",value:function(){return this._name}}]),e}();e.Node=i,t.exports.Node=i}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],14:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.Grid=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("../../Utils.class"),a=e("../../ComponentTransform.class"),l=e("../../Component_GPU.class"),u=e("../../Mesh.class"),c=e("../../Constants"),h=n.Grid=function(){function e(t){o(this,e),this._sce=t,this._project=this._sce.getLoadedProject(),this._gl=this._project.getActiveStage().getWebGLContext(),this._utils=new s.Utils,this.node=new Node,this.node.setName("grid"),this._project.getActiveStage().addNode(this.node),this.mesh=(new u.Mesh).loadBox(),this.comp_transform=new a.ComponentTransform,this.node.addComponent(this.comp_transform),this.comp_renderer=new l.Component_GPU,this.node.addComponent(this.comp_renderer),this.gridColor=$V3([.3,.3,.3])}return i(e,[{key:"generate",value:function(e,t){this.gridEnabled=!0,this.size=void 0!==e&&null!==e?e:this.size,this.separation=void 0!==t&&null!==t?t:this.separation,this.countLines=this.size/this.separation+1,this.countLines*=2;var n=-this.size/2,r=this.size/2,o=-this.size/2,i=this.size/2,s=n,a=o,l=[],u=[],h=[];this.id=0;for(var f=0,d=this.countLines;f<d;f++)a<=i?(l.push(n,0,a,1,r,0,a,1),a+=this.separation):(l.push(s,0,o,1,s,0,i,1),s+=this.separation),u.push(this.gridColor.e[0],this.gridColor.e[1],this.gridColor.e[2],1,this.gridColor.e[0],this.gridColor.e[1],this.gridColor.e[2],1),h.push(this.id,this.id+1),this.id+=2;l.push(0,0,0,1,10,0,0,1),u.push(1,0,0,1,1,0,0,1),h.push(this.id,this.id+1),this.id+=2,l.push(0,0,0,1,0,10,0,1),u.push(0,1,0,1,0,1,0,1),h.push(this.id,this.id+1),this.id+=2,l.push(0,0,0,1,0,0,10,1),u.push(0,0,1,1,0,0,1,1),h.push(this.id,this.id+1),this.comp_renderer.setGPUFor(this.comp_renderer.gl,{"float4*attr vertexPos":function(){return l}.bind(this),"float4*attr vertexColor":function(){return u}.bind(this),indices:function(){return h}.bind(this),"mat4 PMatrix":function(){return this._project.getActiveStage().getActiveCamera().getComponent(c.Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e}.bind(this),"mat4 cameraWMatrix":function(){return this._project.getActiveStage().getActiveCamera().getComponent(c.Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e}.bind(this),"mat4 nodeWMatrix":function(){return this.node.getComponent(c.Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e}.bind(this)},{type:"GRAPHIC",config:[["RGB"],"varying vec4 vVC;\n","vec4 vp = vertexPos[];\nvec4 vc = vertexColor[];\nvVC = vc;gl_Position = PMatrix * cameraWMatrix * nodeWMatrix * vp;\n","varying vec4 vVC;\n","return [vVC];\n"],drawMode:1,depthTest:!0,blend:!0,blendEquation:c.Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,blendSrcMode:c.Constants.BLENDING_MODES.SRC_ALPHA,blendDstMode:c.Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA}),this.comp_renderer.setArgUpdatable("PMatrix",!0),this.comp_renderer.setArgUpdatable("cameraWMatrix",!0),this.comp_renderer.setArgUpdatable("nodeWMatrix",!0),this.comp_renderer.getComponentBufferArg("RGB",this._project.getActiveStage().getActiveCamera().getComponent(c.Constants.COMPONENT_TYPES.GPU))}}]),e}();r.Grid=h,t.exports.Grid=h}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"../../ComponentTransform.class":8,"../../Component_GPU.class":10,"../../Constants":11,"../../Mesh.class":12,"../../Utils.class":22}],15:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.SimpleCamera=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("../../ComponentTransformTarget.class"),a=e("../../ComponentProjection.class"),l=e("../../ComponentControllerTransformTarget.class"),u=e("../../Component_GPU.class"),c=e("../../ComponentKeyboardEvents.class"),h=e("../../ComponentMouseEvents.class"),f=e("../../Node.class"),d=e("../../Constants"),g=function(){function e(t,n){var r=this;o(this,e);var i=t,g=i.getLoadedProject();this._onkeydown=void 0!==n&&void 0!==n.onkeydown&&null!==n.onkeydown?n.onkeydown:null,this._onkeyup=void 0!==n&&void 0!==n.onkeyup&&null!==n.onkeyup?n.onkeyup:null,this._onmousedown=void 0!==n&&void 0!==n.onmousedown&&null!==n.onmousedown?n.onmousedown:null,this._onmouseup=void 0!==n&&void 0!==n.onmouseup&&null!==n.onmouseup?n.onmouseup:null,this._onmousemove=void 0!==n&&void 0!==n.onmousemove&&null!==n.onmousemove?n.onmousemove:null,this._onmousewheel=void 0!==n&&void 0!==n.onmousewheel&&null!==n.onmousewheel?n.onmousewheel:null,this._useAltKey=!0,this.altKeyPressed=!1,this._preventMove=!1,this.camera=new f.Node,this.camera.setName("simple camera"),g.getActiveStage().addNode(this.camera),g.getActiveStage().setActiveCamera(this.camera),this.comp_transformTarget=new s.ComponentTransformTarget,this.camera.addComponent(this.comp_transformTarget),this.comp_transformTarget.setTargetDistance(.5),this.comp_projection=new a.ComponentProjection,this.camera.addComponent(this.comp_projection),this.comp_controllerTransformTarget=new l.ComponentControllerTransformTarget,this.camera.addComponent(this.comp_controllerTransformTarget),this.comp_screenEffects=new u.Component_GPU,this.camera.addComponent(this.comp_screenEffects),this.comp_screenEffects.setGPUFor(this.comp_screenEffects.gl,{"float4* RGB":function(){return new Float32Array(i.getCanvas().width*i.getCanvas().width*4)}},{type:"KERNEL",name:"CAMERA_RGB_KERNEL",viewSource:!1,config:["n",void 0,"","vec4 color = RGB[n];\nreturn color;\n"],drawMode:4,depthTest:!1,blend:!1,blendEquation:d.Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,blendSrcMode:d.Constants.BLENDING_MODES.ONE,blendDstMode:d.Constants.BLENDING_MODES.ZERO}),this.comp_screenEffects.gpufG.onPostProcessKernel(0,function(){}),this.comp_keyboardEvents=new c.ComponentKeyboardEvents,this.camera.addComponent(this.comp_keyboardEvents),this.comp_keyboardEvents.onkeydown(function(e){var t=String.fromCharCode(e.keyCode);"W"===t&&r.comp_controllerTransformTarget.forward(),"S"===t&&r.comp_controllerTransformTarget.backward(),"A"!==t&&"%"!==t||r.comp_controllerTransformTarget.strafeLeft(),"D"!==t&&"'"!==t||r.comp_controllerTransformTarget.strafeRight(),"E"!==t&&"&"!==t||r.comp_controllerTransformTarget.strafeFront(),"C"!==t&&"("!==t||r.comp_controllerTransformTarget.strafeBack(),"L"===t&&r.setView(d.Constants.VIEW_TYPES.LEFT),"R"===t&&r.setView(d.Constants.VIEW_TYPES.RIGHT),"F"===t&&r.setView(d.Constants.VIEW_TYPES.FRONT),"B"===t&&r.setView(d.Constants.VIEW_TYPES.BACK),"T"===t&&r.setView(d.Constants.VIEW_TYPES.TOP),"P"===t&&r.comp_projection.setProjection(d.Constants.PROJECTION_TYPES.PERSPECTIVE),"O"===t&&r.comp_projection.setProjection(d.Constants.PROJECTION_TYPES.ORTHO),!0===e.altKey&&(r.altKeyPressed=!0),null!=r._onkeydown&&r._onkeydown()}),this.comp_keyboardEvents.onkeyup(function(e){var t=String.fromCharCode(e.keyCode);"W"===t&&r.comp_controllerTransformTarget.stopForward(),"S"===t&&r.comp_controllerTransformTarget.stopBackward(),"A"!==t&&"%"!==t||r.comp_controllerTransformTarget.stopStrafeLeft(),"D"!==t&&"'"!==t||r.comp_controllerTransformTarget.stopStrafeRight(),"E"!==t&&"&"!==t||r.comp_controllerTransformTarget.stopStrafeFront(),"C"!==t&&"("!==t||r.comp_controllerTransformTarget.stopStrafeBack(),!1===e.altKey&&(r.altKeyPressed=!1),null!=r._onkeyup&&r._onkeyup()});var _=new h.ComponentMouseEvents;this.camera.addComponent(_),_.onmousedown(function(e){r.comp_controllerTransformTarget.mouseDown(e),!0===e.altKey&&(r.altKeyPressed=!0),null!=r._onmousedown&&r._onmousedown()}),_.onmouseup(function(e){r.comp_controllerTransformTarget.mouseUp(e),!1===e.altKey&&(r.altKeyPressed=!1),null!=r._onmouseup&&r._onmouseup()}),_.onmousemove(function(e,t){null!=r._onmousemove&&r._onmousemove(),!1===r._preventMove&&((r.comp_projection.getProjection()===d.Constants.PROJECTION_TYPES.PERSPECTIVE||!0===r._useAltKey&&!0===r.altKeyPressed||!1===r._useAltKey)&&r.comp_controllerTransformTarget.mouseMove(e),!0===r.comp_controllerTransformTarget.isRightBtnActive()&&r.comp_projection.getProjection()===d.Constants.PROJECTION_TYPES.ORTHO&&!0===r.altKeyPressed&&(t.e[2]>0?r.comp_projection.setFov(r.comp_projection.getFov()*(1+Math.abs(.005*t.e[2]))):r.comp_projection.setFov(r.comp_projection.getFov()/(1+Math.abs(.005*t.e[2])))))}),_.onmousewheel(function(e,t){void 0!==e.wheelDeltaY&&e.wheelDeltaY>=0||void 0!==e.deltaY&&e.deltaY<=0?r.comp_projection.setFov(r.comp_projection.getFov()/1.1):r.comp_projection.setFov(1.1*r.comp_projection.getFov()),r.comp_projection.getProjection()===d.Constants.PROJECTION_TYPES.ORTHO&&(r.comp_transformTarget.setPositionTarget(r.comp_transformTarget.getPositionTarget().add(t)),r.comp_transformTarget.setPositionGoal(r.comp_transformTarget.getPositionGoal().add(t))),null!=r._onmousewheel&&r._onmousewheel()})}return i(e,[{key:"setView",value:function(e){switch(this.comp_projection.setProjection(d.Constants.PROJECTION_TYPES.ORTHO),this.comp_transformTarget.reset(),e){case d.Constants.VIEW_TYPES.LEFT:this.comp_transformTarget.yaw(90);break;case d.Constants.VIEW_TYPES.RIGHT:this.comp_transformTarget.yaw(-90);break;case d.Constants.VIEW_TYPES.FRONT:case d.Constants.VIEW_TYPES.BACK:break;case d.Constants.VIEW_TYPES.TOP:this.comp_transformTarget.pitch(-89.9);break;case d.Constants.VIEW_TYPES.BOTTOM:this.comp_transformTarget.pitch(90)}}},{key:"setLeftButtonAction",value:function(e){this.comp_controllerTransformTarget.setLeftButtonAction(e)}},{key:"setMiddleButtonAction",value:function(e){this.comp_controllerTransformTarget.setMiddleButtonAction(e)}},{key:"setRightButtonAction",value:function(e){this.comp_controllerTransformTarget.setRightButtonAction(e)}},{key:"isAltKeyEnabled",value:function(){return this._useAltKey}},{key:"isAltKeyPressed",value:function(){return this.altKeyPressed}},{key:"setVelocity",value:function(e){this.comp_controllerTransformTarget.setVelocity(e)}},{key:"setFov",value:function(e){this.comp_projection.setFov(e)}},{key:"getFov",value:function(){return this.comp_projection.getFov()}}],[{key:"preventMove",value:function(e){this._preventMove=e}},{key:"useAltKey",value:function(e){this._useAltKey=e}}]),e}();n.SimpleCamera=g,r.SimpleCamera=g,t.exports.SimpleCamera=g}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"../../ComponentControllerTransformTarget.class":4,"../../ComponentKeyboardEvents.class":5,"../../ComponentMouseEvents.class":6,"../../ComponentProjection.class":7,"../../ComponentTransformTarget.class":9,"../../Component_GPU.class":10,"../../Constants":11,"../../Node.class":13}],16:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.SimpleNode=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("../../ComponentTransform.class"),a=e("../../Component_GPU.class"),l=e("../../VFP_RGB.class"),u=e("../../Constants"),c=n.SimpleNode=function(){function e(t){o(this,e),this._sce=t,this._project=this._sce.getLoadedProject(),this.node=new Node,this._project.getActiveStage().addNode(this.node),this._mesh=null;var n=new s.ComponentTransform;this.node.addComponent(n),this.comp_renderer=new a.Component_GPU,this.node.addComponent(this.comp_renderer)}return i(e,[{key:"setMesh",value:function(e){var t=this;this._mesh=e,this.comp_renderer.setGPUFor(this.comp_renderer.gl,{"float4*attr vertexPos":function(){return t._mesh.vertexArray},"float4*attr vertexNormal":function(){return t._mesh.normalArray},"float4*attr vertexTexture":function(){return t._mesh.textureArray},"float*attr vertexTextureUnit":function(){return t._mesh.textureUnitArray},indices:function(){return t._mesh.indexArray},"mat4 PMatrix":function(){return t._project.getActiveStage().getActiveCamera().getComponent(u.Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e},"mat4 cameraWMatrix":function(){return t._project.getActiveStage().getActiveCamera().getComponent(u.Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e},"mat4 nodeWMatrix":function(){return t.node.getComponent(u.Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e},"float nodesSize":function(){return 30},"float4* texAlbedo":function(){return t._mesh.vertexArray}},{type:"GRAPHIC",config:new l.VFP_RGB(1).getSrc(),drawMode:4,depthTest:!0,blend:!0,blendEquation:u.Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,blendSrcMode:u.Constants.BLENDING_MODES.SRC_ALPHA,blendDstMode:u.Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA}),this.comp_renderer.setArgUpdatable("PMatrix",!0),this.comp_renderer.setArgUpdatable("cameraWMatrix",!0),this.comp_renderer.setArgUpdatable("nodeWMatrix",!0),this.comp_renderer.getComponentBufferArg("RGB",this._project.getActiveStage().getActiveCamera().getComponent(u.Constants.COMPONENT_TYPES.GPU)),this.comp_renderer.gpufG.onPreProcessGraphic(0,function(){})}},{key:"setImage",value:function(e){var t=this,n=new Image;n.onload=function(){t.comp_renderer.setArg("texAlbedo",function(){return n})},n.src=e}}]),e}();r.SimpleNode=c,t.exports.SimpleNode=c}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"../../ComponentTransform.class":8,"../../Component_GPU.class":10,"../../Constants":11,"../../VFP_RGB.class":23}],17:[function(e,t,n){(function(e){"use strict";function r(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0});var o=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),i=n.Project=function(){function e(){r(this,e),this.stages=[],this.activeStage=null,this.gl=null}return o(e,[{key:"setActiveStage",value:function(e){this.activeStage=e,this.activeStage.setWebGLContext(this.gl)}},{key:"getActiveStage",value:function(){return this.activeStage}},{key:"addStage",value:function(e){this.stages.push(e)}},{key:"setWebGLContext",value:function(e){this.gl=e}}]),e}();e.Project=i,t.exports.Project=i}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],18:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.SCE=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./SystemEvents.class"),a=e("./Utils.class"),l=e("./Constants"),u=n.SCE=function(){function e(){o(this,e),this.target=null,this.project=null,this.dimensions=null,this.canvas=null,this.gl=null,this._UI=null,this._enableUI=null,this._events=null}return i(e,[{key:"initialize",value:function(e){if(this.target=void 0!==e.target?e.target:void 0,this.dimensions=void 0!==e.dimensions?e.dimensions:{width:512,height:512},this._enableUI=void 0!==e.enableUI&&e.enableUI,null!=this.target){if(this.canvas=document.createElement("canvas"),this.target.appendChild(this.canvas),this.setDimensions(this.dimensions.width,this.dimensions.height),void 0!==e.gl&&null!==e.gl)this.gl=e.gl;else if(!(this.gl=a.Utils.getWebGLContextFromCanvas(this.canvas)))return alert("No WebGLRenderingContext"),!1}else alert("Target DIV required")}},{key:"loadProject",value:function(e){this.project=e,this.project.setWebGLContext(this.gl),!0===this._enableUI&&(this._UI=new UI(this.project).render(this.target,this.canvas)),this._events=new s.SystemEvents(this,this.canvas),this._events.initialize()}},{key:"getLoadedProject",value:function(){return this.project}},{key:"getCanvas",value:function(){return this.canvas}},{key:"getEvents",value:function(){return this._events}},{key:"setDimensions",value:function(e,t){if(this.dimensions={width:e,height:t},this.canvas.setAttribute("width",this.dimensions.width),this.canvas.setAttribute("height",this.dimensions.height),null!=this.project&&null!=this.project.getActiveStage()){var n=this.project.getActiveStage().getActiveCamera();if(null!=n){var r=n.getComponent(l.Constants.COMPONENT_TYPES.PROJECTION),o=n.getComponent(l.Constants.COMPONENT_TYPES.GPU);r.setResolution(this.dimensions.width,this.dimensions.width),o.setArg("RGB",function(){return new Float32Array(this.dimensions.width*this.dimensions.width*4)}.bind(this))}}}},{key:"getDimensions",value:function(){return this.dimensions}}]),e}();r.SCE=u,t.exports.SCE=u}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Constants":11,"./SystemEvents.class":21,"./Utils.class":22}],19:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.Stage=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./Constants"),a=n.Stage=function(){function e(){o(this,e),this.nodes=[],this.activeCamera=null,this.selectedNode=null,this.paused=!1,this.backgroundColor=[0,0,0,1],this.gl=null,this._ontick=null}return i(e,[{key:"setActiveCamera",value:function(e){this.activeCamera=e}},{key:"getActiveCamera",value:function(){return this.activeCamera}},{key:"setSelectedNode",value:function(e){this.selectedNode=e}},{key:"getSelectedNode",value:function(){return this.selectedNode}},{key:"addNode",value:function(e){this.nodes.push(e),e.initialize(null!=e.getName()?e.getName():"node "+(this.nodes.length-1),this.gl)}},{key:"removeNode",value:function(e){for(var t=0;t<this.nodes.length;t++)if(this.nodes[t]===e){this.nodes.splice(t,1);break}}},{key:"getNodes",value:function(){return this.nodes}},{key:"render",value:function(e){this._ontick=e,this.paused=!1,this.setBackgroundColor(this.backgroundColor),this.tick()}},{key:"pause",value:function(){this.paused=!0}},{key:"setWebGLContext",value:function(e){this.gl=e}},{key:"setBackgroundColor",value:function(e){this.backgroundColor=e}},{key:"getBackgroundColor",value:function(){return this.backgroundColor}},{key:"getWebGLContext",value:function(){return this.gl}},{key:"tick",value:function(){if(null!=this.activeCamera){void 0!==this._ontick&&null!==this._ontick&&this._ontick();var e=this.activeCamera.getComponent(s.Constants.COMPONENT_TYPES.PROJECTION).getResolution();this.gl.viewport(0,0,e.width,e.height),this.gl.clearColor(this.backgroundColor[0],this.backgroundColor[1],this.backgroundColor[2],this.backgroundColor[3]),this.gl.clearDepth(1),this.gl.depthFunc(this.gl.LEQUAL);var t=this.activeCamera.getComponent(s.Constants.COMPONENT_TYPES.GPU);void 0!==t&&null!==t&&t.gpufG.fillArg("RGB",[this.backgroundColor[0],this.backgroundColor[1],this.backgroundColor[2],this.backgroundColor[3]]);for(var n=0,r=this.nodes.length;n<r;n++)if(this.nodes[n]!==this.activeCamera){for(var o in this.nodes[n].getComponents()){var i=this.nodes[n].getComponent(o);null!=i.tick&&i.tick()}null!=this.nodes[n].onTick&&this.nodes[n].onTick()}for(var a=0,l=this.nodes.length;a<l;a++)if(this.nodes[a]!==this.activeCamera)for(var u in this.nodes[a].getComponents()){var c=this.nodes[a].getComponent(u);if(null!=c.gpufG)for(var h in c.gpufG.vertexFragmentPrograms){var f=c.gpufG.vertexFragmentPrograms[h],d=WebCLGLUtils.getOutputBuffers(f,c.gpufG._argsValues);!0===f.enabled&&null!=d&&(c.gpufG._gl.bindFramebuffer(c.gpufG._gl.FRAMEBUFFER,d[0].fBuffer),c.gpufG._gl.clear(c.gpufG._gl.DEPTH_BUFFER_BIT),c.gpufG._gl.bindFramebuffer(c.gpufG._gl.FRAMEBUFFER,d[0].fBufferTemp),c.gpufG._gl.clear(c.gpufG._gl.DEPTH_BUFFER_BIT))}}for(var g in this.activeCamera.getComponents()){var _=this.activeCamera.getComponent(g);null!=_.tick&&_.tick()}}}}]),e}();r.Stage=a,t.exports.Stage=a}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Constants":11}],20:[function(e,t,n){(function(e){"use strict";function r(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function o(e){return new a(e)}function i(e){return new l(e)}Object.defineProperty(n,"__esModule",{value:!0});var s=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}();n.$M16=o,n.$V3=i;var a=n.StormM16=function(){function e(t){r(this,e),this.e=null,this.e=void 0!==t&&null!==t?new Float32Array(t):new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])}return s(e,[{key:"x",value:function(t){return o(t instanceof e?[this.e[0]*t.e[0]+this.e[1]*t.e[4]+this.e[2]*t.e[8]+this.e[3]*t.e[12],this.e[0]*t.e[1]+this.e[1]*t.e[5]+this.e[2]*t.e[9]+this.e[3]*t.e[13],this.e[0]*t.e[2]+this.e[1]*t.e[6]+this.e[2]*t.e[10]+this.e[3]*t.e[14],this.e[0]*t.e[3]+this.e[1]*t.e[7]+this.e[2]*t.e[11]+this.e[3]*t.e[15],this.e[4]*t.e[0]+this.e[5]*t.e[4]+this.e[6]*t.e[8]+this.e[7]*t.e[12],this.e[4]*t.e[1]+this.e[5]*t.e[5]+this.e[6]*t.e[9]+this.e[7]*t.e[13],this.e[4]*t.e[2]+this.e[5]*t.e[6]+this.e[6]*t.e[10]+this.e[7]*t.e[14],this.e[4]*t.e[3]+this.e[5]*t.e[7]+this.e[6]*t.e[11]+this.e[7]*t.e[15],this.e[8]*t.e[0]+this.e[9]*t.e[4]+this.e[10]*t.e[8]+this.e[11]*t.e[12],this.e[8]*t.e[1]+this.e[9]*t.e[5]+this.e[10]*t.e[9]+this.e[11]*t.e[13],this.e[8]*t.e[2]+this.e[9]*t.e[6]+this.e[10]*t.e[10]+this.e[11]*t.e[14],this.e[8]*t.e[3]+this.e[9]*t.e[7]+this.e[10]*t.e[11]+this.e[11]*t.e[15],this.e[12]*t.e[0]+this.e[13]*t.e[4]+this.e[14]*t.e[8]+this.e[15]*t.e[12],this.e[12]*t.e[1]+this.e[13]*t.e[5]+this.e[14]*t.e[9]+this.e[15]*t.e[13],this.e[12]*t.e[2]+this.e[13]*t.e[6]+this.e[14]*t.e[10]+this.e[15]*t.e[14],this.e[12]*t.e[3]+this.e[13]*t.e[7]+this.e[14]*t.e[11]+this.e[15]*t.e[15]]:[0,0,0,this.e[0]*t.e[0]+this.e[1]*t.e[1]+this.e[2]*t.e[2],0,0,0,this.e[4]*t.e[0]+this.e[5]*t.e[1]+this.e[6]*t.e[2],0,0,0,this.e[8]*t.e[0]+this.e[9]*t.e[1]+this.e[10]*t.e[2],0,0,0,this.e[12]*t.e[0]+this.e[13]*t.e[1]+this.e[14]*t.e[2]])}},{key:"transpose",value:function(){return o([this.e[0],this.e[4],this.e[8],this.e[12],this.e[1],this.e[5],this.e[9],this.e[13],this.e[2],this.e[6],this.e[10],this.e[14],this.e[3],this.e[7],this.e[11],this.e[15]])}},{key:"inverse",value:function(){var e=o(this.e).transpose().e,t=new Float32Array(12),n=new Float32Array(16);t[0]=e[10]*e[15],t[1]=e[11]*e[14],t[2]=e[9]*e[15],t[3]=e[11]*e[13],t[4]=e[9]*e[14],t[5]=e[10]*e[13],t[6]=e[8]*e[15],t[7]=e[11]*e[12],t[8]=e[8]*e[14],t[9]=e[10]*e[12],t[10]=e[8]*e[13],t[11]=e[9]*e[12],n[0]=t[0]*e[5]+t[3]*e[6]+t[4]*e[7],n[0]-=t[1]*e[5]+t[2]*e[6]+t[5]*e[7],n[1]=t[1]*e[4]+t[6]*e[6]+t[9]*e[7],n[1]-=t[0]*e[4]+t[7]*e[6]+t[8]*e[7],n[2]=t[2]*e[4]+t[7]*e[5]+t[10]*e[7],n[2]-=t[3]*e[4]+t[6]*e[5]+t[11]*e[7],n[3]=t[5]*e[4]+t[8]*e[5]+t[11]*e[6],n[3]-=t[4]*e[4]+t[9]*e[5]+t[10]*e[6],n[4]=t[1]*e[1]+t[2]*e[2]+t[5]*e[3],n[4]-=t[0]*e[1]+t[3]*e[2]+t[4]*e[3],n[5]=t[0]*e[0]+t[7]*e[2]+t[8]*e[3],n[5]-=t[1]*e[0]+t[6]*e[2]+t[9]*e[3],n[6]=t[3]*e[0]+t[6]*e[1]+t[11]*e[3],n[6]-=t[2]*e[0]+t[7]*e[1]+t[10]*e[3],n[7]=t[4]*e[0]+t[9]*e[1]+t[10]*e[2],n[7]-=t[5]*e[0]+t[8]*e[1]+t[11]*e[2],t[0]=e[2]*e[7],t[1]=e[3]*e[6],t[2]=e[1]*e[7],t[3]=e[3]*e[5],t[4]=e[1]*e[6],t[5]=e[2]*e[5],t[6]=e[0]*e[7],t[7]=e[3]*e[4],t[8]=e[0]*e[6],t[9]=e[2]*e[4],t[10]=e[0]*e[5],t[11]=e[1]*e[4],n[8]=t[0]*e[13]+t[3]*e[14]+t[4]*e[15],n[8]-=t[1]*e[13]+t[2]*e[14]+t[5]*e[15],n[9]=t[1]*e[12]+t[6]*e[14]+t[9]*e[15],n[9]-=t[0]*e[12]+t[7]*e[14]+t[8]*e[15],n[10]=t[2]*e[12]+t[7]*e[13]+t[10]*e[15],n[10]-=t[3]*e[12]+t[6]*e[13]+t[11]*e[15],n[11]=t[5]*e[12]+t[8]*e[13]+t[11]*e[14],n[11]-=t[4]*e[12]+t[9]*e[13]+t[10]*e[14],n[12]=t[2]*e[10]+t[5]*e[11]+t[1]*e[9],n[12]-=t[4]*e[11]+t[0]*e[9]+t[3]*e[10],n[13]=t[8]*e[11]+t[0]*e[8]+t[7]*e[10],n[13]-=t[6]*e[10]+t[9]*e[11]+t[1]*e[8],n[14]=t[6]*e[9]+t[11]*e[11]+t[3]*e[8],n[14]-=t[10]*e[11]+t[2]*e[8]+t[7]*e[9],n[15]=t[10]*e[10]+t[4]*e[8]+t[9]*e[9],n[15]-=t[8]*e[9]+t[11]*e[10]+t[5]*e[8];var r=e[0]*n[0]+e[1]*n[1]+e[2]*n[2]+e[3]*n[3];return o([n[0]*r,n[1]*r,n[2]*r,n[3]*r,n[4]*r,n[5]*r,n[6]*r,n[7]*r,n[8]*r,n[9]*r,n[10]*r,n[11]*r,n[12]*r,n[13]*r,n[14]*r,n[15]*r])}},{key:"getLeft",value:function(){return i([this.e[0],this.e[4],this.e[8]]).normalize()}},{key:"getUp",value:function(){return i([this.e[1],this.e[5],this.e[9]]).normalize()}},{key:"getForward",value:function(){return i([this.e[2],this.e[6],this.e[10]]).normalize()}},{key:"setRotation",value:function(e,t,n){return void 0!==n&&null!==n&&null!==n.e[0]?void 0===t||null===t||!0===t?this.x(o([1,0,0,0,0,Math.cos(e),-Math.sin(e),0,0,Math.sin(e),Math.cos(e),0,0,0,0,1])):o([1,0,0,0,0,Math.cos(e),-Math.sin(e),0,0,Math.sin(e),Math.cos(e),0,0,0,0,1]):void 0!==n&&null!==n&&null!==n.e[1]?void 0===t||null===t||!0===t?this.x(o([Math.cos(e),0,Math.sin(e),0,0,1,0,0,-Math.sin(e),0,Math.cos(e),0,0,0,0,1])):o([Math.cos(e),0,Math.sin(e),0,0,1,0,0,-Math.sin(e),0,Math.cos(e),0,0,0,0,1]):void 0!==n&&null!==n&&null!==n.e[2]?void 0===t||null===t||!0===t?this.x(o([Math.cos(e),-Math.sin(e),0,0,Math.sin(e),Math.cos(e),0,0,0,0,1,0,0,0,0,1])):o([Math.cos(e),-Math.sin(e),0,0,Math.sin(e),Math.cos(e),0,0,0,0,1,0,0,0,0,1]):void 0}},{key:"setRotationX",value:function(e,t){return this.setRotation(e,t,i([1,0,0]))}},{key:"setRotationY",value:function(e,t){return this.setRotation(e,t,i([0,1,0]))}},{key:"setRotationZ",value:function(e,t){return this.setRotation(e,t,i([0,0,1]))}},{key:"setScale",value:function(e,t,n){return void 0!==n&&null!==n&&null!==n.e[0]?void 0===t||null===t||!0===t?this.x(o([e,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])):o([e,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]):void 0!==n&&null!==n&&null!==n.e[1]?void 0===t||null===t||!0===t?this.x(o([1,0,0,0,0,e,0,0,0,0,1,0,0,0,0,1])):o([1,0,0,0,0,e,0,0,0,0,1,0,0,0,0,1]):void 0!==n&&null!==n&&null!==n.e[2]?void 0===t||null===t||!0===t?this.x(o([1,0,0,0,0,1,0,0,0,0,e,0,0,0,0,1])):o([1,0,0,0,0,1,0,0,0,0,e,0,0,0,0,1]):void 0}},{key:"setScaleX",value:function(e,t){return this.setScale(e,t,i([1,0,0]))}},{key:"setScaleY",value:function(e,t){return this.setScale(e,t,i([0,1,0]))}},{key:"setScaleZ",value:function(e,t){return this.setScale(e,t,i([0,0,1]))}},{key:"getPosition",value:function(){return i([this.e[3],this.e[7],this.e[11]])}},{key:"setPosition",value:function(e){return this.e[3]=e.e[0],this.e[7]=e.e[1],this.e[11]=e.e[2],this}}],[{key:"makeLookAt",value:function(e,t,n,r,s,a,l,u,c){var h=i([e,t,n]),f=i([r,s,a]),d=i([l,u,c]),g=h.subtract(f).normalize(),_=d.cross(g).normalize(),v=g.cross(_).normalize(),p=o([_.e[0],_.e[1],_.e[2],0,v.e[0],v.e[1],v.e[2],0,g.e[0],g.e[1],g.e[2],0,0,0,0,1]),m=o([1,0,0,-e,0,1,0,-t,0,0,1,-n,0,0,0,1]);return p.x(m)}},{key:"setPerspectiveProjection",value:function(e,t,n,r){var i=Math.tan(e*Math.PI/360)*n,s=-i,a=t*s,l=t*i;return o([2*n/(l-a),0,(l+a)/(l-a),0,0,2*n/(i-s),(i+s)/(i-s),0,0,0,-(r+n)/(r-n),-2*r*n/(r-n),0,0,-1,0])}},{key:"setOrthographicProjection",value:function(e,t,n,r,i,s){return o([2/(t-e),0,0,-(t+e)/(t-e),0,2/(r-n),0,-(r+n)/(r-n),0,0,-2/(s-i),-(s+i)/(s-i),0,0,0,1])}}]),e}();e.StormM16=a,t.exports.StormM16=a,e.$M16=o,t.exports.$M16=o;var l=n.StormV3=function(){function e(t){r(this,e),this.e=new Float32Array(t)}return s(e,[{key:"add",value:function(e){return i([this.e[0]+e.e[0],this.e[1]+e.e[1],this.e[2]+e.e[2]])}},{key:"cross",value:function(e){return i([this.e[1]*e.e[2]-this.e[2]*e.e[1],this.e[2]*e.e[0]-this.e[0]*e.e[2],this.e[0]*e.e[1]-this.e[1]*e.e[0]])}},{key:"distance",value:function(e){return Math.sqrt((this.e[0]-e.e[0])*(this.e[0]-e.e[0])+(this.e[1]-e.e[1])*(this.e[1]-e.e[1])+(this.e[2]-e.e[2])*(this.e[2]-e.e[2]))}},{key:"dot",value:function(e){return this.e[0]*e.e[0]+this.e[1]*e.e[1]+this.e[2]*e.e[2]}},{key:"equal",value:function(e){return this.e[0]===e.e[0]&&this.e[1]===e.e[1]&&this.e[2]===e.e[2]}},{key:"modulus",value:function(){return Math.sqrt(this.sumComponentSqrs())}},{key:"sumComponentSqrs",value:function(){var e=this.sqrComponents();return e[0]+e[1]+e[2]}},{key:"sqrComponents",value:function(){var e=new Float32Array(3);return e.set([this.e[0]*this.e[0],this.e[1]*this.e[1],this.e[2]*this.e[2]]),e}},{key:"x",value:function(t){var n=t instanceof a;if(t instanceof e)return i([this.e[0]*t.e[0],this.e[1]*t.e[1],this.e[2]*t.e[2]]);if(n){var r=o([1,0,0,this.e[0],0,1,0,this.e[1],0,0,1,this.e[2],0,0,0,1]);return o([r.e[0]+0+0+0,0+r.e[1]+0+0,0+r.e[2]+0,r.e[0]*t.e[0]+r.e[1]*t.e[1]+r.e[2]*t.e[2]+r.e[3],r.e[4]+0+0+0,0+r.e[5]+0+0,0+r.e[6]+0,r.e[4]*t.e[0]+r.e[5]*t.e[1]+r.e[6]*t.e[2]+r.e[7],r.e[8]+0+0+0,0+r.e[9]+0+0,0+r.e[10]+0,r.e[8]*t.e[0]+r.e[9]*t.e[1]+r.e[10]*t.e[2]+r.e[11],r.e[12]+0+0+0,0+r.e[13]+0+0,0+r.e[14]+0,r.e[12]*t.e[0]+r.e[13]*t.e[1]+r.e[14]*t.e[2]+r.e[15]*t.e[15]])}return i([this.e[0]*t,this.e[1]*t,this.e[2]*t])}},{key:"reflect",value:function(e){var t=i([this.e[0],this.e[1],this.e[2]]),n=i([e.e[0],e.e[1],e.e[2]]),r=n.x(t.dot(n));return r=r.x(2),r=t.subtract(r)}},{key:"subtract",value:function(e){return i([this.e[0]-e.e[0],this.e[1]-e.e[1],this.e[2]-e.e[2]])}},{key:"normalize",value:function(){var e=1/this.modulus();return i([this.e[0]*e,this.e[1]*e,this.e[2]*e])}}]),e}();e.StormV3=l,t.exports.StormV3=l,e.$V3=i,t.exports.$V3=i}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],21:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.SystemEvents=void 0;var i=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}(),s=e("./Utils.class"),a=e("./Constants"),l=n.SystemEvents=function(){function e(t,n){o(this,e),this._sce=t,this._project=this._sce.getLoadedProject(),this._target=n,this._utils=new s.Utils,this.mousePosX_orig=0,this.mousePosY_orig=0,this.mousePosX=0,this.mousePosY=0,this.mouseOldPosX=0,this.mouseOldPosY=0,this.divPositionX=0,this.divPositionY=0,this.focused=!1}return i(e,[{key:"initialize",value:function(){document.body.addEventListener("keydown",this.keydownListener.bind(this)),document.body.addEventListener("keyup",this.keyupListener.bind(this)),this._target.addEventListener("mousewheel",this.mousewheelListener.bind(this)),this._target.addEventListener("wheel",this.mousewheelListener.bind(this)),document.body.addEventListener("mouseup",this.mouseupListener.bind(this),!1),document.body.addEventListener("touchend",this.mouseupListener.bind(this),!1),this._target.addEventListener("mousedown",this.mousedownListener.bind(this),!1),this._target.addEventListener("touchstart",this.mousedownListener.bind(this).bind(this),!1),document.body.addEventListener("mousemove",this.mousemoveListener.bind(this),!1),document.body.addEventListener("touchmove",this.mousemoveListener.bind(this),!1),this._target.addEventListener("mouseover",function(){this.focused=!0}.bind(this),!1),this._target.addEventListener("mouseleave",function(){this.focused=!1}.bind(this),!1)}},{key:"getMousePosition",value:function(){return{x:this.mousePosX,y:this.mousePosY}}},{key:"updateDivPosition",value:function(e){this.divPositionX=s.Utils.getElementPosition(this._sce.getCanvas()).x,this.divPositionY=s.Utils.getElementPosition(this._sce.getCanvas()).y,this.mousePosX=e.clientX-this.divPositionX,this.mousePosY=e.clientY-this.divPositionY,this.mousePosX_orig=this.mousePosX,this.mousePosY_orig=this.mousePosY,this.mouseOldPosX=this.mousePosX,this.mouseOldPosY=this.mousePosY}},{key:"callComponentEvent",value:function(e,t,n){if(void 0!==this._project&&null!==this._project){var r=this._project.getActiveStage(),o=r.getActiveCamera().getComponent(a.Constants.COMPONENT_TYPES.PROJECTION),i=null;if(t===a.Constants.EVENT_TYPES.MOUSE_DOWN&&this.updateDivPosition(n),t===a.Constants.EVENT_TYPES.MOUSE_MOVE&&(this.mouseOldPosX=this.mousePosX,this.mouseOldPosY=this.mousePosY,this.mousePosX=n.clientX-this.divPositionX,this.mousePosY=n.clientY-this.divPositionY,i=$V3([this.mousePosX-this.mousePosX_orig,0,this.mousePosY-this.mousePosY_orig])),t===a.Constants.EVENT_TYPES.MOUSE_WHEEL){this.updateDivPosition(n);var s=o.getFov(),l=(this.mousePosX-this._sce.getCanvas().width/2)/this._sce.getCanvas().width*s*.2,u=(this.mousePosY-this._sce.getCanvas().height/2)/this._sce.getCanvas().height*s*.2;(void 0!==n.wheelDeltaY&&n.wheelDeltaY>=0||void 0!==n.deltaY&&n.deltaY>=0)&&(l*=-1,u*=-1);var c=r.getActiveCamera().getComponent(a.Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix(),h=c.getLeft().x(l),f=c.getUp().x(u);i=h.add(f)}for(var d=0,g=r.getNodes().length;d<g;d++)for(var _ in r.getNodes()[d].getComponents()){var v=r.getNodes()[d].getComponent(_);v.type===e&&(t===a.Constants.EVENT_TYPES.KEY_DOWN&&null!=v._onkeydown&&!0===this.focused&&v._onkeydown(n),t===a.Constants.EVENT_TYPES.KEY_UP&&null!=v._onkeyup&&v._onkeyup(n),t===a.Constants.EVENT_TYPES.MOUSE_DOWN&&null!=v._onmousedown&&v._onmousedown(n),t===a.Constants.EVENT_TYPES.MOUSE_UP&&null!=v._onmouseup&&v._onmouseup(n),t===a.Constants.EVENT_TYPES.MOUSE_MOVE&&null!=v._onmousemove&&v._onmousemove(n,i),t===a.Constants.EVENT_TYPES.MOUSE_WHEEL&&null!=v._onmousewheel&&v._onmousewheel(n,i))}}}},{key:"keydownListener",value:function(e){this.callComponentEvent(a.Constants.COMPONENT_TYPES.KEYBOARD_EVENTS,a.Constants.EVENT_TYPES.KEY_DOWN,e)}},{key:"keyupListener",value:function(e){this.callComponentEvent(a.Constants.COMPONENT_TYPES.KEYBOARD_EVENTS,a.Constants.EVENT_TYPES.KEY_UP,e)}},{key:"mousedownListener",value:function(e){this.callComponentEvent(a.Constants.COMPONENT_TYPES.MOUSE_EVENTS,a.Constants.EVENT_TYPES.MOUSE_DOWN,e)}},{key:"mouseupListener",value:function(e){this.callComponentEvent(a.Constants.COMPONENT_TYPES.MOUSE_EVENTS,a.Constants.EVENT_TYPES.MOUSE_UP,e)}},{key:"mousemoveListener",value:function(e){this.callComponentEvent(a.Constants.COMPONENT_TYPES.MOUSE_EVENTS,a.Constants.EVENT_TYPES.MOUSE_MOVE,e)}},{key:"mousewheelListener",value:function(e){e.preventDefault(),this.callComponentEvent(a.Constants.COMPONENT_TYPES.MOUSE_EVENTS,a.Constants.EVENT_TYPES.MOUSE_WHEEL,e)}}]),e}();r.SystemEvents=l,t.exports.SystemEvents=l}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./Constants":11,"./Utils.class":22}],22:[function(e,t,n){(function(r){"use strict";function o(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}Object.defineProperty(n,"__esModule",{value:!0}),n.Utils=void 0;var i="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e},s=function(){function e(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}return function(t,n,r){return n&&e(t.prototype,n),r&&e(t,r),t}}();e("webclgl");var a=n.Utils=function(){function e(){o(this,e),window.requestAnimFrame=window.requestAnimationFrame||window.webkitRequestAnimationFrame||window.mozRequestAnimationFrame||window.oRequestAnimationFrame||window.msRequestAnimationFrame||function(e){window.setTimeout(e,1e3/60)}}return s(e,[{key:"getImageFromCanvas",value:function(e,t){var n=document.createElement("img");n.onload=function(){t(n)},n.src=e.toDataURL()}},{key:"pack",value:function(t){var n=[1/255,1/255,1/255,0],r=t,o=e.fract(255*r),i=e.fract(255*o),s=[r,o,i,e.fract(255*i)],a=[s[1]*n[0],s[2]*n[1],s[3]*n[2],s[3]*n[3]];return[s[0]-a[0],s[1]-a[1],s[2]-a[2],s[3]-a[3]]}},{key:"unpack",value:function(t){var n=[1,1/255,1/65025,1/16581375];return e.dot4(t,n)}}],[{key:"getCanvasFromUint8Array",value:function(e,t,n){var r=document.createElement("canvas");r.width=t,r.height=n;for(var o=r.getContext("2d"),i=o.createImageData(t,n),s=0;s<i.data.length;s++)i.data[s]=e[s];return o.putImageData(i,0,0),r}},{key:"getUint8ArrayFromHTMLImageElement",value:function(e){var t=document.createElement("canvas");t.width=e.width,t.height=e.height;var n=t.getContext("2d");return n.drawImage(e,0,0),n.getImageData(0,0,e.width,e.height).data}},{key:"getVector",value:function(t,n){var r=e.cartesianToSpherical(t),o=r.lat,i=r.lng;return o+=n*(180*Math.random()-90),i+=n*(180*Math.random()-90),e.sphericalToCartesian(1,o,i)}},{key:"getVectorGLSLFunctionString",value:function(){return"vec3 getVector(vec3 vecNormal, float degrees, vec2 vecNoise) {\nvec3 ob = cartesianToSpherical(vecNormal);float angleLat = ob.y;float angleLng = ob.z;float desvLat = (vecNoise.x*180.0)-90.0;float desvLng = (vecNoise.y*180.0)-90.0;angleLat += (degrees*desvLat);angleLng += (degrees*desvLng);return sphericalToCartesian(vec3(1.0, angleLat, angleLng));}\n"}},{key:"cartesianToSpherical",value:function(t){var n=Math.sqrt(t.e[0]*t.e[0]+t.e[1]*t.e[1]+t.e[2]*t.e[2]);return{radius:n,lat:e.radToDeg(Math.acos(t.e[1]/n)),lng:e.radToDeg(Math.atan2(t.e[2],t.e[0]))}}},{key:"cartesianToSphericalGLSLFunctionString",value:function(){return"vec3 cartesianToSpherical(vec3 vect) {\nfloat r = sqrt(vect.x*vect.x + vect.y*vect.y + vect.z*vect.z);float angleLat = radToDeg(acos(vect.y/r));float angleLng = radToDeg(atan(vect.z, vect.x));return vec3(r, angleLat, angleLng);}\n"}},{key:"sphericalToCartesian",value:function(t,n,r){var o=t,i=e.degToRad(n),s=e.degToRad(r),a=o*Math.sin(i)*Math.cos(s),l=o*Math.sin(i)*Math.sin(s),u=o*Math.cos(i);return new StormV3([a,u,l])}},{key:"sphericalToCartesianGLSLFunctionString",value:function(){return"vec3 sphericalToCartesian(vec3 vect) {\nfloat r = vect.x;float angleLat = degToRad(vect.y);float angleLng = degToRad(vect.z);float x = r*sin(angleLat)*cos(angleLng);float z = r*sin(angleLat)*sin(angleLng);float y = r*cos(angleLat);return vec3(x,y,z);}\n"}},{key:"refract",value:function(e,t,n,r){var o=n/r,i=-1*t.dot(e),s=1-o*o*(1-i*i);return e.x(o).add(t.x(o*i-Math.sqrt(s)))}},{key:"degToRad",value:function(e){return 3.14159*e/180}},{key:"degToRadGLSLFunctionString",value:function(){return"float degToRad(float deg) {return (deg*3.14159)/180.0;}"}},{key:"radToDeg",value:function(e){return e*(180/3.14159)}},{key:"radToDegGLSLFunctionString",value:function(){return"float radToDeg(float rad) {return rad*(180.0/3.14159);}"}},{key:"hexToRgb",value:function(e){var t=/^#?([a-f\d])([a-f\d])([a-f\d])$/i;e=e.replace(t,function(e,t,n,r){return t+t+n+n+r+r});var n=/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(e);return n?{r:parseInt(n[1],16),g:parseInt(n[2],16),b:parseInt(n[3],16)}:null}},{key:"rgbToHex",value:function(e){return"#"+(16777216+(e[2]|e[1]<<8|e[0]<<16)).toString(16).slice(1)}},{key:"invsqrt",value:function(e){return 1/e}},{key:"smoothstep",value:function(e,t,n){if(n<e)return 0;if(n>=t)return 1;if(e===t)return-1;var r=(n-e)/(t-e);return r*r*(3-2*r)}},{key:"dot4",value:function(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]+e[3]*t[3]}},{key:"fract",value:function(e){return e>0?e-Math.floor(e):e-Math.ceil(e)}},{key:"packGLSLFunctionString",value:function(){return"vec4 pack (float depth) {const vec4 bias = vec4( 1.0 / 255.0,1.0 / 255.0,1.0 / 255.0,0.0);float r = depth;float g = fract(r * 255.0);float b = fract(g * 255.0);float a = fract(b * 255.0);vec4 colour = vec4(r, g, b, a);return colour - (colour.yzww * bias);}"}},{key:"unpackGLSLFunctionString",value:function(){return"float unpack (vec4 colour) {const vec4 bitShifts = vec4(1.0,1.0 / 255.0,1.0 / (255.0 * 255.0),1.0 / (255.0 * 255.0 * 255.0));return dot(colour, bitShifts);}"}},{key:"rayTraversalInitSTR",value:function(){return"float wh = ceil(sqrt(uResolution*uResolution*uResolution));\nfloat cs = uGridsize/uResolution;\nfloat chs = cs/2.0;\nfloat texelSize = 1.0/(wh-1.0);\nvec3 gl = vec3(-(uGridsize/2.0), -(uGridsize/2.0), -(uGridsize/2.0));\nvec3 _r = vec3(uGridsize, uGridsize, uGridsize);\nvec3 _rRes = vec3(uResolution, uResolution, uResolution);\nvec3 _len = _r/_rRes;\nvec3 worldToVoxel(vec3 world) {\nvec3 ijk = (world - gl) / _len;\nijk = vec3(floor(ijk.x), floor(ijk.y), floor(ijk.z));\nreturn ijk;\n}\nfloat voxelToWorldX(float x) {return x * _len.x + gl.x;}\nfloat voxelToWorldY(float y) {return y * _len.y + gl.y;}\nfloat voxelToWorldZ(float z) {return z * _len.z + gl.z;}\n"}},{key:"rayTraversalSTR",value:function(e){return"vec2 getId(vec3 voxel) {\nint tex3dId = (int(voxel.y)*(int(uResolution)*int(uResolution)))+(int(voxel.z)*(int(uResolution)))+int(voxel.x);\nfloat num = float(tex3dId)/wh;\nfloat col = fract(num)*wh;\nfloat row = floor(num);\nreturn vec2(col*texelSize, row*texelSize);\n}\nvec4 getVoxel_Color(vec2 texVec, vec3 voxel, vec3 RayOrigin) {\nvec4 rgba = vec4(0.0,0.0,0.0,0.0);\nvec4 texture = sampler_voxelColor[vec2(texVec.x, texVec.y)];\nif(texture.a/255.0 > 0.5) {\nrgba = vec4(texture.rgb/255.0,distance(vec3(voxelToWorldX(voxel.x), voxelToWorldX(voxel.y), voxelToWorldX(voxel.z)),RayOrigin));\n}\nreturn rgba;\n}\nvec4 getVoxel_Pos(vec2 texVec) {\nvec4 rgba = vec4(0.0,0.0,0.0,0.0);\nvec4 texture = sampler_voxelPos[vec2(texVec.x, texVec.y)];\nrgba = vec4( ((texture.xyz/255.0)*uGridsize)-(uGridsize/2.0), 1.0);\nreturn rgba;\n}\nvec4 getVoxel_Normal(vec2 texVec) {\nvec4 rgba = vec4(0.0,0.0,0.0,0.0);\nvec4 texture = sampler_voxelNormal[vec2(texVec.x, texVec.y)];\nrgba = vec4(((texture.rgb/255.0)*2.0)-1.0, 1.0);\nreturn rgba;\n}\nstruct RayTraversalResponse {vec4 voxelColor;vec4 voxelPos;vec4 voxelNormal;};RayTraversalResponse rayTraversal(vec3 RayOrigin, vec3 RayDir) {\nvec4 fvoxelColor = vec4(0.0, 0.0, 0.0, 0.0);vec4 fvoxelPos = vec4(0.0, 0.0, 0.0, 0.0);vec4 fvoxelNormal = vec4(0.0, 0.0, 0.0, 0.0);vec3 voxel = worldToVoxel(RayOrigin);vec3 _dir = normalize(RayDir);vec3 tMax;if(RayDir.x < 0.0) tMax.x = (voxelToWorldX(voxel.x)-RayOrigin.x)/RayDir.x;if(RayDir.x > 0.0) tMax.x = (voxelToWorldX(voxel.x+1.0)-RayOrigin.x)/RayDir.x;if(RayDir.y < 0.0) tMax.y = (voxelToWorldY(voxel.y)-RayOrigin.y)/RayDir.y;if(RayDir.y < 0.0) tMax.y = (voxelToWorldY(voxel.y+1.0)-RayOrigin.y)/RayDir.y;if(RayDir.z < 0.0) tMax.z = (voxelToWorldZ(voxel.z)-RayOrigin.z)/RayDir.z;if(RayDir.z < 0.0) tMax.z = (voxelToWorldZ(voxel.z+1.0)-RayOrigin.z)/RayDir.z;float tDeltaX = _r.x/abs(RayDir.x);float tDeltaY = _r.y/abs(RayDir.y);float tDeltaZ = _r.z/abs(RayDir.z);float stepX = 1.0; float stepY = 1.0; float stepZ = 1.0;\nfloat outX = _r.x; float outY = _r.y; float outZ = _r.z;\nif(RayDir.x < 0.0) {stepX = -1.0; outX = -1.0;}if(RayDir.y < 0.0) {stepY = -1.0; outY = -1.0;}if(RayDir.z < 0.0) {stepZ = -1.0; outZ = -1.0;}vec4 color = vec4(0.0,0.0,0.0,0.0);\nbool c1; bool c2; bool c3; bool isOut;vec2 vid;for(int c = 0; c < "+e+"*2; c++) {\nc1 = bool(tMax.x < tMax.y);c2 = bool(tMax.x < tMax.z);c3 = bool(tMax.y < tMax.z);isOut = false;if (c1 && c2) {voxel.x += stepX;if(voxel.x==outX) isOut=true;tMax.x += tDeltaX;} else if(( (c1 && !c2) || (!c1 && !c3) )) {voxel.z += stepZ;if(voxel.z==outZ) isOut=true;tMax.z += tDeltaZ;} else if(!c1 && c3) {voxel.y += stepY;if(voxel.y==outY) isOut=true;tMax.y += tDeltaY;}if(isOut == true) break;\nelse {if((voxel.x >= 0.0 && voxel.x <= _rRes.x && voxel.y >= 0.0 && voxel.y <= _rRes.y && voxel.z >= 0.0 && voxel.z <= _rRes.z)) {;\nvid = getId(voxel);vec4 vcc = getVoxel_Color(vid, voxel, RayOrigin);if(vcc.a != 0.0) {fvoxelColor = vcc;break;\n}}}}fvoxelPos = getVoxel_Pos(vid);fvoxelNormal = getVoxel_Normal(vid);return RayTraversalResponse(fvoxelColor, fvoxelPos, fvoxelNormal);}\n"}},{key:"isPowerOfTwo",value:function(e){return 0==(e&e-1)}},{key:"nextHighestPowerOfTwo",value:function(e){--e;for(var t=1;t<32;t<<=1)e|=e>>t;return e+1}},{key:"getElementPosition",value:function(e){for(var t=e,n=0,r=0;"object"===(void 0===t?"undefined":i(t))&&void 0!==t.tagName;)r+=t.offsetTop,n+=t.offsetLeft,"BODY"===t.tagName.toUpperCase()&&(t=0),"object"===(void 0===t?"undefined":i(t))&&"object"===i(t.offsetParent)&&(t=t.offsetParent);return{x:n,y:r}}},{key:"getWebGLContextFromCanvas",value:function(e,t){return WebCLGLUtils.getWebGLContextFromCanvas(e,t)}},{key:"fullScreen",value:function(){document.fullscreenElement||document.mozFullScreenElement||document.webkitFullscreenElement?document.cancelFullScreen?document.cancelFullScreen():document.mozCancelFullScreen?document.mozCancelFullScreen():document.webkitCancelFullScreen&&document.webkitCancelFullScreen():document.documentElement.requestFullscreen?document.documentElement.requestFullscreen():document.documentElement.mozRequestFullScreen?document.documentElement.mozRequestFullScreen():document.documentElement.webkitRequestFullscreen&&document.documentElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT)}}]),e}();r.Utils=a,t.exports.Utils=a}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{webclgl:1}],23:[function(e,t,n){(function(e){"use strict";function r(e){this.getSrc=function(){return[["RGB"],"varying vec4 vVN;\nvarying vec4 vVT;\nvarying float vVTU;\n","vec4 vp = vertexPos[];\nvec4 vn = vertexNormal[];\nvec4 vt = vertexTexture[];\nfloat vtu = vertexTextureUnit[];\nvVN = vn;vVT = vt;vVTU = vtu;gl_Position = PMatrix * cameraWMatrix * nodeWMatrix * vp;\n","varying vec4 vVN;\nvarying vec4 vVT;\nvarying float vVTU;\n","vec4 textureColor = texture2D(texAlbedo, vVT.xy);\nreturn [textureColor];\n"]}}Object.defineProperty(n,"__esModule",{value:!0}),n.VFP_RGB=r,e.VFP_RGB=r,t.exports.VFP_RGB=r}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{}],24:[function(e,t,n){(function(n){"use strict";e("./ArrayGenerator.class"),e("./Component.class"),e("./Component_GPU.class"),e("./ComponentControllerTransformTarget.class"),e("./ComponentKeyboardEvents.class"),e("./ComponentMouseEvents.class"),e("./ComponentProjection.class"),e("./ComponentTransform.class"),e("./ComponentTransformTarget.class"),e("./Constants"),e("./Mesh.class"),e("./Node.class"),e("./Project.class"),e("./SCE.class"),e("./Stage.class"),e("./StormMath.class"),e("./SystemEvents.class"),e("./Utils.class"),e("./VFP_RGB.class"),e("./Prefabs/Grid/Grid.class"),e("./Prefabs/SimpleCamera/SimpleCamera.class"),e("./Prefabs/SimpleNode/SimpleNode.class"),t.exports.ArrayGenerator=n.ArrayGenerator=ArrayGenerator,t.exports.Component=n.Component=Component,t.exports.Component_GPU=n.Component_GPU=Component_GPU,t.exports.ComponentControllerTransformTarget=n.ComponentControllerTransformTarget=ComponentControllerTransformTarget,t.exports.ComponentKeyboardEvents=n.ComponentKeyboardEvents=ComponentKeyboardEvents,t.exports.ComponentMouseEvents=n.ComponentMouseEvents=ComponentMouseEvents,t.exports.ComponentProjection=n.ComponentProjection=ComponentProjection,t.exports.ComponentTransform=n.ComponentTransform=ComponentTransform,t.exports.ComponentTransformTarget=n.ComponentTransformTarget=ComponentTransformTarget,t.exports.Constants=n.Constants=Constants,t.exports.Mesh=n.Mesh=Mesh,t.exports.Node=n.Node=Node,t.exports.Project=n.Project=Project,t.exports.SCE=n.SCE=SCE,t.exports.Stage=n.Stage=Stage,t.exports.StormM16=n.StormM16=StormM16,t.exports.StormV3=n.StormV3=StormV3,t.exports.SystemEvents=n.SystemEvents=SystemEvents,t.exports.Utils=n.Utils=Utils,t.exports.VFP_RGB=n.VFP_RGB=VFP_RGB,t.exports.Grid=n.Grid=Grid,t.exports.SimpleCamera=n.SimpleCamera=SimpleCamera,t.exports.SimpleNode=n.SimpleNode=SimpleNode}).call(this,"undefined"!=typeof global?global:"undefined"!=typeof self?self:"undefined"!=typeof window?window:{})},{"./ArrayGenerator.class":2,"./Component.class":3,"./ComponentControllerTransformTarget.class":4,"./ComponentKeyboardEvents.class":5,"./ComponentMouseEvents.class":6,"./ComponentProjection.class":7,"./ComponentTransform.class":8,"./ComponentTransformTarget.class":9,"./Component_GPU.class":10,"./Constants":11,"./Mesh.class":12,"./Node.class":13,"./Prefabs/Grid/Grid.class":14,"./Prefabs/SimpleCamera/SimpleCamera.class":15,"./Prefabs/SimpleNode/SimpleNode.class":16,"./Project.class":17,"./SCE.class":18,"./Stage.class":19,"./StormMath.class":20,"./SystemEvents.class":21,"./Utils.class":22,"./VFP_RGB.class":23}]},{},[24]);
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],5:[function(require,module,exports){
/*!
 * Unidragger v2.3.0
 * Draggable base class
 * MIT license
 */

/*jshint browser: true, unused: true, undef: true, strict: true */

( function( window, factory ) {
  // universal module definition
  /*jshint strict: false */ /*globals define, module, require */

  if ( typeof define == 'function' && define.amd ) {
    // AMD
    define( [
      'unipointer/unipointer'
    ], function( Unipointer ) {
      return factory( window, Unipointer );
    });
  } else if ( typeof module == 'object' && module.exports ) {
    // CommonJS
    module.exports = factory(
      window,
      require('unipointer')
    );
  } else {
    // browser global
    window.Unidragger = factory(
      window,
      window.Unipointer
    );
  }

}( window, function factory( window, Unipointer ) {

'use strict';

// -------------------------- Unidragger -------------------------- //

function Unidragger() {}

// inherit Unipointer & EvEmitter
var proto = Unidragger.prototype = Object.create( Unipointer.prototype );

// ----- bind start ----- //

proto.bindHandles = function() {
  this._bindHandles( true );
};

proto.unbindHandles = function() {
  this._bindHandles( false );
};

/**
 * Add or remove start event
 * @param {Boolean} isAdd
 */
proto._bindHandles = function( isAdd ) {
  // munge isAdd, default to true
  isAdd = isAdd === undefined ? true : isAdd;
  // bind each handle
  var bindMethod = isAdd ? 'addEventListener' : 'removeEventListener';
  var touchAction = isAdd ? this._touchActionValue : '';
  for ( var i=0; i < this.handles.length; i++ ) {
    var handle = this.handles[i];
    this._bindStartEvent( handle, isAdd );
    handle[ bindMethod ]( 'click', this );
    // touch-action: none to override browser touch gestures. metafizzy/flickity#540
    if ( window.PointerEvent ) {
      handle.style.touchAction = touchAction;
    }
  }
};

// prototype so it can be overwriteable by Flickity
proto._touchActionValue = 'none';

// ----- start event ----- //

/**
 * pointer start
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.pointerDown = function( event, pointer ) {
  var isOkay = this.okayPointerDown( event );
  if ( !isOkay ) {
    return;
  }
  // track start event position
  this.pointerDownPointer = pointer;

  event.preventDefault();
  this.pointerDownBlur();
  // bind move and end events
  this._bindPostStartEvents( event );
  this.emitEvent( 'pointerDown', [ event, pointer ] );
};

// nodes that have text fields
var cursorNodes = {
  TEXTAREA: true,
  INPUT: true,
  SELECT: true,
  OPTION: true,
};

// input types that do not have text fields
var clickTypes = {
  radio: true,
  checkbox: true,
  button: true,
  submit: true,
  image: true,
  file: true,
};

// dismiss inputs with text fields. flickity#403, flickity#404
proto.okayPointerDown = function( event ) {
  var isCursorNode = cursorNodes[ event.target.nodeName ];
  var isClickType = clickTypes[ event.target.type ];
  var isOkay = !isCursorNode || isClickType;
  if ( !isOkay ) {
    this._pointerReset();
  }
  return isOkay;
};

// kludge to blur previously focused input
proto.pointerDownBlur = function() {
  var focused = document.activeElement;
  // do not blur body for IE10, metafizzy/flickity#117
  var canBlur = focused && focused.blur && focused != document.body;
  if ( canBlur ) {
    focused.blur();
  }
};

// ----- move event ----- //

/**
 * drag move
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.pointerMove = function( event, pointer ) {
  var moveVector = this._dragPointerMove( event, pointer );
  this.emitEvent( 'pointerMove', [ event, pointer, moveVector ] );
  this._dragMove( event, pointer, moveVector );
};

// base pointer move logic
proto._dragPointerMove = function( event, pointer ) {
  var moveVector = {
    x: pointer.pageX - this.pointerDownPointer.pageX,
    y: pointer.pageY - this.pointerDownPointer.pageY
  };
  // start drag if pointer has moved far enough to start drag
  if ( !this.isDragging && this.hasDragStarted( moveVector ) ) {
    this._dragStart( event, pointer );
  }
  return moveVector;
};

// condition if pointer has moved far enough to start drag
proto.hasDragStarted = function( moveVector ) {
  return Math.abs( moveVector.x ) > 3 || Math.abs( moveVector.y ) > 3;
};

// ----- end event ----- //

/**
 * pointer up
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto.pointerUp = function( event, pointer ) {
  this.emitEvent( 'pointerUp', [ event, pointer ] );
  this._dragPointerUp( event, pointer );
};

proto._dragPointerUp = function( event, pointer ) {
  if ( this.isDragging ) {
    this._dragEnd( event, pointer );
  } else {
    // pointer didn't move enough for drag to start
    this._staticClick( event, pointer );
  }
};

// -------------------------- drag -------------------------- //

// dragStart
proto._dragStart = function( event, pointer ) {
  this.isDragging = true;
  // prevent clicks
  this.isPreventingClicks = true;
  this.dragStart( event, pointer );
};

proto.dragStart = function( event, pointer ) {
  this.emitEvent( 'dragStart', [ event, pointer ] );
};

// dragMove
proto._dragMove = function( event, pointer, moveVector ) {
  // do not drag if not dragging yet
  if ( !this.isDragging ) {
    return;
  }

  this.dragMove( event, pointer, moveVector );
};

proto.dragMove = function( event, pointer, moveVector ) {
  event.preventDefault();
  this.emitEvent( 'dragMove', [ event, pointer, moveVector ] );
};

// dragEnd
proto._dragEnd = function( event, pointer ) {
  // set flags
  this.isDragging = false;
  // re-enable clicking async
  setTimeout( function() {
    delete this.isPreventingClicks;
  }.bind( this ) );

  this.dragEnd( event, pointer );
};

proto.dragEnd = function( event, pointer ) {
  this.emitEvent( 'dragEnd', [ event, pointer ] );
};

// ----- onclick ----- //

// handle all clicks and prevent clicks when dragging
proto.onclick = function( event ) {
  if ( this.isPreventingClicks ) {
    event.preventDefault();
  }
};

// ----- staticClick ----- //

// triggered after pointer down & up with no/tiny movement
proto._staticClick = function( event, pointer ) {
  // ignore emulated mouse up clicks
  if ( this.isIgnoringMouseUp && event.type == 'mouseup' ) {
    return;
  }

  this.staticClick( event, pointer );

  // set flag for emulated clicks 300ms after touchend
  if ( event.type != 'mouseup' ) {
    this.isIgnoringMouseUp = true;
    // reset flag after 300ms
    setTimeout( function() {
      delete this.isIgnoringMouseUp;
    }.bind( this ), 400 );
  }
};

proto.staticClick = function( event, pointer ) {
  this.emitEvent( 'staticClick', [ event, pointer ] );
};

// ----- utils ----- //

Unidragger.getPointerPoint = Unipointer.getPointerPoint;

// -----  ----- //

return Unidragger;

}));

},{"unipointer":6}],6:[function(require,module,exports){
/*!
 * Unipointer v2.3.0
 * base class for doing one thing with pointer event
 * MIT license
 */

/*jshint browser: true, undef: true, unused: true, strict: true */

( function( window, factory ) {
  // universal module definition
  /* jshint strict: false */ /*global define, module, require */
  if ( typeof define == 'function' && define.amd ) {
    // AMD
    define( [
      'ev-emitter/ev-emitter'
    ], function( EvEmitter ) {
      return factory( window, EvEmitter );
    });
  } else if ( typeof module == 'object' && module.exports ) {
    // CommonJS
    module.exports = factory(
      window,
      require('ev-emitter')
    );
  } else {
    // browser global
    window.Unipointer = factory(
      window,
      window.EvEmitter
    );
  }

}( window, function factory( window, EvEmitter ) {

'use strict';

function noop() {}

function Unipointer() {}

// inherit EvEmitter
var proto = Unipointer.prototype = Object.create( EvEmitter.prototype );

proto.bindStartEvent = function( elem ) {
  this._bindStartEvent( elem, true );
};

proto.unbindStartEvent = function( elem ) {
  this._bindStartEvent( elem, false );
};

/**
 * Add or remove start event
 * @param {Boolean} isAdd - remove if falsey
 */
proto._bindStartEvent = function( elem, isAdd ) {
  // munge isAdd, default to true
  isAdd = isAdd === undefined ? true : isAdd;
  var bindMethod = isAdd ? 'addEventListener' : 'removeEventListener';

  // default to mouse events
  var startEvent = 'mousedown';
  if ( window.PointerEvent ) {
    // Pointer Events
    startEvent = 'pointerdown';
  } else if ( 'ontouchstart' in window ) {
    // Touch Events. iOS Safari
    startEvent = 'touchstart';
  }
  elem[ bindMethod ]( startEvent, this );
};

// trigger handler methods for events
proto.handleEvent = function( event ) {
  var method = 'on' + event.type;
  if ( this[ method ] ) {
    this[ method ]( event );
  }
};

// returns the touch that we're keeping track of
proto.getTouch = function( touches ) {
  for ( var i=0; i < touches.length; i++ ) {
    var touch = touches[i];
    if ( touch.identifier == this.pointerIdentifier ) {
      return touch;
    }
  }
};

// ----- start event ----- //

proto.onmousedown = function( event ) {
  // dismiss clicks from right or middle buttons
  var button = event.button;
  if ( button && ( button !== 0 && button !== 1 ) ) {
    return;
  }
  this._pointerDown( event, event );
};

proto.ontouchstart = function( event ) {
  this._pointerDown( event, event.changedTouches[0] );
};

proto.onpointerdown = function( event ) {
  this._pointerDown( event, event );
};

/**
 * pointer start
 * @param {Event} event
 * @param {Event or Touch} pointer
 */
proto._pointerDown = function( event, pointer ) {
  // dismiss right click and other pointers
  // button = 0 is okay, 1-4 not
  if ( event.button || this.isPointerDown ) {
    return;
  }

  this.isPointerDown = true;
  // save pointer identifier to match up touch events
  this.pointerIdentifier = pointer.pointerId !== undefined ?
    // pointerId for pointer events, touch.indentifier for touch events
    pointer.pointerId : pointer.identifier;

  this.pointerDown( event, pointer );
};

proto.pointerDown = function( event, pointer ) {
  this._bindPostStartEvents( event );
  this.emitEvent( 'pointerDown', [ event, pointer ] );
};

// hash of events to be bound after start event
var postStartEvents = {
  mousedown: [ 'mousemove', 'mouseup' ],
  touchstart: [ 'touchmove', 'touchend', 'touchcancel' ],
  pointerdown: [ 'pointermove', 'pointerup', 'pointercancel' ],
};

proto._bindPostStartEvents = function( event ) {
  if ( !event ) {
    return;
  }
  // get proper events to match start event
  var events = postStartEvents[ event.type ];
  // bind events to node
  events.forEach( function( eventName ) {
    window.addEventListener( eventName, this );
  }, this );
  // save these arguments
  this._boundPointerEvents = events;
};

proto._unbindPostStartEvents = function() {
  // check for _boundEvents, in case dragEnd triggered twice (old IE8 bug)
  if ( !this._boundPointerEvents ) {
    return;
  }
  this._boundPointerEvents.forEach( function( eventName ) {
    window.removeEventListener( eventName, this );
  }, this );

  delete this._boundPointerEvents;
};

// ----- move event ----- //

proto.onmousemove = function( event ) {
  this._pointerMove( event, event );
};

proto.onpointermove = function( event ) {
  if ( event.pointerId == this.pointerIdentifier ) {
    this._pointerMove( event, event );
  }
};

proto.ontouchmove = function( event ) {
  var touch = this.getTouch( event.changedTouches );
  if ( touch ) {
    this._pointerMove( event, touch );
  }
};

/**
 * pointer move
 * @param {Event} event
 * @param {Event or Touch} pointer
 * @private
 */
proto._pointerMove = function( event, pointer ) {
  this.pointerMove( event, pointer );
};

// public
proto.pointerMove = function( event, pointer ) {
  this.emitEvent( 'pointerMove', [ event, pointer ] );
};

// ----- end event ----- //


proto.onmouseup = function( event ) {
  this._pointerUp( event, event );
};

proto.onpointerup = function( event ) {
  if ( event.pointerId == this.pointerIdentifier ) {
    this._pointerUp( event, event );
  }
};

proto.ontouchend = function( event ) {
  var touch = this.getTouch( event.changedTouches );
  if ( touch ) {
    this._pointerUp( event, touch );
  }
};

/**
 * pointer up
 * @param {Event} event
 * @param {Event or Touch} pointer
 * @private
 */
proto._pointerUp = function( event, pointer ) {
  this._pointerDone();
  this.pointerUp( event, pointer );
};

// public
proto.pointerUp = function( event, pointer ) {
  this.emitEvent( 'pointerUp', [ event, pointer ] );
};

// ----- pointer done ----- //

// triggered on pointer up & pointer cancel
proto._pointerDone = function() {
  this._pointerReset();
  this._unbindPostStartEvents();
  this.pointerDone();
};

proto._pointerReset = function() {
  // reset properties
  this.isPointerDown = false;
  delete this.pointerIdentifier;
};

proto.pointerDone = noop;

// ----- pointer cancel ----- //

proto.onpointercancel = function( event ) {
  if ( event.pointerId == this.pointerIdentifier ) {
    this._pointerCancel( event, event );
  }
};

proto.ontouchcancel = function( event ) {
  var touch = this.getTouch( event.changedTouches );
  if ( touch ) {
    this._pointerCancel( event, touch );
  }
};

/**
 * pointer cancel
 * @param {Event} event
 * @param {Event or Touch} pointer
 * @private
 */
proto._pointerCancel = function( event, pointer ) {
  this._pointerDone();
  this.pointerCancel( event, pointer );
};

// public
proto.pointerCancel = function( event, pointer ) {
  this.emitEvent( 'pointerCancel', [ event, pointer ] );
};

// -----  ----- //

// utility function for getting x/y coords from event
Unipointer.getPointerPoint = function( pointer ) {
  return {
    x: pointer.pageX,
    y: pointer.pageY
  };
};

// -----  ----- //

return Unipointer;

}));

},{"ev-emitter":2}],7:[function(require,module,exports){
(function (global){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

// Average Window class from CONVNETJS
var AvgWin = exports.AvgWin = function () {
    function AvgWin(size, minsize) {
        _classCallCheck(this, AvgWin);

        this.v = [];
        this.size = typeof size === 'undefined' ? 100 : size;
        this.minsize = typeof minsize === 'undefined' ? 20 : minsize;
        this.sum = 0;
    }

    _createClass(AvgWin, [{
        key: 'add',
        value: function add(x) {
            this.v.push(x);
            this.sum += x;
            if (this.v.length > this.size) {
                var xold = this.v.shift();
                this.sum -= xold;
            }
        }
    }, {
        key: 'get_average',
        value: function get_average() {
            if (this.v.length < this.minsize) return -1;else return this.sum / this.v.length;
        }
    }, {
        key: 'reset',
        value: function reset(x) {
            this.v = [];
            this.sum = 0;
        }
    }]);

    return AvgWin;
}();

global.AvgWin = AvgWin;
module.exports.AvgWin = AvgWin;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],8:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.Graph = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

require("scejs");

var _KERNEL_DIR = require("./KERNEL_DIR.class");

var _KERNEL_ADJMATRIX_UPDATE = require("./KERNEL_ADJMATRIX_UPDATE.class");

var _VFP_NODE = require("./VFP_NODE.class");

var _VFP_NODEPICKDRAG = require("./VFP_NODEPICKDRAG.class");

var _ProccessImg = require("./ProccessImg.class");

var _resources = require("./resources");

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

/**
* @class
 * @param {SCE} sce
 * @param {Object} jsonIn
 * @param {boolean} jsonIn.enableFonts
 * @param {String} [jsonIn.nodeDrawMode="plane"] - "plane" or "point"
*/
var Graph = exports.Graph = function () {
    function Graph(sce, jsonIn) {
        var _this = this;

        _classCallCheck(this, Graph);

        this._sce = sce;
        this._project = this._sce.getLoadedProject();
        this._gl = this._project.getActiveStage().getWebGLContext();
        this._utils = new Utils();

        this.NODE_IMG_COLUMNS = 8.0;
        this.NODE_IMG_WIDTH = 2048;

        this._enableFont = jsonIn && jsonIn.enableFonts === true;
        this._enableHover = false;
        this._enabledForceLayout = false;
        this._MAX_ADJ_MATRIX_WIDTH = 2048;

        this._playAnimation = false;
        this._loop = false;
        this._animationFrames = 500;

        this._geometryLength = jsonIn === undefined || jsonIn && jsonIn.nodeDrawMode === undefined || jsonIn && jsonIn.nodeDrawMode === "plane" ? 4 : 1;
        this.circleSegments = 12;
        this.nodesTextPlanes = 12;

        this.lineVertexCount = 1;

        this.layer_defs = null;
        this.layerCount = 0;
        this.batch_repeats = 1;
        this.gpu_batch_size = 7;
        this.maxacts = [];
        this.afferentNodesCount = 0;
        this.efferentNodesCount = 0;
        this.trainTickCount = 0;
        this.onAction = null;
        this.onTrained = null;
        this.afferentNeuron = [];
        this.efferentNeuron = [];
        this.v_val = 0.0;
        this.return_v = false;

        this._only2d = false;
        this.multiplyOutput = 1;

        this._nodesByName = {};
        this._nodesById = {};
        this._links = {};
        this._customArgs = {}; // {ARG: {"arg": String, "value": Array<Float>}}


        this.arrAdjMatrix = null; // null, null, linkWeight, columnAsParent
        this.arrAdjMatrixB = null;
        this.arrAdjMatrixC = null;
        this.arrAdjMatrixD = null;

        this.enableTrain = 0;
        this.updateTheta = 0;
        this._ADJ_MATRIX_WIDTH = null;

        this.readPixel = false;
        this.selectedId = -1;
        this._initialPosDrag = null;

        this._onClickNode = null;

        // meshes
        this.mesh_nodes = this._geometryLength === 1 ? new Mesh().loadPoint() : new Mesh().loadQuad(4.0, 4.0);
        this.mesh_arrows = new Mesh().loadTriangle({ "scale": 1.75, "side": 0.3 });
        this.mesh_nodesText = new Mesh().loadQuad(4.0, 4.0);

        // nodes image
        this.objNodeImages = {};

        this._stackNodesImg = [];
        this.nodesImgMask = null;
        this.nodesImgMaskLoaded = false;
        this.nodesImgCrosshair = null;
        this.nodesImgCrosshairLoaded = false;

        this.FONT_IMG_COLUMNS = 7.0;

        // Nodes
        // Data: nodeId, acums, null, null
        // DataB: biasNode, null, netFOutputA, netErrorWeightA (SHARED with LINKS, ARROWS & NODESTEXT)

        // Links
        // Data: nodeId origin, nodeId target, currentLineVertex, repeatId

        // nodes
        this.arrayNodeData = [];
        this.arrayNodeDataB = [];
        this.arrayNodeDataF = [];
        this.arrayNodeDataG = [];
        this.arrayNodeDataH = [];
        this.arrayNodePosXYZW = [];
        this.arrayNodeVertexPos = [];
        this.arrayNodeVertexNormal = [];
        this.arrayNodeVertexTexture = [];
        this.startIndexId = 0;
        this.arrayNodeIndices = [];

        this.arrayNodeImgId = [];

        this.arrayNodeDir = [];

        this.currentNodeId = 0;
        this.nodeArrayItemStart = 0;

        // links
        this.linksObj = [];
        this.currentLinksObjItem = -1;
        this.currentLinkId = 0;

        // arrows
        this.arrowsObj = [];
        this.currentArrowsObjItem = -1;
        this.currentArrowId = 0;

        /*this.arrayArrowData = [];
        this.arrayArrowNodeName = [];
        this.arrayArrowPosXYZW = [];
        this.arrayArrowVertexPos = [];
        this.arrayArrowVertexNormal = [];
        this.arrayArrowVertexTexture = [];
        this.startIndexId_arrow = 0;
        this.arrayArrowIndices = [];
          this.currentArrowId = 0;
        this.arrowArrayItemStart = 0;*/

        // nodesText
        this.arrayNodeTextData = [];
        this.arrayNodeTextNodeName = [];
        this.arrayNodeTextPosXYZW = [];
        this.arrayNodeTextVertexPos = [];
        this.arrayNodeTextVertexNormal = [];
        this.arrayNodeTextVertexTexture = [];
        this.startIndexId_nodestext = 0;
        this.arrayNodeTextIndices = [];

        this.arrayNodeText_itemStart = [];
        this.arrayNodeTextLetterId = [];

        this.currentNodeTextId = 0;
        this.nodeTextArrayItemStart = 0;

        this.layout = null;

        this.currentNeuron = 0;
        this.nodSep = 5.0;

        //**************************************************
        //  NODES
        //**************************************************
        this.nodes = new Node();
        this.nodes.setName("graph_nodes");
        this._project.getActiveStage().addNode(this.nodes);

        // ComponentTransform
        this.nodes.addComponent(new ComponentTransform());

        // Component_GPU
        this.comp_renderer_nodes = new Component_GPU();
        this.nodes.addComponent(this.comp_renderer_nodes);

        // ComponentMouseEvents
        var comp_mouseEvents = new ComponentMouseEvents();
        this.nodes.addComponent(comp_mouseEvents);
        comp_mouseEvents.onmousedown(function (evt) {
            _this.selectedId = -1;
            _this.mouseDown();
        });
        comp_mouseEvents.onmouseup(function (evt) {
            if (_this.selectedId !== -1) {
                var n = _this._nodesById[_this.selectedId];
                if (n !== undefined && n !== null && n.onmouseup !== undefined && n.onmouseup !== null) n.onmouseup(n, evt);
            }
            _this.mouseUp();
        });
        comp_mouseEvents.onmousemove(function (evt, dir) {
            _this.makeDrag(evt, dir);
        });
        comp_mouseEvents.onmousewheel(function (evt) {});

        //**************************************************
        //  LINKS
        //**************************************************
        this.createLinksObjItem();

        //**************************************************
        //  ARROWS
        //**************************************************
        this.createArrowsObjItem();

        /*let arrows = new Node();
        arrows.setName("graph_arrows");
        this._project.getActiveStage().addNode(arrows);
          // ComponentTransform
        arrows.addComponent(new ComponentTransform());
          // Component_GPU
        let comp_renderer_arrows = new Component_GPU();
        arrows.addComponent(comp_renderer_arrows);*/

        //**************************************************
        //  NODESTEXT
        //**************************************************
        this.nodesText = null;
        this.comp_renderer_nodesText = null;
        if (this._enableFont === true) {
            this.nodesText = new Node();
            this.nodesText.setName("graph_nodesText");
            this._project.getActiveStage().addNode(this.nodesText);

            // ComponentTransform
            this.nodesText.addComponent(new ComponentTransform());

            // Component_GPU
            this.comp_renderer_nodesText = new Component_GPU();
            this.nodesText.addComponent(this.comp_renderer_nodesText);

            var image = new Image();
            image.onload = function () {
                _this.comp_renderer_nodesText.setArg("fontsImg", function () {
                    return image;
                });
            };
            image.src = _resources.Resources.fonts();
        }
    }

    _createClass(Graph, [{
        key: "createLinksObjItem",
        value: function createLinksObjItem() {
            this.currentLinksObjItem++;

            this.linksObj.push({
                "node": new Node(),
                "componentTransform": new ComponentTransform(),
                "componentRenderer": new Component_GPU(),
                "arrayLinkData": [],
                "arrayLinkNodeName": [],
                "arrayLinkPosXYZW": [],
                "arrayLinkVertexPos": [],
                "startIndexId_link": 0,
                "arrayLinkIndices": [] });

            this.linksObj[this.currentLinksObjItem].node.setName("graph_links" + this.currentLinksObjItem);
            this._project.getActiveStage().addNode(this.linksObj[this.currentLinksObjItem].node);

            this.linksObj[this.currentLinksObjItem].node.addComponent(this.linksObj[this.currentLinksObjItem].componentTransform);

            this.linksObj[this.currentLinksObjItem].node.addComponent(this.linksObj[this.currentLinksObjItem].componentRenderer);
        }
    }, {
        key: "createArrowsObjItem",
        value: function createArrowsObjItem() {
            this.currentArrowsObjItem++;

            this.arrowsObj.push({
                "node": new Node(),
                "componentTransform": new ComponentTransform(),
                "componentRenderer": new Component_GPU(),
                "arrayArrowData": [],
                "arrayArrowNodeName": [],
                "arrayArrowPosXYZW": [],
                "arrayArrowVertexPos": [],
                "arrayArrowVertexNormal": [],
                "arrayArrowVertexTexture": [],
                "startIndexId_arrow": 0,
                "arrayArrowIndices": [],
                "arrowArrayItemStart": 0 });

            this.arrowsObj[this.currentArrowsObjItem].node.setName("graph_arrows" + this.currentArrowsObjItem);
            this._project.getActiveStage().addNode(this.arrowsObj[this.currentArrowsObjItem].node);

            this.arrowsObj[this.currentArrowsObjItem].node.addComponent(this.arrowsObj[this.currentArrowsObjItem].componentTransform);

            this.arrowsObj[this.currentArrowsObjItem].node.addComponent(this.arrowsObj[this.currentArrowsObjItem].componentRenderer);
        }
    }, {
        key: "onClickNode",


        /**
         * @param {Function} fn
         */
        value: function onClickNode(fn) {
            this._onClickNode = fn;
        }
    }, {
        key: "getNodesCount",


        /**
         * @returns {int}
         */
        value: function getNodesCount() {
            return Object.keys(this._nodesByName).length;
        }
    }, {
        key: "getLinksCount",


        /**
         * @returns {int}
         */
        value: function getLinksCount() {
            return Object.keys(this._links).length;
        }
    }, {
        key: "getNodeByName",


        /**
         * @param {String} name
         * @returns {Node}
         */
        value: function getNodeByName(name) {
            return this._nodesByName[name];
        }
    }, {
        key: "getNodeById",


        /**
         * @param {int} id
         * @returns {Node}
         */
        value: function getNodeById(id) {
            return this._nodesById[id];
        }
    }, {
        key: "selectNode",


        /**
         * @param {int} nodeId
         */
        value: function selectNode(nodeId) {
            this.selectedId = nodeId;
            this.makeDrag(null, $V3([0.0, 0.0, 0.0]));
        }
    }, {
        key: "getSelectedId",


        /**
         * @returns {number}
         */
        value: function getSelectedId() {
            return this.selectedId;
        }
    }, {
        key: "enableHover",
        value: function enableHover() {
            this._enableHover = true;
            this.readPixel = true;
            this.comp_renderer_nodes.gpufG.enableGraphic(1);
        }
    }, {
        key: "only2d",


        /**
         *  @param {boolean} mode2d
         */
        value: function only2d(mode2d) {
            this._only2d = mode2d;
        }
    }, {
        key: "setNodeMesh",


        /**
         * @param {Mesh} mesh
         */
        value: function setNodeMesh(mesh) {
            this.mesh_nodes = mesh;
        }
    }, {
        key: "applyLayout",


        // ██████╗ ██████╗ ██╗   ██╗███████╗ ██████╗ ██████╗
        //██╔════╝ ██╔══██╗██║   ██║██╔════╝██╔═══██╗██╔══██╗
        //██║  ███╗██████╔╝██║   ██║█████╗  ██║   ██║██████╔╝
        //██║   ██║██╔═══╝ ██║   ██║██╔══╝  ██║   ██║██╔══██╗
        //╚██████╔╝██║     ╚██████╔╝██║     ╚██████╔╝██║  ██║
        // ╚═════╝ ╚═╝      ╚═════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝
        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.argsDirection - example "float4* argA, float* argB, mat4 argC, float4 argD, float argE"
         * @param {String} jsonIn.codeDirection
         * @param {String} jsonIn.argsObject - example "float4* argA, float* argB, mat4 argC, float4 argD, float argE"
         * @param {String} jsonIn.codeObject
         */
        value: function applyLayout(jsonIn) {
            this.layout = jsonIn;
            // Create custom user arrays args
            var createCustomArgsArrays = function createCustomArgsArrays(obj, arr) {
                for (var n = 0, fn = arr.length; n < fn; n++) {
                    obj[arr[n].trim().split(" ")[1]] = { "arg": arr[n].trim(),
                        "nodes_array_value": [],
                        "links_array_value": [],
                        "arrows_array_value": [],
                        "nodestext_array_value": [] };
                }
                return obj;
            };

            this.layout.argsDirection = this.layout.argsDirection !== undefined && this.layout.argsDirection !== null ? this.layout.argsDirection.split(",") : null;
            this.layout.argsObject = this.layout.argsObject !== undefined && this.layout.argsObject !== null ? this.layout.argsObject.split(",") : null;

            this._customArgs = {};
            if (this.layout.argsDirection != null) this._customArgs = createCustomArgsArrays(this._customArgs, this.layout.argsDirection);

            if (this.layout.argsObject != null) this._customArgs = createCustomArgsArrays(this._customArgs, this.layout.argsObject);
        }
    }, {
        key: "createWebGLBuffers",
        value: function createWebGLBuffers() {
            var _this2 = this;

            var varDef_VFPNode = {
                'float4* posXYZW': function float4PosXYZW() {
                    return null;
                },
                "float4* dataB": function float4DataB() {
                    return null;
                }, // in nodes (SHARED with LINKS, ARROWS & NODESTEXT)
                "float4* dataF": function float4DataF() {
                    return null;
                }, // in nodes
                "float4* dataG": function float4DataG() {
                    return null;
                }, // in nodes
                "float4* dataH": function float4DataH() {
                    return null;
                }, // in nodes
                "float4*attr data": function float4AttrData() {
                    return null;
                }, // in nodes, nodesText, links & arrows
                'float4*attr nodeVertexPos': function float4AttrNodeVertexPos() {
                    return null;
                },
                'float4*attr nodeVertexNormal': function float4AttrNodeVertexNormal() {
                    return null;
                },
                'float4*attr nodeVertexTexture': function float4AttrNodeVertexTexture() {
                    return null;
                },
                'float*attr letterId': function floatAttrLetterId() {
                    return null;
                },
                'float*attr nodeImgId': function floatAttrNodeImgId() {
                    return null;
                },
                'indices': function indices() {
                    return null;
                },
                "float4* adjacencyMatrix": function float4AdjacencyMatrix() {
                    return null;
                },
                "float4* adjacencyMatrixB": function float4AdjacencyMatrixB() {
                    return null;
                },
                "float4* adjacencyMatrixC": function float4AdjacencyMatrixC() {
                    return null;
                },
                "float4* adjacencyMatrixD": function float4AdjacencyMatrixD() {
                    return null;
                },
                "float widthAdjMatrix": function floatWidthAdjMatrix() {
                    return null;
                },
                'float nodesCount': function floatNodesCount() {
                    return null;
                },
                'mat4 PMatrix': function mat4PMatrix() {
                    return null;
                },
                'mat4 cameraWMatrix': function mat4CameraWMatrix() {
                    return null;
                },
                'mat4 nodeWMatrix': function mat4NodeWMatrix() {
                    return null;
                },
                'float isNode': function floatIsNode() {
                    return null;
                },
                'float isLink': function floatIsLink() {
                    return null;
                },
                'float isArrow': function floatIsArrow() {
                    return null;
                },
                'float isNodeText': function floatIsNodeText() {
                    return null;
                },
                'float bufferNodesWidth': function floatBufferNodesWidth() {
                    return null;
                },
                'float bufferLinksWidth': function floatBufferLinksWidth() {
                    return null;
                },
                'float bufferArrowsWidth': function floatBufferArrowsWidth() {
                    return null;
                },
                'float bufferTextsWidth': function floatBufferTextsWidth() {
                    return null;
                },
                'float idToDrag': function floatIdToDrag() {
                    return null;
                },
                'float idToHover': function floatIdToHover() {
                    return null;
                },
                "float enableForceLayout": function floatEnableForceLayout() {
                    return null;
                },
                'float currentTrainLayer': function floatCurrentTrainLayer() {
                    return null;
                },
                'float layerCount': function floatLayerCount() {
                    return null;
                },
                "float learningRate": function floatLearningRate() {
                    return null;
                },
                "float batch_repeats": function floatBatch_repeats() {
                    return null;
                },
                'float afferentNodesCount': function floatAfferentNodesCount() {
                    return null;
                },
                'float efferentNodesCount': function floatEfferentNodesCount() {
                    return null;
                },
                'float efferentStart': function floatEfferentStart() {
                    return null;
                },
                'float freezeOutput': function floatFreezeOutput() {
                    return null;
                },
                'float enableTrain': function floatEnableTrain() {
                    return null;
                },
                'float updateTheta': function floatUpdateTheta() {
                    return null;
                },
                'float multiplyOutput': function floatMultiplyOutput() {
                    return null;
                },
                'float viewNeuronDynamics': function floatViewNeuronDynamics() {
                    return null;
                },
                'float showValues': function floatShowValues() {
                    return null;
                },
                'float only2d': function floatOnly2d() {
                    return null;
                },
                'float nodeImgColumns': function floatNodeImgColumns() {
                    return null;
                },
                'float fontImgColumns': function floatFontImgColumns() {
                    return null;
                },
                'float4* fontsImg': function float4FontsImg() {
                    return null;
                },
                'float4* nodesImg': function float4NodesImg() {
                    return null;
                },
                'float4* nodesImgCrosshair': function float4NodesImgCrosshair() {
                    return null;
                }
            };
            var aC = this.afferentNodesCount === 0 ? 1 : this.afferentNodesCount;
            var eC = this.efferentNodesCount === 0 ? 1 : this.efferentNodesCount;
            varDef_VFPNode['float afferentNodesA[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesA[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesB[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesB[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesC[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesC[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesD[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesD[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesE[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesE[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesF[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesF[' + eC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float afferentNodesG[' + aC + ']'] = function () {
                return null;
            };
            varDef_VFPNode['float efferentNodesG[' + eC + ']'] = function () {
                return null;
            };

            if (this.layout.argsDirection !== undefined && this.layout.argsDirection !== null) {
                for (var n = 0; n < this.layout.argsDirection.length; n++) {
                    varDef_VFPNode[this.layout.argsDirection[n]] = function () {
                        return null;
                    };
                }
            }

            if (this.layout.argsObject !== undefined && this.layout.argsObject !== null) {
                for (var _n = 0; _n < this.layout.argsObject.length; _n++) {
                    varDef_VFPNode[this.layout.argsObject[_n]] = function () {
                        return null;
                    };
                }
            }

            var varDef_NodesKernel = {
                'float4* dir': function float4Dir() {
                    return null;
                },
                'float nodesCount': function floatNodesCount() {
                    return null;
                },
                'float enableDrag': function floatEnableDrag() {
                    return null;
                },
                'float initialPosX': function floatInitialPosX() {
                    return null;
                },
                'float initialPosY': function floatInitialPosY() {
                    return null;
                },
                'float initialPosZ': function floatInitialPosZ() {
                    return null;
                },
                'float MouseDragTranslationX': function floatMouseDragTranslationX() {
                    return null;
                },
                'float MouseDragTranslationY': function floatMouseDragTranslationY() {
                    return null;
                },
                'float MouseDragTranslationZ': function floatMouseDragTranslationZ() {
                    return null;
                } };

            ///////////////////////////////////////////////////////////////////////////////////////////
            //                          LINKS
            ///////////////////////////////////////////////////////////////////////////////////////////
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setGPUFor(this.linksObj[na].componentRenderer.gl, Object.create(varDef_VFPNode), { "type": "GRAPHIC",
                    "name": "LINKS_VFP_NODE",
                    "viewSource": false,
                    "config": _VFP_NODE.VFP_NODE.getSrc(this.layout.codeObject, this._geometryLength),
                    "drawMode": 1,
                    "depthTest": true,
                    "blend": false,
                    "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                    "blendSrcMode": Constants.BLENDING_MODES.ONE,
                    "blendDstMode": Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA });
                this.linksObj[na].componentRenderer.getComponentBufferArg("RGB", this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.GPU));
            }
            ///////////////////////////////////////////////////////////////////////////////////////////
            //                          ARROWS
            ///////////////////////////////////////////////////////////////////////////////////////////
            for (var _na = 0; _na < this.arrowsObj.length; _na++) {
                this.arrowsObj[_na].componentRenderer.setGPUFor(this.arrowsObj[_na].componentRenderer.gl, Object.create(varDef_VFPNode), { "type": "GRAPHIC",
                    "name": "ARROWS_VFP_NODE",
                    "viewSource": false,
                    "config": _VFP_NODE.VFP_NODE.getSrc(this.layout.codeObject, this._geometryLength),
                    "drawMode": 4,
                    "depthTest": true,
                    "blend": true,
                    "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                    "blendSrcMode": Constants.BLENDING_MODES.SRC_ALPHA,
                    "blendDstMode": Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA });
                this.arrowsObj[_na].componentRenderer.getComponentBufferArg("RGB", this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.GPU));
            }
            ///////////////////////////////////////////////////////////////////////////////////////////
            //                          NODES
            ///////////////////////////////////////////////////////////////////////////////////////////
            var nodesVarDef = Object.create(varDef_VFPNode);
            for (var key in varDef_NodesKernel) {
                nodesVarDef[key] = varDef_NodesKernel[key];
            }this.comp_renderer_nodes.setGPUFor(this.comp_renderer_nodes.gl, nodesVarDef, { "type": "KERNEL",
                "name": "NODES_KERNEL_DIR",
                "viewSource": false,
                "config": _KERNEL_DIR.KERNEL_DIR.getSrc(this.layout.codeDirection, this._geometryLength, this.afferentNodesCount, this.currentNodeId - this.efferentNodesCount, this.efferentNodesCount),
                "drawMode": this._geometryLength === 1 ? 0 : 4,
                "depthTest": true,
                "blend": false,
                "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                "blendSrcMode": Constants.BLENDING_MODES.ONE,
                "blendDstMode": Constants.BLENDING_MODES.ONE }, { "type": "KERNEL",
                "name": "NODES_KERNEL_ADJMATRIX_UPDATE",
                "viewSource": false,
                "config": _KERNEL_ADJMATRIX_UPDATE.KERNEL_ADJMATRIX_UPDATE.getSrc(this._geometryLength),
                "drawMode": this._geometryLength === 1 ? 0 : 4,
                "depthTest": true,
                "blend": false,
                "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                "blendSrcMode": Constants.BLENDING_MODES.ONE,
                "blendDstMode": Constants.BLENDING_MODES.ONE }, { "type": "GRAPHIC",
                "name": "NODES_VFP_NODE",
                "viewSource": false,
                "config": _VFP_NODE.VFP_NODE.getSrc(this.layout.codeObject, this._geometryLength),
                "drawMode": this._geometryLength === 1 ? 0 : 4,
                "depthTest": true,
                "blend": true,
                "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                "blendSrcMode": Constants.BLENDING_MODES.SRC_ALPHA,
                "blendDstMode": Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA }, { "type": "GRAPHIC",
                "name": "NODES_VFP_NODEPICKDRAG",
                "viewSource": false,
                "config": _VFP_NODEPICKDRAG.VFP_NODEPICKDRAG.getSrc(this._geometryLength),
                "drawMode": this._geometryLength === 1 ? 0 : 4,
                "depthTest": true,
                "blend": true,
                "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                "blendSrcMode": Constants.BLENDING_MODES.ONE,
                "blendDstMode": Constants.BLENDING_MODES.ZERO });
            var comp_screenEffects = this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.GPU);
            this.comp_renderer_nodes.getComponentBufferArg("RGB", comp_screenEffects);

            // KERNEL_DIR
            this.comp_renderer_nodes.gpufG.onPreProcessKernel(0, function () {
                _this2.comp_renderer_nodes.setArg("only2d", function () {
                    return _this2._only2d === true ? 1.0 : 0.0;
                });
            });

            // KERNEL_ADJMATRIX_UPDATE
            this.comp_renderer_nodes.gpufG.disableKernel(1);

            // VFP_NODEPICKDRAG
            this.comp_renderer_nodes.gpufG.onPostProcessGraphic(1, function () {
                _this2.procSelectedOrHover();
            });
            this.comp_renderer_nodes.gpufG.disableGraphic(1);

            ///////////////////////////////////////////////////////////////////////////////////////////
            //                          NODESTEXT
            ///////////////////////////////////////////////////////////////////////////////////////////
            if (this._enableFont === true) {
                this.comp_renderer_nodesText.setGPUFor(this.comp_renderer_nodesText.gl, Object.create(varDef_VFPNode), { "type": "GRAPHIC",
                    "name": "NODESTEXT_VFP_NODE",
                    "viewSource": false,
                    "config": _VFP_NODE.VFP_NODE.getSrc(this.layout.codeObject, this._geometryLength),
                    "drawMode": 4,
                    "depthTest": true,
                    "blend": true,
                    "blendEquation": Constants.BLENDING_EQUATION_TYPES.FUNC_ADD,
                    "blendSrcMode": Constants.BLENDING_MODES.SRC_ALPHA,
                    "blendDstMode": Constants.BLENDING_MODES.ONE_MINUS_SRC_ALPHA });
                this.comp_renderer_nodesText.getComponentBufferArg("RGB", this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.GPU));
            }

            this.enableHov(-1);

            this.updateNodes();
            this.updateLinks();

            this.comp_renderer_nodes.setArg("enableTrain", function () {
                return 1.0;
            });
            this.comp_renderer_nodes.gpufG.enableKernel(1);
            this.comp_renderer_nodes.setArg("currentTrainLayer", function () {
                return -10.0;
            });
            this._sce.getLoadedProject().getActiveStage().tick();
            this.comp_renderer_nodes.gpufG.disableKernel(1);
            this.comp_renderer_nodes.setArg("enableTrain", function () {
                return 0.0;
            });
        }
    }, {
        key: "mouseDown",
        value: function mouseDown() {
            if (this._enableHover === false) {
                this.readPixel = true;

                this.comp_renderer_nodes.gpufG.enableGraphic(1);
            }
        }
    }, {
        key: "mouseUp",
        value: function mouseUp() {
            this.comp_renderer_nodes.setArg("enableDrag", function () {
                return 0;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("enableDrag", function () {
                    return 0;
                });
            }for (var _na2 = 0; _na2 < this.arrowsObj.length; _na2++) {
                this.arrowsObj[_na2].componentRenderer.setArg("enableDrag", function () {
                    return 0;
                });
            }if (this._enableFont === true) this.comp_renderer_nodesText.setArg("enableDrag", function () {
                return 0;
            });

            if (this.selectedId === -1) {
                this.comp_renderer_nodes.setArg("idToDrag", function () {
                    return -1;
                });
                for (var _na3 = 0; _na3 < this.linksObj.length; _na3++) {
                    this.linksObj[_na3].componentRenderer.setArg("idToDrag", function () {
                        return -1;
                    });
                }for (var _na4 = 0; _na4 < this.arrowsObj.length; _na4++) {
                    this.arrowsObj[_na4].componentRenderer.setArg("idToDrag", function () {
                        return -1;
                    });
                }if (this._enableFont === true) this.comp_renderer_nodesText.setArg("idToDrag", function () {
                    return -1;
                });
            }

            if (this._enableHover === true) {
                this.readPixel = true;

                this.comp_renderer_nodes.gpufG.enableGraphic(1);
            }
        }
    }, {
        key: "procSelectedOrHover",
        value: function procSelectedOrHover() {
            if (this._enableHover === false) {
                if (this.readPixel === true) {
                    this.readPixel = false;

                    this.readPix();

                    this.comp_renderer_nodes.gpufG.disableGraphic(1);
                }
            } else {
                var comp_controller_trans_target = this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.CONTROLLER_TRANSFORM_TARGET);
                if (comp_controller_trans_target.isLeftBtnActive() === true) {
                    this.readPixel = false;
                }

                this.readPix();

                if (comp_controller_trans_target.isLeftBtnActive() === true) {
                    this.comp_renderer_nodes.gpufG.disableGraphic(1);
                }
            }
        }
    }, {
        key: "readPix",
        value: function readPix() {
            var arrayPick = new Uint8Array(4);
            var mousePos = this._sce.getEvents().getMousePosition();
            this._gl.readPixels(mousePos.x, this._sce.getCanvas().height - mousePos.y, 1, 1, this._gl.RGBA, this._gl.UNSIGNED_BYTE, arrayPick);

            var unpackValue = this._utils.unpack([arrayPick[0] / 255, arrayPick[1] / 255, arrayPick[2] / 255, arrayPick[3] / 255]); // value from 0.0 to 1.0
            this.selectedId = Math.round(unpackValue * 1000000.0) - 1.0;
            //console.log("hoverId: "+this.selectedId);
            if (this.selectedId !== -1 && this.selectedId < this.currentNodeId) {
                var node = this._nodesById[this.selectedId];
                if (node !== undefined && node !== null && node.onmousedown !== undefined && node.onmousedown !== null) node.onmousedown(node);

                if (node !== undefined && node !== null && this._onClickNode !== undefined && this._onClickNode !== null) this._onClickNode(node);

                var arr4Uint8_XYZW = this.comp_renderer_nodes.gpufG.readArg("posXYZW");
                var x = arr4Uint8_XYZW[this._nodesById[this.selectedId].itemStart * 4];
                var y = arr4Uint8_XYZW[this._nodesById[this.selectedId].itemStart * 4 + 1];
                var z = arr4Uint8_XYZW[this._nodesById[this.selectedId].itemStart * 4 + 2];
                this._initialPosDrag = $V3([x, y, z]);

                this.makeDrag(null, $V3([0.0, 0.0, 0.0]));
            }
        }
    }, {
        key: "makeDrag",


        /**
         * @param {MouseEvent} [evt]
         * @param {StormV3} dir
         */
        value: function makeDrag(evt, dir) {
            var comp_controller_trans_target = this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.CONTROLLER_TRANSFORM_TARGET);

            if (this._enableHover === false) {
                if (comp_controller_trans_target.isLeftBtnActive() === true) {
                    if (this.selectedId === -1) this.disableDrag();else {
                        this.enableDrag(this.selectedId, dir);
                        console.log("selectedId: " + this.selectedId);
                    }
                }
            } else {
                if (comp_controller_trans_target.isLeftBtnActive() === true) {
                    this.enableHov(-1);

                    if (this.selectedId === -1) this.disableDrag();else {
                        this.enableDrag(this.selectedId, dir);
                        console.log("selectedId: " + this.selectedId);
                    }
                } else {
                    this.enableHov(this.selectedId);
                }
            }
        }
    }, {
        key: "enableHov",


        /**
         * @param {int} selectedId
         */
        value: function enableHov(selectedId) {
            this.comp_renderer_nodes.setArg("idToHover", function () {
                return selectedId;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("idToHover", function () {
                    return selectedId;
                });
            }for (var _na5 = 0; _na5 < this.arrowsObj.length; _na5++) {
                this.arrowsObj[_na5].componentRenderer.setArg("idToHover", function () {
                    return selectedId;
                });
            }if (this._enableFont === true) {
                this.comp_renderer_nodesText.setArg("idToHover", function () {
                    return selectedId;
                });
            }
        }
    }, {
        key: "enableDrag",


        /**
         * @param {int} selectedId
         * @param {StormV3} dir
         */
        value: function enableDrag(selectedId, dir) {
            var _this3 = this;

            var comp_projection = this._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.PROJECTION);
            var finalPos = this._initialPosDrag.add(dir.x(comp_projection.getFov() * 2.0 / this._sce.getCanvas().width));

            this.comp_renderer_nodes.setArg("enableDrag", function () {
                return 1;
            });
            this.comp_renderer_nodes.setArg("idToDrag", function () {
                return selectedId;
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationX", function () {
                return finalPos.e[0];
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationY", function () {
                return finalPos.e[1];
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationZ", function () {
                return finalPos.e[2];
            });
            this.comp_renderer_nodes.setArg("initialPosX", function () {
                return _this3._initialPosDrag.e[0];
            });
            this.comp_renderer_nodes.setArg("initialPosY", function () {
                return _this3._initialPosDrag.e[1];
            });
            this.comp_renderer_nodes.setArg("initialPosZ", function () {
                return _this3._initialPosDrag.e[2];
            });

            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("enableDrag", function () {
                    return 1;
                });
                this.linksObj[na].componentRenderer.setArg("idToDrag", function () {
                    return selectedId;
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationX", function () {
                    return finalPos.e[0];
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationY", function () {
                    return finalPos.e[1];
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationZ", function () {
                    return finalPos.e[2];
                });
                this.linksObj[na].componentRenderer.setArg("initialPosX", function () {
                    return _this3._initialPosDrag.e[0];
                });
                this.linksObj[na].componentRenderer.setArg("initialPosY", function () {
                    return _this3._initialPosDrag.e[1];
                });
                this.linksObj[na].componentRenderer.setArg("initialPosZ", function () {
                    return _this3._initialPosDrag.e[2];
                });
            }

            for (var _na6 = 0; _na6 < this.arrowsObj.length; _na6++) {
                this.arrowsObj[_na6].componentRenderer.setArg("enableDrag", function () {
                    return 1;
                });
                this.arrowsObj[_na6].componentRenderer.setArg("idToDrag", function () {
                    return selectedId;
                });
                this.arrowsObj[_na6].componentRenderer.setArg("MouseDragTranslationX", function () {
                    return finalPos.e[0];
                });
                this.arrowsObj[_na6].componentRenderer.setArg("MouseDragTranslationY", function () {
                    return finalPos.e[1];
                });
                this.arrowsObj[_na6].componentRenderer.setArg("MouseDragTranslationZ", function () {
                    return finalPos.e[2];
                });
                this.arrowsObj[_na6].componentRenderer.setArg("initialPosX", function () {
                    return _this3._initialPosDrag.e[0];
                });
                this.arrowsObj[_na6].componentRenderer.setArg("initialPosY", function () {
                    return _this3._initialPosDrag.e[1];
                });
                this.arrowsObj[_na6].componentRenderer.setArg("initialPosZ", function () {
                    return _this3._initialPosDrag.e[2];
                });
            }

            if (this._enableFont === true) {
                this.comp_renderer_nodesText.setArg("enableDrag", function () {
                    return 1;
                });
                this.comp_renderer_nodesText.setArg("idToDrag", function () {
                    return selectedId;
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationX", function () {
                    return finalPos.e[0];
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationY", function () {
                    return finalPos.e[1];
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationZ", function () {
                    return finalPos.e[2];
                });
                this.comp_renderer_nodesText.setArg("initialPosX", function () {
                    return _this3._initialPosDrag.e[0];
                });
                this.comp_renderer_nodesText.setArg("initialPosY", function () {
                    return _this3._initialPosDrag.e[1];
                });
                this.comp_renderer_nodesText.setArg("initialPosZ", function () {
                    return _this3._initialPosDrag.e[2];
                });
            }
        }
    }, {
        key: "disableDrag",


        /**
         * disableDrag
         */
        value: function disableDrag() {
            this.comp_renderer_nodes.setArg("enableDrag", function () {
                return 0;
            });
            this.comp_renderer_nodes.setArg("idToDrag", function () {
                return 0;
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationX", function () {
                return 0;
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationY", function () {
                return 0;
            });
            this.comp_renderer_nodes.setArg("MouseDragTranslationZ", function () {
                return 0;
            });

            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("enableDrag", function () {
                    return 0;
                });
                this.linksObj[na].componentRenderer.setArg("idToDrag", function () {
                    return 0;
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationX", function () {
                    return 0;
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationY", function () {
                    return 0;
                });
                this.linksObj[na].componentRenderer.setArg("MouseDragTranslationZ", function () {
                    return 0;
                });
            }

            for (var _na7 = 0; _na7 < this.arrowsObj.length; _na7++) {
                this.arrowsObj[_na7].componentRenderer.setArg("enableDrag", function () {
                    return 0;
                });
                this.arrowsObj[_na7].componentRenderer.setArg("idToDrag", function () {
                    return 0;
                });
                this.arrowsObj[_na7].componentRenderer.setArg("MouseDragTranslationX", function () {
                    return 0;
                });
                this.arrowsObj[_na7].componentRenderer.setArg("MouseDragTranslationY", function () {
                    return 0;
                });
                this.arrowsObj[_na7].componentRenderer.setArg("MouseDragTranslationZ", function () {
                    return 0;
                });
            }

            if (this._enableFont === true) {
                this.comp_renderer_nodesText.setArg("enableDrag", function () {
                    return 0;
                });
                this.comp_renderer_nodesText.setArg("idToDrag", function () {
                    return 0;
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationX", function () {
                    return 0;
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationY", function () {
                    return 0;
                });
                this.comp_renderer_nodesText.setArg("MouseDragTranslationZ", function () {
                    return 0;
                });
            }
        }
    }, {
        key: "enableNeuronalNetwork",
        value: function enableNeuronalNetwork() {
            this.applyLayout({
                // DIRECTION
                "argsDirection":
                // destination
                "float4* dest,float* enableDestination",
                "codeDirection":
                // destination
                "if(enableDestination[x] == 1.0) {\n                    vec3 destinationPos = dest[x].xyz;\n                    vec3 dirDestination = normalize(destinationPos-currentPos);\n                    float distan = abs(distance(currentPos,destinationPos));\n                    float dirDestWeight = sqrt(distan);\n                    currentDir = (currentDir+(dirDestination*dirDestWeight))*dirDestWeight*0.1;\n                }",

                // OBJECT
                "argsObject":
                // nodeColor
                "float4*attr nodeColor",
                "codeObject":
                // nodeColor
                //'if(isNode == 1.0) nodeVertexColor = nodeColor[x];'+
                //'if(isLink == 1.0 && currentLineVertex == 1.0) nodeVertexColor = vec4(0.0, 1.0, 0.0, 1.0);'+ // this is isTarget for arrows

                //'float degr = (currentLineVertex/vertexCount)/2.0;'+
                //'if(isLink == 1.0) nodeVertexColor = vec4(0.3, 0.2, 0.2, 1.0);'+ // this is isTarget for arrows
                'if(isArrow == 1.0 && currentLineVertex == 1.0) nodeVertexColor = vec4(0.3, 0.2, 0.2, 1.0);' + // this is isTarget for arrows
                'if(isArrow == 1.0 && currentLineVertex == 0.0) nodeVertexColor = vec4(1.0, 0.0, 0.0, 0.0);' // this is isTarget for arrows

            });
        }
    }, {
        key: "addNeuron",


        /**
         * @param {String} neuronName
         * @param {Array<number>} [destination=[0.0, 0.0, 0.0, 1.0]]
         * @param {int} [biasNeuron]
         */
        value: function addNeuron(neuronName, destination, biasNeuron) {
            var offs = 1000;
            var pos = [-(offs / 2) + Math.random() * offs, -(offs / 2) + Math.random() * offs, -(offs / 2) + Math.random() * offs, 1.0];
            var dest = destination !== undefined && destination !== null ? destination : [0.0, 0.0, 0.0, 1.0];
            var enableDest = destination !== undefined && destination !== null ? 1.0 : 0.0;

            this.addNode({
                "name": neuronName,
                "data": neuronName,
                "label": neuronName.toString(),
                "position": pos,
                "color": _resources.Resources.imgWhite(),
                "layoutNodeArgumentData": {
                    "nodeColor": [1.0, 1.0, 1.0, 1.0],
                    "enableDestination": enableDest,
                    "dest": dest
                },
                "onmouseup": function onmouseup(nodeData) {},
                "biasNeuron": biasNeuron !== undefined && biasNeuron !== null ? biasNeuron : 0.0 });
        }
    }, {
        key: "addEfferentNeuron",


        /**
         * @param {String} neuronName
         * @param {Array<number>} [destination=[0.0, 0.0, 0.0, 1.0]]
         */
        value: function addEfferentNeuron(neuronName, destination) {
            this.efferentNodesCount++;
            this.efferentNeuron.push(neuronName);
            this.addNeuron(neuronName, destination);
        }
    }, {
        key: "createNeuronLayer",


        /**
         * @param {int} numX
         * @param {int} numY
         * @param {Array<number>} pos
         * @param {number} nodSep
         * @param {int} [hasBias]
         * @param {int} [isInput]
         */
        value: function createNeuronLayer(numX, numY, pos, nodSep, hasBias, isInput) {
            var arr = [];
            for (var x = 0; x < numX; x++) {
                for (var y = 0; y < numY; y++) {
                    var position = [pos[0] + (x - numX / 2) * nodSep, pos[1], pos[2] + (y - numY / 2) * nodSep, pos[3]];

                    if (isInput !== undefined && isInput !== null && isInput === 1) {
                        var name = numX === 1 ? "I" + this.currentNeuron.toString() : this.layerCount.toString() + x + "_" + y;
                        this.addNeuron(name, position);
                        arr.push(name);
                        this.currentNeuron++;
                        this.afferentNodesCount++;
                    } else {
                        var _name = "H" + this.currentNeuron.toString();
                        this.addNeuron(_name, position);
                        arr.push(_name);
                        this.currentNeuron++;
                    }
                }
            }

            // bias neuron
            if (hasBias === 1.0) {
                var _position = [pos[0], pos[1] - 10.0, pos[2] + (numY + 3 - numY / 2) * nodSep, pos[3]];

                var _name2 = "B" + this.currentNeuron.toString();
                this.addNeuron(_name2, _position, hasBias);
                arr.push(_name2);
                this.currentNeuron++;
            }

            return arr;
        }
    }, {
        key: "connectConvXYNeuronsFromXYNeurons",


        /**
         * @param {Object} jsonIn
         * @param {int} jsonIn.w
         * @param {Array<int>} jsonIn.neuronLayerOrigin
         * @param {Array<int>} jsonIn.neuronLayerTarget
         * @param {number|null|Array<number>} [jsonIn.weight]
         * @param {int} [jsonIn.layer_neurons_count]
         * @param {number} [jsonIn.multiplier=1.0]
         * @param {int} jsonIn.layerNum
         * @param {int} jsonIn.hasBias
         * @param {int} jsonIn.convMatrixId
         */
        value: function connectConvXYNeuronsFromXYNeurons(jsonIn) {
            var convMatrix = {};
            convMatrix[0] = [// emboss
            -2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
            convMatrix[1] = [// bottom sobel
            -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
            convMatrix[2] = [// left sobel
            1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
            convMatrix[3] = [// outline
            -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
            convMatrix[4] = [// right sobel
            -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
            convMatrix[5] = [// sharpen
            0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

            var idds = [{ x: -1, y: -1 }, { x: 0, y: -1 }, { x: 1, y: -1 }, { x: -1, y: 0 }, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: -1, y: 1 }, { x: 0, y: 1 }, { x: 1, y: 1 }];

            var arr = [];
            var xT = 0;
            var yT = 0;
            var xO = 1;
            var yO = 1;
            for (var n = 0; n < (jsonIn.hasBias === 1.0 ? jsonIn.neuronLayerTarget.length - 1 : jsonIn.neuronLayerTarget.length); n++) {
                if (xT === jsonIn.w) {
                    xT = 0;
                    yT++;
                    xO = 1;
                    yO++;
                } else {
                    xT++;
                    xO++;
                }

                //let idT = (yT*jsonIn.w)+xT;

                var idConvM = 0;
                for (var nb = 0; nb < idds.length; nb++) {
                    var xOf = xO + idds[nb].x;
                    var yOf = yO + idds[nb].y;

                    var idO = yOf * (jsonIn.w + 2) + xOf;

                    this.addSinapsis({ "neuronNameA": jsonIn.neuronLayerOrigin[idO].toString(),
                        "neuronNameB": jsonIn.neuronLayerTarget[n].toString(),
                        "activationFunc": jsonIn.activationFunc,
                        "weight": jsonIn.weight !== undefined && jsonIn.weight !== null && jsonIn.weight.constructor === Array ? jsonIn.weight[n] : jsonIn.weight,
                        "layer_neurons_count": jsonIn.layer_neurons_count,
                        "multiplier": convMatrix[jsonIn.convMatrixId][nb] + 0.0001,
                        "layerNum": jsonIn.layerNum });
                    idConvM++;
                }
                if (jsonIn.hasBias === 1.0) {
                    this.addSinapsis({ "neuronNameA": jsonIn.neuronLayerOrigin[jsonIn.neuronLayerOrigin.length - 1].toString(),
                        "neuronNameB": jsonIn.neuronLayerTarget[n].toString(),
                        "activationFunc": jsonIn.activationFunc,
                        "weight": jsonIn.weight !== undefined && jsonIn.weight !== null && jsonIn.weight.constructor === Array ? jsonIn.weight[n] : jsonIn.weight,
                        "layer_neurons_count": jsonIn.layer_neurons_count,
                        "multiplier": jsonIn.multiplier,
                        "layerNum": jsonIn.layerNum });
                }
            }

            return arr;
        }
    }, {
        key: "addSinapsis",


        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.neuronNameA
         * @param {String} jsonIn.neuronNameB
         * @param {int} [jsonIn.activationFunc=1.0] 1.0=use weight*data;0.0=use multiplier*data
         * @param {number} [jsonIn.weight]
         * @param {int} [jsonIn.layer_neurons_count]
         * @param {number} [jsonIn.multiplier=1.0]
         * @param {int} jsonIn.layerNum
         */
        value: function addSinapsis(jsonIn) {
            var _this4 = this;

            var gaussRandom = function gaussRandom() {
                if (_this4.return_v === true) {
                    _this4.return_v = false;
                    return _this4.v_val;
                }
                var u = 2 * Math.random() - 1;
                var v = 2 * Math.random() - 1;
                var r = u * u + v * v;
                if (r === 0 || r > 1) return gaussRandom();
                var c = Math.sqrt(-2 * Math.log(r) / r);
                _this4.v_val = v * c; // cache this
                _this4.return_v = true;
                return u * c;
            };
            var randn = function randn(mu, std) {
                return mu + gaussRandom() * std;
            };
            var scale = jsonIn.layer_neurons_count !== undefined && jsonIn.layer_neurons_count !== null ? 0.14 - Math.sqrt(2.0 / jsonIn.layer_neurons_count) : Math.sqrt(2.0 / 50);

            var _activationFunc = jsonIn.activationFunc !== undefined && jsonIn.activationFunc !== null ? jsonIn.activationFunc : 1.0;

            var _weight = jsonIn.weight !== undefined && jsonIn.weight !== null ? jsonIn.weight : null;
            if (this._nodesByName[jsonIn.neuronNameA].biasNeuron === 1.0) {
                if (_weight === null) _weight = 0.01;
            } else {
                if (_weight === null) _weight = randn(0.0, scale);
            }
            var _linkMultiplier = jsonIn.multiplier !== undefined && jsonIn.multiplier !== null ? jsonIn.multiplier : 1.0;

            this.addLink({
                "origin": jsonIn.neuronNameA,
                "target": jsonIn.neuronNameB,
                "directed": true,
                "showArrow": false,
                "activationFunc": _activationFunc,
                "weight": _weight,
                "linkMultiplier": _linkMultiplier,
                "layerNum": jsonIn.layerNum });
        }
    }, {
        key: "connectNeuronWithNeuronLayer",


        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.neuron
         * @param {Array<int>} jsonIn.neuronLayer
         * @param {int} [jsonIn.activationFunc=1.0] 1.0=use weight*data;0.0=use multiplier*data
         * @param {number|null|Array<number>} [jsonIn.weight]
         * @param {int} [jsonIn.layer_neurons_count]
         * @param {number} [jsonIn.multiplier=1.0]
         * @param {int} jsonIn.layerNum
         * @param {int} jsonIn.hasBias
         */
        value: function connectNeuronWithNeuronLayer(jsonIn) {
            for (var n = 0; n < (jsonIn.hasBias === 1.0 ? jsonIn.neuronLayer.length - 1 : jsonIn.neuronLayer.length); n++) {
                this.addSinapsis({ "neuronNameA": jsonIn.neuron.toString(),
                    "neuronNameB": jsonIn.neuronLayer[n].toString(),
                    "activationFunc": jsonIn.activationFunc,
                    "weight": jsonIn.weight !== undefined && jsonIn.weight !== null && jsonIn.weight.constructor === Array ? jsonIn.weight[n] : jsonIn.weight,
                    "layer_neurons_count": jsonIn.layer_neurons_count,
                    "multiplier": jsonIn.multiplier,
                    "layerNum": jsonIn.layerNum });
            }
        }
    }, {
        key: "connectNeuronLayerWithNeuron",


        /**
         * @param {Object} jsonIn
         * @param {Array<int>} jsonIn.neuronLayer
         * @param {number|null|Array<number>} [jsonIn.weight]
         * @param {int} [jsonIn.layer_neurons_count]
         * @param {String} jsonIn.neuron
         * @param {int} jsonIn.layerNum
         */
        value: function connectNeuronLayerWithNeuron(jsonIn) {
            for (var n = 0; n < jsonIn.neuronLayer.length; n++) {
                this.addSinapsis({ "neuronNameA": jsonIn.neuronLayer[n].toString(),
                    "neuronNameB": jsonIn.neuron,
                    "activationFunc": 1.0,
                    "weight": jsonIn.weight !== undefined && jsonIn.weight !== null && jsonIn.weight.constructor === Array ? jsonIn.weight[n] : jsonIn.weight,
                    "layer_neurons_count": jsonIn.layer_neurons_count,
                    "layerNum": jsonIn.layerNum });
            }
        }
    }, {
        key: "connectNeuronLayerWithNeuronLayer",


        /**
         * @param {Object} jsonIn
         * @param {Array<int>} jsonIn.neuronLayerOrigin
         * @param {Array<int>} jsonIn.neuronLayerTarget
         * @param {Array<number>} jsonIn.weights
         * @param {int} [jsonIn.layer_neurons_count]
         * @param {int} jsonIn.layerNum
         * @param {int} jsonIn.hasBias
         */
        value: function connectNeuronLayerWithNeuronLayer(jsonIn) {
            var we = jsonIn.weights;

            for (var n = 0; n < jsonIn.neuronLayerOrigin.length; n++) {
                var neuronOrigin = jsonIn.neuronLayerOrigin[n];
                this.connectNeuronWithNeuronLayer({ "neuron": neuronOrigin.toString(),
                    "neuronLayer": jsonIn.neuronLayerTarget,
                    "layerNum": jsonIn.layerNum,
                    "hasBias": jsonIn.hasBias,
                    "weight": jsonIn.weights !== undefined && jsonIn.weights !== null ? we.slice(0, jsonIn.neuronLayerTarget.length - jsonIn.hasBias) : null,
                    "layer_neurons_count": jsonIn.layer_neurons_count });

                if (jsonIn.weights !== undefined && jsonIn.weights !== null) we = we.slice(jsonIn.neuronLayerTarget.length - jsonIn.hasBias);
            }
        }
    }, {
        key: "toJson",


        /**
         * @returns {String}
         */
        value: function toJson() {
            var adjMA = this.comp_renderer_nodes.gpufG.readArg("adjacencyMatrix");

            var outJson = {};
            outJson.layers = [];

            outJson.layers.push({ "out_depth": this.layer_defs[0].depth,
                "layer_type": "input" });
            var currStartN = this.layer_defs[0].depth + this.layer_defs[0].hasBias;

            for (var n = 1; n < this.layer_defs.length; n++) {
                var currHasBias = this.layer_defs[n].hasBias !== undefined && this.layer_defs[n].hasBias !== null ? this.layer_defs[n].hasBias : 0.0;
                var currLayerDepthU = this.layer_defs[n].depth !== undefined && this.layer_defs[n].depth !== null ? this.layer_defs[n].depth : this.layer_defs[n].num_neurons;
                var currLayerDepth = this.layer_defs[n].depth !== undefined && this.layer_defs[n].depth !== null ? this.layer_defs[n].depth + currHasBias : this.layer_defs[n].num_neurons + currHasBias;

                var lastHasBias = this.layer_defs[n - 1].hasBias !== undefined && this.layer_defs[n - 1].hasBias !== null ? this.layer_defs[n - 1].hasBias : 0.0;
                var lastLayerDepthU = this.layer_defs[n - 1].depth !== undefined && this.layer_defs[n - 1].depth !== null ? this.layer_defs[n - 1].depth : this.layer_defs[n - 1].num_neurons;
                var lastLayerDepth = this.layer_defs[n - 1].depth !== undefined && this.layer_defs[n - 1].depth !== null ? this.layer_defs[n - 1].depth + lastHasBias : this.layer_defs[n - 1].num_neurons + lastHasBias;

                outJson.layers.push({ "out_depth": this.layer_defs[n].num_neurons,
                    "layer_type": n === this.layer_defs.length - 1 ? "regression" : "fc",
                    "num_inputs": lastLayerDepthU,
                    "l1_decay_mul": 0,
                    "l2_decay_mul": 1,
                    "filters": [] });
                var lastL = outJson.layers[outJson.layers.length - 1];

                // filters minus bias
                for (var p = currStartN; p < currStartN + currLayerDepthU; p++) {
                    lastL.filters.push({
                        "depth": lastLayerDepthU,
                        "w": {},
                        "activation": "relu" });

                    var c_w = 0;
                    for (var c = currStartN - lastLayerDepth; c < currStartN - lastLayerDepth + lastLayerDepthU; c++) {
                        var pixelChild = this.getPixelChild(p, c) * 4;
                        lastL.filters[lastL.filters.length - 1].w[c_w] = adjMA[pixelChild + 2]; // c+" "+p
                        c_w++;
                    }
                }

                // bias
                if (this.layer_defs[n - 1].hasBias === 1.0) {
                    lastL.biases = {
                        "depth": this.layer_defs[n].num_neurons,
                        "w": {} };
                    var _c = currStartN - 1.0;
                    var _c_w = 0;
                    for (var _p = currStartN; _p < currStartN + currLayerDepthU; _p++) {
                        var _pixelChild = this.getPixelChild(_p, _c) * 4;
                        lastL.biases.w[_c_w] = adjMA[_pixelChild + 2]; // c+" "+p
                        _c_w++;
                    }
                }

                currStartN += currLayerDepth;
            }

            var str = JSON.stringify(outJson);
            console.log(str);

            return str;
        }
    }, {
        key: "forward",


        /**
         * @param {Object} jsonIn
         * @param {Array<number>} jsonIn.state
         * @param {boolean} [jsonIn.readOutput=true]
         * @param {Function} jsonIn.onAction
         */
        value: function forward(jsonIn) {
            var _this5 = this;

            this.onAction = jsonIn.onAction;
            var lett = ["A", "B", "C", "D", "E", "F", "G"];
            var loc = [["dataB", 2], ["dataF", 0], ["dataF", 2], ["dataG", 0], ["dataG", 2], ["dataH", 0], ["dataH", 2]];

            this.comp_renderer_nodes.setArg("freezeOutput", function () {
                return 0.0;
            });

            // reset reward
            var reward = [];
            for (var n = 0; n < this.afferentNodesCount * this.gpu_batch_size; n++) {
                reward[n] = 0.0;
            }for (var _n2 = 0; _n2 < this.gpu_batch_size; _n2++) {
                this.comp_renderer_nodes.setArg("efferentNodes" + lett[_n2], function () {
                    return reward.slice(0, _this5.efferentNodesCount);
                });
                reward = reward.slice(this.efferentNodesCount);
            }

            this.maxacts = [];
            var state = jsonIn.state.slice(0);

            var _loop = function _loop(r) {
                var stateSet = state.slice(0, _this5.afferentNodesCount * _this5.gpu_batch_size);

                for (var _n3 = stateSet.length; _n3 < _this5.afferentNodesCount * _this5.gpu_batch_size; _n3++) {
                    stateSet[_n3] = 0.0;
                }for (var _n4 = 0; _n4 < _this5.gpu_batch_size; _n4++) {
                    _this5.comp_renderer_nodes.setArg("afferentNodes" + lett[_n4], function () {
                        return stateSet.slice(0, _this5.afferentNodesCount);
                    });
                    stateSet = stateSet.slice(_this5.afferentNodesCount);
                }

                for (var _n5 = 0; _n5 < _this5.layerCount; _n5++) {
                    _this5.comp_renderer_nodes.gpufG.processKernel(_this5.comp_renderer_nodes.gpufG.kernels[0], true, true);
                }_this5._sce.getLoadedProject().getActiveStage().tick();

                //if(jsonIn.readOutput === undefined || jsonIn.readOutput === null || (jsonIn.readOutput !== null && jsonIn.readOutput === true)) {
                var o = [[]];
                var currO = 0;
                for (var _n6 = 0; _n6 < _this5.efferentNodesCount * _this5.gpu_batch_size; _n6++) {
                    if (o[currO].length === _this5.efferentNodesCount) {
                        o.push([]);
                        currO++;
                    }

                    var u = _this5.getNeuronOutput(_this5.efferentNeuron[o[currO].length], loc[currO]);
                    if (isNaN(u[2]) === true) debugger;
                    o[currO].push({ "output": u[loc[currO][1]] });
                }

                for (var _n7 = 0; _n7 < o.length; _n7++) {
                    var maxk = 0;
                    var maxval = o[_n7][0].output;

                    var vals = [maxval];
                    var y = [0];
                    var outs = [0];
                    var sm = [0];
                    var ced = [0];
                    var smd = [0];

                    for (var nb = 1; nb < o[_n7].length; nb++) {
                        vals.push(o[_n7][nb].output);
                        y.push(0);
                        outs.push(0);
                        sm.push(0);
                        ced.push(0);
                        smd.push(0);

                        if (o[_n7][nb].output > maxval) {
                            maxk = nb;
                            maxval = o[_n7][nb].output;
                        }
                    }
                    _this5.maxacts.push({ "action": maxk, "value": maxval, "values": vals, "y": y, "o": outs, "sm": sm, "ced": ced, "smd": smd });
                }

                // softmax
                /*for(let n=0; n < this.maxacts.length; n++) {
                    let sm = 0.0;
                    for(let nb=0; nb < this.efferentNodesCount; nb++)
                        sm += Math.exp(this.maxacts[n].values[nb]);
                      for(let nb=0; nb < this.efferentNodesCount; nb++) {
                        this.maxacts[n].sm[nb] = Math.exp(this.maxacts[n].values[nb])/sm;
                          if(nb === this.maxacts[n].action)
                            this.maxacts[n].value = this.maxacts[n].sm[nb];
                    }
                }*/
                //}
            };

            for (var r = 0; r < this.batch_repeats; r++) {
                _loop(r);
            }
            if (this.onAction !== null) this.onAction(this.maxacts);
        }
    }, {
        key: "train",


        /**
         * @param {Object} jsonIn
         * @param {Object} jsonIn.reward
         * @param {Function} jsonIn.onTrained
         */
        value: function train(jsonIn) {
            var _this6 = this;

            this.onTrained = jsonIn.onTrained;
            var lett = ["A", "B", "C", "D", "E", "F", "G"];

            // softmax regression
            /*let cost = 0.0;
            for(let n=0; n < this.maxacts.length; n++) {
                for(let nb=0; nb < this.efferentNodesCount; nb++) {
                    //let rr = (jsonIn.reward[n].val >= 0) ? Math.exp(1.0) : Math.exp(-1);
                    this.maxacts[n].y[nb] = (jsonIn.reward[n] !== undefined && jsonIn.reward[n].dim === nb) ? jsonIn.reward[n].val/10 : 0.0;
                }
            }
              for(let n=0; n < this.maxacts.length; n++) {
                for(let nb=0; nb < this.efferentNodesCount; nb++) {
                    // cross-entropy cost
                    cost += -1.0*((this.maxacts[n].y[nb]*Math.log(this.maxacts[n].sm[nb])) + ((1.0-this.maxacts[n].y[nb])*Math.log(1.0-this.maxacts[n].sm[nb])));
                }
            }
              for(let n=0; n < this.maxacts.length; n++) {
                for(let nb=0; nb < this.efferentNodesCount; nb++) {
                    // cross-entropy derivative (CED)
                    this.maxacts[n].ced[nb] = ((this.maxacts[n].y[nb]*(1.0/this.maxacts[n].sm[nb])) + ((1.0-this.maxacts[n].y[nb])*(1.0/(1.0-this.maxacts[n].sm[nb]))));
                }
            }
              for(let n=0; n < this.maxacts.length; n++) {
                let sm = 0.0;
                for(let nb=0; nb < this.efferentNodesCount; nb++)
                    sm += Math.exp(this.maxacts[n].values[nb]);
                sm = sm*sm;
                  // softmax derivative
                for(let nb=0; nb < this.efferentNodesCount; nb++) {
                    let sumOpposites = 0.0;
                    for(let nc=0; nc < this.efferentNodesCount; nc++) {
                        if(nb !== nc)
                            sumOpposites += Math.exp(this.maxacts[n].values[nc]);
                    }
                    this.maxacts[n].smd[nb] = (Math.exp(this.maxacts[n].values[nb])*sumOpposites)/sm;
                }
            }
              let dd = [];
            for(let n=0; n < this.maxacts.length; n++) {
                for(let nb=0; nb < this.efferentNodesCount; nb++) {
                    let cc = this.maxacts[n].ced[nb]*this.maxacts[n].smd[nb];
                    dd.push(cc);
                }
            }*/

            // linear regression
            var cost = 0.0;
            for (var n = 0; n < this.maxacts.length; n++) {
                for (var nb = 0; nb < this.efferentNodesCount; nb++) {
                    if (jsonIn.reward[n] !== undefined && jsonIn.reward[n].dim === nb) {
                        this.maxacts[n].y[nb] = jsonIn.reward[n].val;
                        this.maxacts[n].o[nb] = -(this.maxacts[n].y[nb] - this.maxacts[n].values[nb]);

                        // MSE
                        cost += 0.5 * this.maxacts[n].o[nb] * this.maxacts[n].o[nb];
                    } else {
                        this.maxacts[n].y[nb] = 0.0;
                        this.maxacts[n].o[nb] = 0.0;
                    }
                }
            }

            this.comp_renderer_nodes.setArg("freezeOutput", function () {
                return 1.0;
            });
            this.comp_renderer_nodes.setArg("currentTrainLayer", function () {
                return -10.0;
            });

            this.comp_renderer_nodes.setArg("enableTrain", function () {
                return 1.0;
            });
            this.comp_renderer_nodes.gpufG.enableKernel(1);
            this.comp_renderer_nodes.setArg("updateTheta", function () {
                return 0.0;
            });
            this._sce.getLoadedProject().getActiveStage().tick();
            this._sce.getLoadedProject().getActiveStage().tick();
            this.comp_renderer_nodes.gpufG.disableKernel(1);
            this.comp_renderer_nodes.setArg("enableTrain", function () {
                return 0.0;
            });

            var cr = 0;

            var _loop2 = function _loop2(r) {
                var dd = [];
                for (var _n8 = 0; _n8 < _this6.gpu_batch_size; _n8++) {
                    for (var _nb = 0; _nb < _this6.efferentNodesCount; _nb++) {
                        var cc = _this6.maxacts[cr].o[_nb];
                        dd.push(cc);
                    }
                    cr++;
                }
                // send
                for (var _n9 = 0; _n9 < _this6.gpu_batch_size; _n9++) {
                    _this6.comp_renderer_nodes.setArg("efferentNodes" + lett[_n9], function () {
                        return dd.slice(0, _this6.efferentNodesCount);
                    });
                    dd = dd.slice(_this6.efferentNodesCount);
                }

                _this6.comp_renderer_nodes.gpufG.processKernel(_this6.comp_renderer_nodes.gpufG.kernels[0], true, true);

                var _loop3 = function _loop3(_n10) {
                    _this6.comp_renderer_nodes.setArg("currentTrainLayer", function () {
                        return _n10;
                    });

                    _this6.comp_renderer_nodes.gpufG.processKernel(_this6.comp_renderer_nodes.gpufG.kernels[0], true, true);

                    _this6.comp_renderer_nodes.setArg("enableTrain", function () {
                        return 1.0;
                    });
                    _this6.comp_renderer_nodes.gpufG.enableKernel(1);
                    _this6._sce.getLoadedProject().getActiveStage().tick();
                    _this6._sce.getLoadedProject().getActiveStage().tick();
                    //this.comp_renderer_nodes.tick();
                    //this.comp_renderer_nodes.gpufG.processKernel(this.comp_renderer_nodes.gpufG.kernels[1], true, true);
                    _this6.comp_renderer_nodes.gpufG.disableKernel(1);
                    _this6.comp_renderer_nodes.setArg("enableTrain", function () {
                        return 0.0;
                    });
                };

                for (var _n10 = _this6.layerCount - 2; _n10 >= 0; _n10--) {
                    _loop3(_n10);
                }
            };

            for (var r = 0; r < this.batch_repeats; r++) {
                _loop2(r);
            }

            if (this.onTrained !== null) this.onTrained(cost);
        }
    }, {
        key: "getNeuronOutput",
        value: function getNeuronOutput(neuronName, loc) {
            var o = this.comp_renderer_nodes.gpufG.readArg(loc[0]);

            var n = this._nodesByName[neuronName].itemStart * 4;
            return [o[n], o[n + 1], o[n + 2], o[n + 3]];
        }
    }, {
        key: "setLearningRate",
        value: function setLearningRate(v) {
            this.comp_renderer_nodes.setArg("learningRate", function () {
                return v;
            });
        }
    }, {
        key: "setLayoutArgumentData",


        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.argName
         * @param {number|Array<number>} jsonIn.value [number, number, number, number]
         */
        value: function setLayoutArgumentData(jsonIn) {
            this._customArgs[jsonIn.argName]["nodes_array_value"] = jsonIn.value;
            this._customArgs[jsonIn.argName]["links_array_value"] = jsonIn.value;
            this._customArgs[jsonIn.argName]["arrows_array_value"] = jsonIn.value;
            this._customArgs[jsonIn.argName]["nodestext_array_value"] = jsonIn.value;
            this.comp_renderer_nodes.setArg(jsonIn.argName, function () {
                return jsonIn.value;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg(jsonIn.argName, function () {
                    return jsonIn.value;
                });
            }for (var _na8 = 0; _na8 < this.arrowsObj.length; _na8++) {
                this.arrowsObj[_na8].componentRenderer.setArg(jsonIn.argName, function () {
                    return jsonIn.value;
                });
            }if (this._enableFont === true) this.comp_renderer_nodesText.setArg(jsonIn.argName, function () {
                return jsonIn.value;
            });
        }
    }, {
        key: "getLayoutNodeArgumentData",


        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.nodeName
         * @param {String} jsonIn.argName
         * @returns {number|Array<number>}
         */
        value: function getLayoutNodeArgumentData(jsonIn) {
            var node = this._nodesByName[jsonIn.nodeName];
            var expl = this._customArgs[jsonIn.argName].arg.split("*");
            var type = expl[0]; // float or float4

            for (var n = 0; n < this.arrayNodeData.length / 4; n++) {
                if (jsonIn.nodeName !== undefined && jsonIn.nodeName !== null && this.arrayNodeData[n * 4] === node.nodeId) {
                    var ca = this._customArgs[jsonIn.argName]["nodes_array_value"];
                    if (type === "float") {
                        var id = n;
                        if (ca[id] !== undefined && ca[id] !== null) return ca[id];
                    } else {
                        var _id = n * 4;
                        if (ca[_id] !== undefined && ca[_id] !== null) return [ca[_id], ca[_id + 1], ca[_id + 2], ca[_id + 3]];
                    }
                }
            }
        }
    }, {
        key: "setLayoutNodeArgumentData",


        /**
         * @param {Object} jsonIn
         * @param {String} [jsonIn.nodeName=undefined] - If undefined then value is setted in all nodes
         * @param {String} jsonIn.argName
         * @param {number|Array<number>} jsonIn.value [number, number, number, number]
         * @param {boolean} jsonIn.update
         */
        value: function setLayoutNodeArgumentData(jsonIn) {
            var _this7 = this;

            var node = this._nodesByName[jsonIn.nodeName];
            var expl = this._customArgs[jsonIn.argName].arg.split("*");
            var type = expl[0]; // float or float4

            /**
             * @param {String} type - "float" | "float4"
             * @param {String} argName -
             * @param {String} targetArray - "nodes_array_value" | "links_array_value" | "arrows_array_value" | "nodestext_array_value"
             * @param {int} n
             * @param {number} value
             */
            var setVal = function setVal(type, argName, targetArray, n, value) {
                if (type === "float") {
                    _this7._customArgs[argName][targetArray][n] = value;
                } else {
                    var id = n * 4;
                    _this7._customArgs[argName][targetArray][id] = value[0];
                    _this7._customArgs[argName][targetArray][id + 1] = value[1];
                    _this7._customArgs[argName][targetArray][id + 2] = value[2];
                    _this7._customArgs[argName][targetArray][id + 3] = value[3];
                }
            };

            // nodes id
            for (var n = 0; n < this.arrayNodeData.length / 4; n++) {
                if (jsonIn.nodeName === undefined || jsonIn.nodeName === null || jsonIn.nodeName !== undefined && jsonIn.nodeName !== null && this.arrayNodeData[n * 4] === node.nodeId) setVal(type, jsonIn.argName, "nodes_array_value", n, jsonIn.value);else {
                    var id = type === "float" ? n : n * 4;
                    if (this._customArgs[jsonIn.argName]["nodes_array_value"][id] === undefined && this._customArgs[jsonIn.argName]["nodes_array_value"][id] === null && jsonIn.update === false) {
                        if (type === "float") setVal(type, jsonIn.argName, "nodes_array_value", n, 0.0);else setVal(type, jsonIn.argName, "nodes_array_value", n, [0.0, 0.0, 0.0, 0.0]);
                    }
                }
            }
            this.comp_renderer_nodes.setArg(jsonIn.argName, function () {
                return _this7._customArgs[jsonIn.argName].nodes_array_value;
            });

            // link id
            for (var na = 0; na < this.linksObj.length; na++) {
                for (var _n11 = 0; _n11 < this.linksObj[na].arrayLinkData.length / 4; _n11++) {
                    if (jsonIn.nodeName === undefined || jsonIn.nodeName === null || jsonIn.nodeName !== undefined && jsonIn.nodeName !== null && this.linksObj[na].arrayLinkData[_n11 * 4] === node.nodeId) setVal(type, jsonIn.argName, "links_array_value", _n11, jsonIn.value);else {
                        var _id2 = type === "float" ? _n11 : _n11 * 4;
                        if (this._customArgs[jsonIn.argName]["links_array_value"][_id2] === undefined && this._customArgs[jsonIn.argName]["links_array_value"][_id2] === null && jsonIn.update === false) {
                            if (type === "float") setVal(type, jsonIn.argName, "links_array_value", _n11, 0.0);else setVal(type, jsonIn.argName, "links_array_value", _n11, [0.0, 0.0, 0.0, 0.0]);
                        }
                    }
                }
                this.linksObj[na].componentRenderer.setArg(jsonIn.argName, function () {
                    return _this7._customArgs[jsonIn.argName].links_array_value;
                });
            }

            // arrow id
            for (var _na9 = 0; _na9 < this.arrowsObj.length; _na9++) {
                for (var _n12 = 0; _n12 < this.arrowsObj[_na9].arrayArrowData.length / 4; _n12++) {
                    if (jsonIn.nodeName === undefined || jsonIn.nodeName === null || jsonIn.nodeName !== undefined && jsonIn.nodeName !== null && this.arrowsObj[_na9].arrayArrowData[_n12 * 4] === node.nodeId) setVal(type, jsonIn.argName, "arrows_array_value", _n12, jsonIn.value);else {
                        var _id3 = type === "float" ? _n12 : _n12 * 4;
                        if (this._customArgs[jsonIn.argName]["arrows_array_value"][_id3] === undefined && this._customArgs[jsonIn.argName]["arrows_array_value"][_id3] === null && jsonIn.update === false) {
                            if (type === "float") setVal(type, jsonIn.argName, "arrows_array_value", _n12, 0.0);else setVal(type, jsonIn.argName, "arrows_array_value", _n12, [0.0, 0.0, 0.0, 0.0]);
                        }
                    }
                }
                this.arrowsObj[_na9].componentRenderer.setArg(jsonIn.argName, function () {
                    return _this7._customArgs[jsonIn.argName].arrows_array_value;
                });
            }

            if (this._enableFont === true) {
                // nodeText id
                for (var _n13 = 0; _n13 < this.arrayNodeTextData.length / 4; _n13++) {
                    if (jsonIn.nodeName === undefined || jsonIn.nodeName === null || jsonIn.nodeName !== undefined && jsonIn.nodeName !== null && this.arrayNodeTextData[_n13 * 4] === node.nodeId) setVal(type, jsonIn.argName, "nodestext_array_value", _n13, jsonIn.value);else {
                        var _id4 = type === "float" ? _n13 : _n13 * 4;
                        if (this._customArgs[jsonIn.argName]["nodestext_array_value"][_id4] === undefined && this._customArgs[jsonIn.argName]["nodestext_array_value"][_id4] === null && jsonIn.update === false) {
                            if (type === "float") setVal(type, jsonIn.argName, "nodestext_array_value", _n13, 0.0);else setVal(type, jsonIn.argName, "nodestext_array_value", _n13, [0.0, 0.0, 0.0, 0.0]);
                        }
                    }
                }
                this.comp_renderer_nodesText.setArg(jsonIn.argName, function () {
                    return _this7._customArgs[jsonIn.argName].nodestext_array_value;
                });
            }
        }
    }, {
        key: "setLayoutNodeArgumentArrayData",


        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.argName
         * @param {Array<number>} jsonIn.value [number] or [number, number, number, number]
         */
        value: function setLayoutNodeArgumentArrayData(jsonIn) {
            var _this8 = this;

            var expl = this._customArgs[jsonIn.argName].arg.split("*");
            var type = expl[0]; // float or float4

            // nodes
            var currentId = -1;
            var x = 0,
                y = 0,
                z = 0,
                w = 0;
            this._customArgs[jsonIn.argName].nodes_array_value = [];
            for (var n = 0; n < this.arrayNodeData.length / 4; n++) {
                if (currentId !== this.arrayNodeData[n * 4]) {
                    currentId = this.arrayNodeData[n * 4];

                    if (type === "float") {
                        x = jsonIn.value[currentId];
                        this._customArgs[jsonIn.argName].nodes_array_value.push(x);
                    } else {
                        x = jsonIn.value[currentId * 4];
                        y = jsonIn.value[currentId * 4 + 1];
                        z = jsonIn.value[currentId * 4 + 2];
                        w = jsonIn.value[currentId * 4 + 3];
                        this._customArgs[jsonIn.argName].nodes_array_value.push(x, y, z, w);
                    }
                } else {
                    if (type === "float") this._customArgs[jsonIn.argName].nodes_array_value.push(x);else this._customArgs[jsonIn.argName].nodes_array_value.push(x, y, z, w);
                }
            }
            this.comp_renderer_nodes.setArg(jsonIn.argName, function () {
                return _this8._customArgs[jsonIn.argName].nodes_array_value;
            });

            // links
            this._customArgs[jsonIn.argName].links_array_value = [];
            for (var na = 0; na < this.linksObj.length; na++) {
                for (var _n14 = 0; _n14 < this.linksObj[na].arrayLinkNodeName.length; _n14++) {
                    var currentLinkNodeName = this.linksObj[na].arrayLinkNodeName[_n14];
                    var nodeNameItemStart = this._nodesByName[currentLinkNodeName].itemStart;

                    if (type === "float") {
                        this._customArgs[jsonIn.argName].links_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[nodeNameItemStart]);
                    } else {
                        this._customArgs[jsonIn.argName].links_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[nodeNameItemStart * 4], this._customArgs[jsonIn.argName].nodes_array_value[nodeNameItemStart * 4 + 1], this._customArgs[jsonIn.argName].nodes_array_value[nodeNameItemStart * 4 + 2], this._customArgs[jsonIn.argName].nodes_array_value[nodeNameItemStart * 4 + 3]);
                    }
                }
                this.linksObj[na].componentRenderer.setArg(jsonIn.argName, function () {
                    return _this8._customArgs[jsonIn.argName].links_array_value;
                });
            }

            // arrows
            this._customArgs[jsonIn.argName].arrows_array_value = [];
            for (var _na10 = 0; _na10 < this.arrowsObj.length; _na10++) {
                for (var _n15 = 0; _n15 < this.arrowsObj[_na10].arrayArrowNodeName.length; _n15++) {
                    var currentArrowNodeName = this.arrowsObj[_na10].arrayArrowNodeName[_n15];
                    var _nodeNameItemStart = this._nodesByName[currentArrowNodeName].itemStart;

                    if (type === "float") {
                        this._customArgs[jsonIn.argName].arrows_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart]);
                    } else {
                        this._customArgs[jsonIn.argName].arrows_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart * 4], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart * 4 + 1], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart * 4 + 2], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart * 4 + 3]);
                    }
                }
                this.arrowsObj[_na10].componentRenderer.setArg(jsonIn.argName, function () {
                    return _this8._customArgs[jsonIn.argName].arrows_array_value;
                });
            }

            // nodestext
            if (this._enableFont === true) {
                this._customArgs[jsonIn.argName].nodestext_array_value = [];
                for (var _n16 = 0; _n16 < this.arrayNodeTextNodeName.length; _n16++) {
                    var currentNodeTextNodeName = this.arrayNodeTextNodeName[_n16];
                    var _nodeNameItemStart2 = this._nodesByName[currentNodeTextNodeName].itemStart;

                    if (type === "float") {
                        this._customArgs[jsonIn.argName].nodestext_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart2]);
                    } else {
                        this._customArgs[jsonIn.argName].nodestext_array_value.push(this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart2 * 4], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart2 * 4 + 1], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart2 * 4 + 2], this._customArgs[jsonIn.argName].nodes_array_value[_nodeNameItemStart2 * 4 + 3]);
                    }
                }
                this.comp_renderer_nodesText.setArg(jsonIn.argName, function () {
                    return _this8._customArgs[jsonIn.argName].nodestext_array_value;
                });
            }
        }
    }, {
        key: "addNode",


        // █████╗ ██████╗ ██████╗     ███╗   ██╗ ██████╗ ██████╗ ███████╗
        //██╔══██╗██╔══██╗██╔══██╗    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝
        //███████║██║  ██║██║  ██║    ██╔██╗ ██║██║   ██║██║  ██║█████╗
        //██╔══██║██║  ██║██║  ██║    ██║╚██╗██║██║   ██║██║  ██║██╔══╝
        //██║  ██║██████╔╝██████╔╝    ██║ ╚████║╚██████╔╝██████╔╝███████╗
        //╚═╝  ╚═╝╚═════╝ ╚═════╝     ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝
        /**
        * @param {Object} jsonIn
        * @param {String} jsonIn.name - Name of node
           * @param {String} [jsonIn.label=undefined] - Label to show
        * @param {String} [jsonIn.data=""] - Custom data associated to this node
        * @param {Array<number>} [jsonIn.position=new Array(Math.Random(), Math.Random(), Math.Random(), 1.0)] - Position of node
        * @param {String} [jsonIn.color=undefined] - URL of image
        * @param {Object} [jsonIn.layoutNodeArgumentData=undefined] - Data for the custom layout
        * @param {Function} [jsonIn.onmousedown=undefined] - Event when mousedown
        * @param {Function} [jsonIn.onmouseup=undefined] - Event when mouseup
        * @param {int} [jsonIn.biasNeuron]
        * @returns {String|boolean} - Name of node
         */
        value: function addNode(jsonIn) {
            if (this._nodesByName.hasOwnProperty(jsonIn.name) === false) {
                var node = this.createNode(jsonIn);
                this._nodesByName[jsonIn.name] = node;
                this._nodesById[node.nodeId] = node;

                if (node.label !== undefined && node.label !== null && this._enableFont === true) this.createNodeText(node);

                console.log("%cnode " + Object.keys(this._nodesByName).length + " (" + jsonIn.name + ")", "color:green");

                return jsonIn.name;
            } else {
                console.log("node " + jsonIn.name + " already exists");
                return false;
            }
        }
    }, {
        key: "createNode",


        /**
         * @param {Object} jsonIn
         * @param {Array<number>} [jsonIn.position=[Math.Random(), Math.Random(), Math.Random(), 1.0]] - Position of node
         * @param {Object} [jsonIn.layoutNodeArgumentData=undefined]
         * @param {String} [jsonIn.color]
         * @param {int} [jsonIn.nodeId]
         * @param {int} [jsonIn.itemStart]
         * @param {int} [jsonIn.biasNeuron]
         * @returns {Object}
         */
        value: function createNode(jsonIn) {
            var nAIS = this.nodeArrayItemStart;

            var offs = 100.0;
            var pos = jsonIn.position !== undefined && jsonIn.position !== null ? jsonIn.position : [-(offs / 2) + Math.random() * offs, -(offs / 2) + Math.random() * offs, -(offs / 2) + Math.random() * offs, 1.0];

            var color = jsonIn.color;

            var nodeImgId = -1;
            if (color !== undefined && color !== null && color.constructor === String) {
                // color is string URL
                if (this.objNodeImages.hasOwnProperty(color) === false) {
                    var locationIdx = Object.keys(this.objNodeImages).length;
                    this.objNodeImages[color] = locationIdx;

                    this.addNodesImage(color, locationIdx);
                }
                nodeImgId = this.objNodeImages[color];
            }
            for (var n = 0; n < this.mesh_nodes.vertexArray.length / 4; n++) {
                var idxVertex = n * 4;

                this.arrayNodeData.push(this.currentNodeId, 0.0, 0.0, 0.0);
                this.arrayNodeDataB.push(jsonIn.biasNeuron, 0.0, 0.0, 0.0);
                this.arrayNodeDataF.push(0.0, 0.0, 0.0, 0.0);
                this.arrayNodeDataG.push(0.0, 0.0, 0.0, 0.0);
                this.arrayNodeDataH.push(0.0, 0.0, 0.0, 0.0);
                this.arrayNodePosXYZW.push(pos[0], pos[1], pos[2], pos[3]);
                this.arrayNodeDir.push(0, 0, 0, 1.0);
                this.arrayNodeVertexPos.push(this.mesh_nodes.vertexArray[idxVertex], this.mesh_nodes.vertexArray[idxVertex + 1], this.mesh_nodes.vertexArray[idxVertex + 2], 1.0);
                this.arrayNodeVertexNormal.push(this.mesh_nodes.normalArray[idxVertex], this.mesh_nodes.normalArray[idxVertex + 1], this.mesh_nodes.normalArray[idxVertex + 2], 1.0);
                this.arrayNodeVertexTexture.push(this.mesh_nodes.textureArray[idxVertex], this.mesh_nodes.textureArray[idxVertex + 1], this.mesh_nodes.textureArray[idxVertex + 2], 1.0);

                this.arrayNodeImgId.push(nodeImgId);

                if (jsonIn.layoutNodeArgumentData !== undefined && jsonIn.layoutNodeArgumentData !== null) {
                    for (var argNameKey in this._customArgs) {
                        var expl = this._customArgs[argNameKey].arg.split("*");
                        if (expl.length > 0) {
                            // argument is type buffer
                            if (jsonIn.layoutNodeArgumentData.hasOwnProperty(argNameKey) === true && jsonIn.layoutNodeArgumentData[argNameKey] !== undefined && jsonIn.layoutNodeArgumentData[argNameKey] !== null) {
                                if (expl[0] === "float") this._customArgs[argNameKey].nodes_array_value.push(jsonIn.layoutNodeArgumentData[argNameKey]);else if (expl[0] === "float4") this._customArgs[argNameKey].nodes_array_value.push(jsonIn.layoutNodeArgumentData[argNameKey][0], jsonIn.layoutNodeArgumentData[argNameKey][1], jsonIn.layoutNodeArgumentData[argNameKey][2], jsonIn.layoutNodeArgumentData[argNameKey][3]);
                            }
                        }
                    }
                }

                this.nodeArrayItemStart++;
            }

            var maxNodeIndexId = 0;
            for (var _n17 = 0; _n17 < this.mesh_nodes.indexArray.length; _n17++) {
                var idxIndex = _n17;

                this.arrayNodeIndices.push(this.startIndexId + this.mesh_nodes.indexArray[idxIndex]);

                if (this.mesh_nodes.indexArray[idxIndex] > maxNodeIndexId) maxNodeIndexId = this.mesh_nodes.indexArray[idxIndex];
            }
            this.startIndexId += maxNodeIndexId + 1;

            jsonIn.nodeId = this.currentNodeId++;
            jsonIn.itemStart = nAIS; // nodeArrayItemStart
            return jsonIn;
        }
    }, {
        key: "createNodeText",
        value: function createNodeText(jsonIn) {
            var getLetterId = function getLetterId(letter) {
                var obj = { "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                    "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13,
                    "#": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20,
                    "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26, " ": 27,
                    "0": 28, "1": 29, "2": 30, "3": 31, "4": 32, "5": 33, "6": 34,
                    "7": 35, "8": 36, "9": 37, "-": 38, ".": 39
                };
                return obj[letter];
            };

            for (var i = 0; i < this.nodesTextPlanes; i++) {
                var letterId = null;
                if (jsonIn.label !== undefined && jsonIn.label !== null && jsonIn.label[i] !== undefined && jsonIn.label[i] !== null) letterId = getLetterId(jsonIn.label[i].toUpperCase());
                if (letterId === null) letterId = getLetterId(" ");

                for (var n = 0; n < this.mesh_nodesText.vertexArray.length / 4; n++) {
                    var idxVertex = n * 4;

                    this.arrayNodeTextData.push(jsonIn.nodeId, i, 0.0, 0.0);
                    this.arrayNodeTextPosXYZW.push(0.0, 0.0, 0.0, 1.0);
                    this.arrayNodeTextVertexPos.push(this.mesh_nodesText.vertexArray[idxVertex] + i * 5, this.mesh_nodesText.vertexArray[idxVertex + 1], this.mesh_nodesText.vertexArray[idxVertex + 2], 1.0);
                    this.arrayNodeTextVertexNormal.push(this.mesh_nodesText.normalArray[idxVertex], this.mesh_nodesText.normalArray[idxVertex + 1], this.mesh_nodesText.normalArray[idxVertex + 2], 1.0);
                    this.arrayNodeTextVertexTexture.push(this.mesh_nodesText.textureArray[idxVertex], this.mesh_nodesText.textureArray[idxVertex + 1], this.mesh_nodesText.textureArray[idxVertex + 2], 1.0);

                    this.arrayNodeTextNodeName.push(jsonIn.name);

                    this.arrayNodeText_itemStart.push(jsonIn.itemStart);

                    this.arrayNodeTextLetterId.push(letterId);

                    if (jsonIn.layoutNodeArgumentData !== undefined && jsonIn.layoutNodeArgumentData !== null) {
                        for (var argNameKey in this._customArgs) {
                            var expl = this._customArgs[argNameKey].arg.split("*");
                            if (expl.length > 0) {
                                // argument is type buffer
                                if (jsonIn.layoutNodeArgumentData.hasOwnProperty(argNameKey) === true && jsonIn.layoutNodeArgumentData[argNameKey] !== undefined && jsonIn.layoutNodeArgumentData[argNameKey] !== null) {
                                    if (expl[0] === "float") this._customArgs[argNameKey].nodestext_array_value.push(jsonIn.layoutNodeArgumentData[argNameKey]);else if (expl[0] === "float4") this._customArgs[argNameKey].nodestext_array_value.push(jsonIn.layoutNodeArgumentData[argNameKey][0], jsonIn.layoutNodeArgumentData[argNameKey][1], jsonIn.layoutNodeArgumentData[argNameKey][2], jsonIn.layoutNodeArgumentData[argNameKey][3]);
                                }
                            }
                        }
                    }

                    this.nodeTextArrayItemStart++;
                }
            }
            var maxNodeIndexId = 0;
            for (var _i = 0; _i < this.nodesTextPlanes; _i++) {
                for (var _n18 = 0; _n18 < this.mesh_nodesText.indexArray.length; _n18++) {
                    var idxIndex = _n18;

                    var b = _i * 4; // 4 = indices length of quad (0, 1, 2, 0, 2, 3)
                    var ii = this.mesh_nodesText.indexArray[idxIndex] + b;

                    this.arrayNodeTextIndices.push(this.startIndexId_nodestext + ii);

                    if (ii > maxNodeIndexId) maxNodeIndexId = ii;
                }
            }
            this.startIndexId_nodestext += maxNodeIndexId + 1;

            this.currentNodeTextId++; // augment node id
        }
    }, {
        key: "addNodesImage",


        /**
         * @param {String} url
         * @param {int} locationIdx
         */
        value: function addNodesImage(url, locationIdx) {
            this._stackNodesImg.push({
                "url": url,
                "locationIdx": locationIdx });
        }
    }, {
        key: "generateNodesImage",
        value: function generateNodesImage() {
            var _this9 = this;

            if (this.nodesImgMaskLoaded === false) {
                this.nodesImgMask = new Image();
                this.nodesImgMask.onload = function () {
                    _this9.nodesImgMaskLoaded = true;
                    _this9.generateNodesImage();
                };
                this.nodesImgMask.src = _resources.Resources.nodesImgMask();
            } else if (this.nodesImgCrosshairLoaded === false) {
                this.nodesImgCrosshair = new Image();
                this.nodesImgCrosshair.onload = function () {
                    _this9.nodesImgCrosshairLoaded = true;
                    _this9.generateNodesImage();
                };
                this.nodesImgCrosshair.src = _resources.Resources.nodesImgCrosshair();
            } else {
                new _ProccessImg.ProccessImg({
                    "stackNodesImg": this._stackNodesImg,
                    "NODE_IMG_WIDTH": this.NODE_IMG_WIDTH,
                    "NODE_IMG_COLUMNS": this.NODE_IMG_COLUMNS,
                    "nodesImgMask": this.nodesImgMask,
                    "nodesImgCrosshair": this.nodesImgCrosshair,
                    "onend": function onend(jsonIn) {
                        _this9.comp_renderer_nodes.setArg("nodesImg", function () {
                            return jsonIn.nodesImg;
                        });
                        _this9.comp_renderer_nodes.setArg("nodesImgCrosshair", function () {
                            return jsonIn.nodesImgCrosshair;
                        });
                    } });
            }
        }
    }, {
        key: "addLink",


        // █████╗ ██████╗ ██████╗     ██╗     ██╗███╗   ██╗██╗  ██╗
        //██╔══██╗██╔══██╗██╔══██╗    ██║     ██║████╗  ██║██║ ██╔╝
        //███████║██║  ██║██║  ██║    ██║     ██║██╔██╗ ██║█████╔╝
        //██╔══██║██║  ██║██║  ██║    ██║     ██║██║╚██╗██║██╔═██╗
        //██║  ██║██████╔╝██████╔╝    ███████╗██║██║ ╚████║██║  ██╗
        //╚═╝  ╚═╝╚═════╝ ╚═════╝     ╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
        /**
         * @param {Object} jsonIn
         * @param {String} jsonIn.origin - NodeName Origin for this link
         * @param {String} jsonIn.target - NodeName Target for this link
         * @param {boolean} [jsonIn.directed=false] - Default false=bidir
         * @param {number} [jsonIn.activationFunc=1.0] 1.0=linkWeight*data; 0.0=multiplier*data
         * @param {number|String} [jsonIn.weight=1.0] - Float weight or "RANDOM"
         * @param {number} [jsonIn.linkMultiplier=1.0]
         * @param {number} [jsonIn.layerNum]
         * @param {boolean} [jsonIn.showArrow]
         * @param {String} [jsonIn.origin_nodeName] - NodeName Origin for this link
         * @param {String} [jsonIn.target_nodeName] - NodeName Target for this link
         * @param {int} [jsonIn.origin_nodeId]
         * @param {int} [jsonIn.target_nodeId]
         * @param {int} [jsonIn.origin_itemStart]
         * @param {int} [jsonIn.target_itemStart]
         * @param {Array<>} [jsonIn.origin_layoutNodeArgumentData]
         * @param {Array<>} [jsonIn.target_layoutNodeArgumentData]
         * @param {int} [jsonIn.repeatId]
         */
        value: function addLink(jsonIn) {
            var pass = true;

            if (this._nodesByName[jsonIn.origin] === undefined && this._nodesByName[jsonIn.origin] === null) {
                console.log("%clink " + jsonIn.origin + "->" + jsonIn.target + ". Node " + jsonIn.origin + " not exists", "color:red");
                pass = false;
            }

            if (this._nodesByName[jsonIn.target] === undefined && this._nodesByName[jsonIn.target] === null) {
                console.log("%clink " + jsonIn.origin + "->" + jsonIn.target + ". Node " + jsonIn.target + " not exists", "color:red");
                pass = false;
            }

            if (jsonIn.origin === jsonIn.target) {
                console.log("%cDiscarting autolink " + jsonIn.origin + "->" + jsonIn.target, "color:orange");
                pass = false;
            }

            if (pass === true) {
                console.log("%clink " + jsonIn.origin + "->" + jsonIn.target, "color:green");

                jsonIn.origin_nodeName = jsonIn.origin.toString();
                jsonIn.target_nodeName = jsonIn.target.toString();
                jsonIn.origin_nodeId = this._nodesByName[jsonIn.origin].nodeId;
                jsonIn.target_nodeId = this._nodesByName[jsonIn.target].nodeId;
                jsonIn.origin_itemStart = this._nodesByName[jsonIn.origin].itemStart;
                jsonIn.target_itemStart = this._nodesByName[jsonIn.target].itemStart;
                jsonIn.origin_layoutNodeArgumentData = this._nodesByName[jsonIn.origin].layoutNodeArgumentData;
                jsonIn.target_layoutNodeArgumentData = this._nodesByName[jsonIn.target].layoutNodeArgumentData;

                jsonIn.activationFunc = jsonIn.activationFunc !== undefined && jsonIn.activationFunc !== null ? jsonIn.activationFunc : 1.0;
                if (jsonIn.weight !== undefined && jsonIn.weight !== null) {
                    if (jsonIn.weight.constructor === String) jsonIn.weight = Math.random();
                } else jsonIn.weight = Math.random();
                jsonIn.linkMultiplier = jsonIn.linkMultiplier !== undefined && jsonIn.linkMultiplier !== null ? jsonIn.linkMultiplier : 1.0;

                var repeatId = 1;
                while (true) {
                    var exists = this._links.hasOwnProperty(jsonIn.origin + "->" + jsonIn.target + "_" + repeatId) || this._links.hasOwnProperty(jsonIn.target + "->" + jsonIn.origin + "_" + repeatId);
                    if (exists === true) {
                        repeatId++;
                    } else break;
                }
                jsonIn.repeatId = repeatId;

                jsonIn = this.createLink(jsonIn);

                if (jsonIn.directed !== undefined && jsonIn.directed !== null && jsonIn.directed === true) {
                    if (jsonIn.showArrow !== undefined && jsonIn.showArrow !== null && jsonIn.showArrow === false) {} else this.createArrow(jsonIn);
                }

                // ADD LINK TO ARRAY LINKS
                this._links[jsonIn.origin + "->" + jsonIn.target + "_" + repeatId] = jsonIn;
            }
        }
    }, {
        key: "createLink",


        /**
         * Create new link for the graph
         * @param {Object} jsonIn
         * @param {String} jsonIn.origin_nodeName
         * @param {String} jsonIn.target_nodeName
         * @param {int} jsonIn.origin_nodeId
         * @param {int} jsonIn.target_nodeId
         * @param {int} jsonIn.origin_itemStart
         * @param {int} jsonIn.target_itemStart
         * @param {Array<>} jsonIn.origin_layoutNodeArgumentData
         * @param {Array<>} jsonIn.target_layoutNodeArgumentData
         * @param {int} [jsonIn.linkId]
         * @param {boolean} [jsonIn.directed=false]
         * @param {number} [jsonIn.weight=1.0] - Float weight or "RANDOM"
         * @returns {Object}
         */
        value: function createLink(jsonIn) {
            if (this.currentLinkId % 60000 === 0) this.createLinksObjItem();

            for (var n = 0; n < this.lineVertexCount * 2; n++) {
                this.linksObj[this.currentLinksObjItem].arrayLinkData.push(jsonIn.origin_nodeId, jsonIn.target_nodeId, Math.ceil(n / 2), jsonIn.repeatId);

                if (Math.ceil(n / 2) !== this.lineVertexCount - 1) {
                    this.linksObj[this.currentLinksObjItem].arrayLinkNodeName.push(jsonIn.origin_nodeName);
                } else {
                    this.linksObj[this.currentLinksObjItem].arrayLinkNodeName.push(jsonIn.target_nodeName);
                }
                this.linksObj[this.currentLinksObjItem].arrayLinkPosXYZW.push(0.0, 0.0, 0.0, 1.0);
                this.linksObj[this.currentLinksObjItem].arrayLinkVertexPos.push(0.0, 0.0, 0.0, 1.0);

                if (jsonIn.origin_layoutNodeArgumentData !== undefined && jsonIn.origin_layoutNodeArgumentData !== null) {
                    for (var argNameKey in this._customArgs) {
                        var expl = this._customArgs[argNameKey].arg.split("*");
                        if (expl.length > 0) {
                            // argument is type buffer
                            if (jsonIn.origin_layoutNodeArgumentData.hasOwnProperty(argNameKey) === true && jsonIn.origin_layoutNodeArgumentData[argNameKey] !== undefined && jsonIn.origin_layoutNodeArgumentData[argNameKey] !== null) {
                                if (expl[0] === "float") this._customArgs[argNameKey].links_array_value.push(jsonIn.origin_layoutNodeArgumentData[argNameKey]);else if (expl[0] === "float4") this._customArgs[argNameKey].links_array_value.push(jsonIn.origin_layoutNodeArgumentData[argNameKey][0], jsonIn.origin_layoutNodeArgumentData[argNameKey][1], jsonIn.origin_layoutNodeArgumentData[argNameKey][2], jsonIn.origin_layoutNodeArgumentData[argNameKey][3]);
                            }
                        }
                    }
                }
            }

            for (var _n19 = 0; _n19 < this.lineVertexCount * 2; _n19++) {
                this.linksObj[this.currentLinksObjItem].arrayLinkIndices.push(this.linksObj[this.currentLinksObjItem].startIndexId_link++);
            }this.currentLinkId += 2; // augment link id

            jsonIn.linkId = this.currentLinkId - 2;
            return jsonIn;
        }
    }, {
        key: "createArrow",


        /**
         * Create new arrow for the graph
         * @param {Object} jsonIn
         * @param {int} jsonIn.origin_nodeName
         * @param {int} jsonIn.target_nodeName
         * @param {int} jsonIn.origin_nodeId
         * @param {int} jsonIn.target_nodeId
         * @param {int} jsonIn.origin_itemStart
         * @param {int} jsonIn.target_itemStart
         * @param {int} jsonIn.origin_layoutNodeArgumentData
         * @param {int} jsonIn.target_layoutNodeArgumentData
         * @param {boolean} [jsonIn.directed=false]
         * @param {Mesh} [jsonIn.node] - Node with the mesh for the node
         * @returns {int}
         */
        value: function createArrow(jsonIn) {
            if (this.currentArrowId % 20000 === 0) this.createArrowsObjItem();

            if (jsonIn !== undefined && jsonIn !== null && jsonIn.node !== undefined && jsonIn.node !== null) this.mesh_arrows = jsonIn.node;

            var oppositeId = 0;

            for (var _o = 0; _o < 2; _o++) {
                for (var n = 0; n < this.mesh_arrows.vertexArray.length / 4; n++) {
                    var idxVertex = n * 4;
                    if (_o === 0) oppositeId = this.arrowsObj[this.currentArrowsObjItem].arrowArrayItemStart;

                    this.arrowsObj[this.currentArrowsObjItem].arrayArrowPosXYZW.push(0.0, 0.0, 0.0, 1.0);
                    this.arrowsObj[this.currentArrowsObjItem].arrayArrowVertexPos.push(this.mesh_arrows.vertexArray[idxVertex], this.mesh_arrows.vertexArray[idxVertex + 1], this.mesh_arrows.vertexArray[idxVertex + 2], 1.0);
                    this.arrowsObj[this.currentArrowsObjItem].arrayArrowVertexNormal.push(this.mesh_arrows.normalArray[idxVertex], this.mesh_arrows.normalArray[idxVertex + 1], this.mesh_arrows.normalArray[idxVertex + 2], 1.0);
                    this.arrowsObj[this.currentArrowsObjItem].arrayArrowVertexTexture.push(this.mesh_arrows.textureArray[idxVertex], this.mesh_arrows.textureArray[idxVertex + 1], this.mesh_arrows.textureArray[idxVertex + 2], 1.0);
                    if (_o === 0) {
                        this.arrowsObj[this.currentArrowsObjItem].arrayArrowData.push(jsonIn.origin_nodeId, jsonIn.target_nodeId, 0.0, jsonIn.repeatId);
                        this.arrowsObj[this.currentArrowsObjItem].arrayArrowNodeName.push(jsonIn.origin_nodeName);
                        if (jsonIn.origin_layoutNodeArgumentData !== undefined && jsonIn.origin_layoutNodeArgumentData !== null) {
                            for (var argNameKey in this._customArgs) {
                                var expl = this._customArgs[argNameKey].arg.split("*");
                                if (expl.length > 0) {
                                    // argument is type buffer
                                    if (jsonIn.origin_layoutNodeArgumentData.hasOwnProperty(argNameKey) === true && jsonIn.origin_layoutNodeArgumentData[argNameKey] !== undefined && jsonIn.origin_layoutNodeArgumentData[argNameKey] !== null) {
                                        if (expl[0] === "float") this._customArgs[argNameKey].arrows_array_value.push(jsonIn.origin_layoutNodeArgumentData[argNameKey]);else if (expl[0] === "float4") this._customArgs[argNameKey].arrows_array_value.push(jsonIn.origin_layoutNodeArgumentData[argNameKey][0], jsonIn.origin_layoutNodeArgumentData[argNameKey][1], jsonIn.origin_layoutNodeArgumentData[argNameKey][2], jsonIn.origin_layoutNodeArgumentData[argNameKey][3]);
                                    }
                                }
                            }
                        }
                    } else {
                        this.arrowsObj[this.currentArrowsObjItem].arrayArrowData.push(jsonIn.target_nodeId, jsonIn.origin_nodeId, 1.0, jsonIn.repeatId);
                        this.arrowsObj[this.currentArrowsObjItem].arrayArrowNodeName.push(jsonIn.target_nodeName);
                        if (jsonIn.target_layoutNodeArgumentData !== undefined && jsonIn.target_layoutNodeArgumentData !== null) {
                            for (var _argNameKey in this._customArgs) {
                                var _expl = this._customArgs[_argNameKey].arg.split("*");
                                if (_expl.length > 0) {
                                    // argument is type buffer
                                    if (jsonIn.target_layoutNodeArgumentData.hasOwnProperty(_argNameKey) === true && jsonIn.target_layoutNodeArgumentData[_argNameKey] !== undefined && jsonIn.target_layoutNodeArgumentData[_argNameKey] !== null) {
                                        if (_expl[0] === "float") this._customArgs[_argNameKey].arrows_array_value.push(jsonIn.target_layoutNodeArgumentData[_argNameKey]);else if (_expl[0] === "float4") this._customArgs[_argNameKey].arrows_array_value.push(jsonIn.target_layoutNodeArgumentData[_argNameKey][0], jsonIn.target_layoutNodeArgumentData[_argNameKey][1], jsonIn.target_layoutNodeArgumentData[_argNameKey][2], jsonIn.target_layoutNodeArgumentData[_argNameKey][3]);
                                    }
                                }
                            }
                        }
                    }

                    this.arrowsObj[this.currentArrowsObjItem].arrowArrayItemStart++;
                }

                var maxArrowIndexId = 0;
                for (var _n20 = 0; _n20 < this.mesh_arrows.indexArray.length; _n20++) {
                    var idxIndex = _n20;

                    this.arrowsObj[this.currentArrowsObjItem].arrayArrowIndices.push(this.arrowsObj[this.currentArrowsObjItem].startIndexId_arrow + this.mesh_arrows.indexArray[idxIndex]);

                    if (this.mesh_arrows.indexArray[idxIndex] > maxArrowIndexId) {
                        maxArrowIndexId = this.mesh_arrows.indexArray[idxIndex];
                    }
                }
                this.arrowsObj[this.currentArrowsObjItem].startIndexId_arrow += maxArrowIndexId + 1;

                this.currentArrowId++; // augment arrow id
            }
        }
    }, {
        key: "updateNodes",


        //██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗
        //██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝
        //██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗      ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗
        //██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝      ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║
        //╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║
        // ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
        value: function updateNodes() {
            var _this10 = this;

            console.log(this.currentNodeId + " nodes");

            this._ADJ_MATRIX_WIDTH = this.currentNodeId;

            this.comp_renderer_nodes.setArg("adjacencyMatrix", function () {
                return new Float32Array(_this10._ADJ_MATRIX_WIDTH * _this10._ADJ_MATRIX_WIDTH * 4);
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.getComponentBufferArg("adjacencyMatrix", this.comp_renderer_nodes);
            }for (var _na11 = 0; _na11 < this.arrowsObj.length; _na11++) {
                this.arrowsObj[_na11].componentRenderer.getComponentBufferArg("adjacencyMatrix", this.comp_renderer_nodes);
            }this.comp_renderer_nodes.setArg("adjacencyMatrixB", function () {
                return new Float32Array(_this10._ADJ_MATRIX_WIDTH * _this10._ADJ_MATRIX_WIDTH * 4);
            });
            for (var _na12 = 0; _na12 < this.linksObj.length; _na12++) {
                this.linksObj[_na12].componentRenderer.getComponentBufferArg("adjacencyMatrixB", this.comp_renderer_nodes);
            }for (var _na13 = 0; _na13 < this.arrowsObj.length; _na13++) {
                this.arrowsObj[_na13].componentRenderer.getComponentBufferArg("adjacencyMatrixB", this.comp_renderer_nodes);
            }this.comp_renderer_nodes.setArg("widthAdjMatrix", function () {
                return _this10._ADJ_MATRIX_WIDTH;
            });
            for (var _na14 = 0; _na14 < this.linksObj.length; _na14++) {
                this.linksObj[_na14].componentRenderer.setArg("widthAdjMatrix", function () {
                    return _this10._ADJ_MATRIX_WIDTH;
                });
            }for (var _na15 = 0; _na15 < this.arrowsObj.length; _na15++) {
                this.arrowsObj[_na15].componentRenderer.setArg("widthAdjMatrix", function () {
                    return _this10._ADJ_MATRIX_WIDTH;
                });
            }this.comp_renderer_nodes.setArg("data", function () {
                return _this10.arrayNodeData;
            });
            this.comp_renderer_nodes.setArg("dataB", function () {
                return _this10.arrayNodeDataB;
            });
            this.comp_renderer_nodes.setArg("dataF", function () {
                return _this10.arrayNodeDataF;
            });
            this.comp_renderer_nodes.setArg("dataG", function () {
                return _this10.arrayNodeDataG;
            });
            this.comp_renderer_nodes.setArg("dataH", function () {
                return _this10.arrayNodeDataH;
            });

            if (this.comp_renderer_nodes.getBuffers()["posXYZW"] !== undefined && this.comp_renderer_nodes.getBuffers()["posXYZW"] !== null) this.arrayNodePosXYZW = Array.prototype.slice.call(this.comp_renderer_nodes.gpufG.readArg("posXYZW"));
            this.comp_renderer_nodes.setArg("posXYZW", function () {
                return _this10.arrayNodePosXYZW;
            });

            this.comp_renderer_nodes.setArg("nodeVertexPos", function () {
                return _this10.arrayNodeVertexPos;
            });
            this.comp_renderer_nodes.setArg("nodeVertexNormal", function () {
                return _this10.arrayNodeVertexNormal;
            });
            this.comp_renderer_nodes.setArg("nodeVertexTexture", function () {
                return _this10.arrayNodeVertexTexture;
            });

            this.comp_renderer_nodes.setArg("nodesCount", function () {
                return _this10.currentNodeId;
            });
            this.comp_renderer_nodes.setArg("nodeImgColumns", function () {
                return _this10.NODE_IMG_COLUMNS;
            });
            this.comp_renderer_nodes.setArg("nodeImgId", function () {
                return _this10.arrayNodeImgId;
            });
            this.comp_renderer_nodes.setArg("indices", function () {
                return _this10.arrayNodeIndices;
            });

            if (this.comp_renderer_nodes.getBuffers()["dir"] !== undefined && this.comp_renderer_nodes.getBuffers()["dir"] !== null) this.arrayNodeDir = Array.prototype.slice.call(this.comp_renderer_nodes.gpufG.readArg("dir"));
            this.comp_renderer_nodes.setArg("dir", function () {
                return _this10.arrayNodeDir;
            });

            this.comp_renderer_nodes.setArg("PMatrix", function () {
                return _this10._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e;
            });
            this.comp_renderer_nodes.setArgUpdatable("PMatrix", true);
            this.comp_renderer_nodes.setArg("cameraWMatrix", function () {
                return _this10._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e;
            });
            this.comp_renderer_nodes.setArgUpdatable("cameraWMatrix", true);
            this.comp_renderer_nodes.setArg("nodeWMatrix", function () {
                return _this10.nodes.getComponent(Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e;
            });
            this.comp_renderer_nodes.setArgUpdatable("nodeWMatrix", true);

            this.comp_renderer_nodes.setArg("currentTrainLayer", function () {
                return -10;
            });
            this.comp_renderer_nodes.setArg("layerCount", function () {
                return _this10.layerCount;
            });
            this.comp_renderer_nodes.setArg("learningRate", function () {
                return 0.01;
            });
            this.comp_renderer_nodes.setArg("viewNeuronDynamics", function () {
                return 0.0;
            });
            this.comp_renderer_nodes.setArg("batch_repeats", function () {
                return _this10.batch_repeats;
            });
            this.comp_renderer_nodes.setArg("freezeOutput", function () {
                return 0.0;
            });
            this.comp_renderer_nodes.setArg("enableTrain", function () {
                return _this10.enableTrain;
            });
            this.comp_renderer_nodes.setArg("updateTheta", function () {
                return _this10.updateTheta;
            });
            this.comp_renderer_nodes.setArg("afferentNodesCount", function () {
                return _this10.afferentNodesCount;
            });
            this.comp_renderer_nodes.setArg("efferentNodesCount", function () {
                return _this10.efferentNodesCount;
            });
            this.comp_renderer_nodes.setArg("efferentStart", function () {
                return _this10.currentNodeId - _this10.efferentNodesCount;
            });

            this.comp_renderer_nodes.setArg("afferentNodesA", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesA", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesB", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesB", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesC", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesC", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesD", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesD", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesE", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesE", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesF", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesF", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("afferentNodesG", function () {
                return new Float32Array(_this10.afferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("efferentNodesG", function () {
                return new Float32Array(_this10.efferentNodesCount);
            });
            this.comp_renderer_nodes.setArg("isNode", function () {
                return 1;
            });
            this.comp_renderer_nodes.setArg("bufferNodesWidth", function () {
                return _this10.comp_renderer_nodes.getBuffers()["posXYZW"].W;
            });

            var _loop4 = function _loop4(argNameKey) {
                var expl = _this10._customArgs[argNameKey].arg.split("*");
                if (expl.length > 0) {
                    // argument is type buffer
                    _this10.comp_renderer_nodes.setArg(argNameKey, function () {
                        return _this10._customArgs[argNameKey].nodes_array_value;
                    });
                }
            };

            for (var argNameKey in this._customArgs) {
                _loop4(argNameKey);
            }

            this.generateNodesImage();
            if (this._enableFont === true) this.updateNodesText();
        }
    }, {
        key: "updateNodesText",
        value: function updateNodesText() {
            var _this11 = this;

            this.comp_renderer_nodesText.setArg("data", function () {
                return _this11.arrayNodeTextData;
            });
            this.comp_renderer_nodesText.getComponentBufferArg("dataB", this.comp_renderer_nodes);
            this.comp_renderer_nodesText.getComponentBufferArg("posXYZW", this.comp_renderer_nodes);

            this.comp_renderer_nodesText.setArg("nodeVertexPos", function () {
                return _this11.arrayNodeTextVertexPos;
            });
            this.comp_renderer_nodesText.setArg("nodeVertexNormal", function () {
                return _this11.arrayNodeTextVertexNormal;
            });
            this.comp_renderer_nodesText.setArg("nodeVertexTexture", function () {
                return _this11.arrayNodeTextVertexTexture;
            });

            this.comp_renderer_nodesText.setArg("fontImgColumns", function () {
                return _this11.FONT_IMG_COLUMNS;
            });
            this.comp_renderer_nodesText.setArg("letterId", function () {
                return _this11.arrayNodeTextLetterId;
            });
            this.comp_renderer_nodesText.setArg("indices", function () {
                return _this11.arrayNodeTextIndices;
            });

            this.comp_renderer_nodesText.setArg("PMatrix", function () {
                return _this11._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e;
            });
            this.comp_renderer_nodesText.setArgUpdatable("PMatrix", true);
            this.comp_renderer_nodesText.setArg("cameraWMatrix", function () {
                return _this11._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e;
            });
            this.comp_renderer_nodesText.setArgUpdatable("cameraWMatrix", true);
            this.comp_renderer_nodesText.setArg("nodeWMatrix", function () {
                return _this11.nodes.getComponent(Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e;
            });
            this.comp_renderer_nodesText.setArgUpdatable("nodeWMatrix", true);

            this.comp_renderer_nodesText.setArg("isNodeText", function () {
                return 1;
            });
            this.comp_renderer_nodesText.setArg("bufferNodesWidth", function () {
                return _this11.comp_renderer_nodes.getBuffers()["posXYZW"].W;
            });
            this.comp_renderer_nodesText.setArg("bufferTextsWidth", function () {
                return _this11.comp_renderer_nodesText.getBuffers()["data"].W;
            });
            this.comp_renderer_nodesText.setArg("showValues", function () {
                return 1.0;
            });

            var _loop5 = function _loop5(argNameKey) {
                var expl = _this11._customArgs[argNameKey].arg.split("*");
                if (expl.length > 0) // argument is type buffer
                    _this11.comp_renderer_nodesText.setArg(argNameKey, function () {
                        return _this11._customArgs[argNameKey].nodestext_array_value;
                    });
            };

            for (var argNameKey in this._customArgs) {
                _loop5(argNameKey);
            }
        }
    }, {
        key: "updateLinks",


        //██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗    ██╗     ██╗███╗   ██╗██╗  ██╗███████╗
        //██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝    ██║     ██║████╗  ██║██║ ██╔╝██╔════╝
        //██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗      ██║     ██║██╔██╗ ██║█████╔╝ ███████╗
        //██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝      ██║     ██║██║╚██╗██║██╔═██╗ ╚════██║
        //╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗    ███████╗██║██║ ╚████║██║  ██╗███████║
        // ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚══════╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
        value: function updateLinks() {
            var _this12 = this;

            console.log(Object.keys(this._links).length + " links");

            var _loop6 = function _loop6(na) {
                _this12.linksObj[na].componentRenderer.setArg("data", function () {
                    return _this12.linksObj[na].arrayLinkData;
                });
                _this12.linksObj[na].componentRenderer.getComponentBufferArg("dataB", _this12.comp_renderer_nodes);
                _this12.linksObj[na].componentRenderer.getComponentBufferArg("posXYZW", _this12.comp_renderer_nodes);
                _this12.linksObj[na].componentRenderer.setArg("nodeVertexPos", function () {
                    return _this12.linksObj[na].arrayLinkVertexPos;
                });
                _this12.linksObj[na].componentRenderer.setArg("indices", function () {
                    return _this12.linksObj[na].arrayLinkIndices;
                });

                _this12.linksObj[na].componentRenderer.setArg("nodesCount", function () {
                    return _this12.currentNodeId;
                });
                _this12.linksObj[na].componentRenderer.setArg("PMatrix", function () {
                    return _this12._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e;
                });
                _this12.linksObj[na].componentRenderer.setArgUpdatable("PMatrix", true);
                _this12.linksObj[na].componentRenderer.setArg("cameraWMatrix", function () {
                    return _this12._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e;
                });
                _this12.linksObj[na].componentRenderer.setArgUpdatable("cameraWMatrix", true);
                _this12.linksObj[na].componentRenderer.setArg("nodeWMatrix", function () {
                    return _this12.nodes.getComponent(Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e;
                });
                _this12.linksObj[na].componentRenderer.setArgUpdatable("nodeWMatrix", true);

                _this12.linksObj[na].componentRenderer.setArg("isLink", function () {
                    return 1;
                });
                _this12.linksObj[na].componentRenderer.setArg("bufferNodesWidth", function () {
                    return _this12.comp_renderer_nodes.getBuffers()["posXYZW"].W;
                });
                _this12.linksObj[na].componentRenderer.setArg("bufferLinksWidth", function () {
                    return _this12.linksObj[na].componentRenderer.getBuffers()["data"].W;
                });

                var _loop7 = function _loop7(argNameKey) {
                    var expl = _this12._customArgs[argNameKey].arg.split("*");
                    if (expl.length > 0) {
                        // argument is type buffer
                        _this12.linksObj[na].componentRenderer.setArg(argNameKey, function () {
                            return _this12._customArgs[argNameKey].links_array_value;
                        });
                    }
                };

                for (var argNameKey in _this12._customArgs) {
                    _loop7(argNameKey);
                }
            };

            for (var na = 0; na < this.linksObj.length; na++) {
                _loop6(na);
            }

            this.updateArrows();

            if (Object.keys(this._links).length > 0) this.updateAdjMat();
        }
    }, {
        key: "updateArrows",
        value: function updateArrows() {
            var _this13 = this;

            var _loop8 = function _loop8(na) {
                _this13.arrowsObj[na].componentRenderer.setArg("data", function () {
                    return _this13.arrowsObj[na].arrayArrowData;
                });
                _this13.arrowsObj[na].componentRenderer.getComponentBufferArg("dataB", _this13.comp_renderer_nodes);
                _this13.arrowsObj[na].componentRenderer.getComponentBufferArg("posXYZW", _this13.comp_renderer_nodes);

                _this13.arrowsObj[na].componentRenderer.setArg("nodeVertexPos", function () {
                    return _this13.arrowsObj[na].arrayArrowVertexPos;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("nodeVertexNormal", function () {
                    return _this13.arrowsObj[na].arrayArrowVertexNormal;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("nodeVertexTexture", function () {
                    return _this13.arrowsObj[na].arrayArrowVertexTexture;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("indices", function () {
                    return _this13.arrowsObj[na].arrayArrowIndices;
                });

                _this13.arrowsObj[na].componentRenderer.setArg("PMatrix", function () {
                    return _this13._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.PROJECTION).getMatrix().transpose().e;
                });
                _this13.arrowsObj[na].componentRenderer.setArgUpdatable("PMatrix", true);
                _this13.arrowsObj[na].componentRenderer.setArg("cameraWMatrix", function () {
                    return _this13._project.getActiveStage().getActiveCamera().getComponent(Constants.COMPONENT_TYPES.TRANSFORM_TARGET).getMatrix().transpose().e;
                });
                _this13.arrowsObj[na].componentRenderer.setArgUpdatable("cameraWMatrix", true);
                _this13.arrowsObj[na].componentRenderer.setArg("nodeWMatrix", function () {
                    return _this13.nodes.getComponent(Constants.COMPONENT_TYPES.TRANSFORM).getMatrixPosition().transpose().e;
                });
                _this13.arrowsObj[na].componentRenderer.setArgUpdatable("nodeWMatrix", true);

                _this13.arrowsObj[na].componentRenderer.setArg("nodesCount", function () {
                    return _this13.currentNodeId;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("isArrow", function () {
                    return 1.0;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("bufferNodesWidth", function () {
                    return _this13.comp_renderer_nodes.getBuffers()["posXYZW"].W;
                });
                _this13.arrowsObj[na].componentRenderer.setArg("bufferArrowsWidth", function () {
                    return _this13.arrowsObj[na].componentRenderer.getBuffers()["data"].W;
                });

                var _loop9 = function _loop9(argNameKey) {
                    var expl = _this13._customArgs[argNameKey].arg.split("*");
                    if (expl.length > 0) {
                        // argument is type buffer
                        _this13.arrowsObj[na].componentRenderer.setArg(argNameKey, function () {
                            return _this13._customArgs[argNameKey].arrows_array_value;
                        });
                    }
                };

                for (var argNameKey in _this13._customArgs) {
                    _loop9(argNameKey);
                }
            };

            for (var na = 0; na < this.arrowsObj.length; na++) {
                _loop8(na);
            }
        }
    }, {
        key: "updateAdjMat",
        value: function updateAdjMat() {
            var _this14 = this;

            var setAdjMat = function setAdjMat(columnAsParent, pixel, nodeId, nodeIdInv, weight, linkMultiplier, activationFunc, layerNum) {
                var idx = pixel * 4;

                _this14.arrAdjMatrix[idx] = layerNum;
                _this14.arrAdjMatrix[idx + 1] = 0.0; // weightQuadSum
                _this14.arrAdjMatrix[idx + 2] = columnAsParent === true ? 0.0 : weight;
                _this14.arrAdjMatrix[idx + 3] = columnAsParent === true ? 1.0 : 0.5; // columnAsParent=1.0;

                _this14.arrAdjMatrixB[idx] = 0.0; // weightAbsSum
                _this14.arrAdjMatrixB[idx + 1] = 0.0; // costA
                _this14.arrAdjMatrixB[idx + 2] = nodeId;
                _this14.arrAdjMatrixB[idx + 3] = nodeIdInv;

                _this14.arrAdjMatrixC[idx] = 0.0; // costB
                _this14.arrAdjMatrixC[idx + 1] = 0.0; // costC
                _this14.arrAdjMatrixC[idx + 2] = 0.0; // costD
                _this14.arrAdjMatrixC[idx + 3] = 0.0; // costE

                _this14.arrAdjMatrixD[idx] = 0.0; // costF
                _this14.arrAdjMatrixD[idx + 1] = 0.0; // costG
                _this14.arrAdjMatrixD[idx + 2] = linkMultiplier;
                _this14.arrAdjMatrixD[idx + 3] = activationFunc;
            };

            this.arrAdjMatrix = new Float32Array(this._ADJ_MATRIX_WIDTH * this._ADJ_MATRIX_WIDTH * 4);
            this.arrAdjMatrixB = new Float32Array(this._ADJ_MATRIX_WIDTH * this._ADJ_MATRIX_WIDTH * 4);
            this.arrAdjMatrixC = new Float32Array(this._ADJ_MATRIX_WIDTH * this._ADJ_MATRIX_WIDTH * 4);
            this.arrAdjMatrixD = new Float32Array(this._ADJ_MATRIX_WIDTH * this._ADJ_MATRIX_WIDTH * 4);
            for (var key in this._links) {
                var childNodeId = this._links[key].origin_nodeId;
                var parentNodeId = this._links[key].target_nodeId;

                var pixelParent = this.getPixelParent(parentNodeId, childNodeId);
                var pixelChild = this.getPixelChild(parentNodeId, childNodeId);
                setAdjMat(true, pixelParent, parentNodeId, childNodeId, this._links[key].weight, this._links[key].linkMultiplier, this._links[key].activationFunc, -99); // (columns=child;rows=parent)
                setAdjMat(false, pixelChild, childNodeId, parentNodeId, this._links[key].weight, this._links[key].linkMultiplier, this._links[key].activationFunc, this._links[key].layerNum); // (columns=parent;rows=child)
            }

            this.comp_renderer_nodes.setArg("adjacencyMatrix", function () {
                return _this14.arrAdjMatrix;
            });
            this.comp_renderer_nodes.setArg("adjacencyMatrixB", function () {
                return _this14.arrAdjMatrixB;
            });
            this.comp_renderer_nodes.setArg("adjacencyMatrixC", function () {
                return _this14.arrAdjMatrixC;
            });
            this.comp_renderer_nodes.setArg("adjacencyMatrixD", function () {
                return _this14.arrAdjMatrixD;
            });

            /*
            this.adjacencyMatrixToImage(this.arrAdjMatrix, this._ADJ_MATRIX_WIDTH, (img) => {
                document.body.appendChild(img);
                img.style.border = "1px solid red";
            }); */
        }
    }, {
        key: "getPixelParent",
        value: function getPixelParent(p, c) {
            return p * this._ADJ_MATRIX_WIDTH + c;
        }
    }, {
        key: "getPixelChild",
        value: function getPixelChild(p, c) {
            return c * this._ADJ_MATRIX_WIDTH + p;
        }

        /**
         * adjacencyMatrixToImage
         * @param {Float32Array} adjMat
         * @param {int} width
         * @param {Function} onload
         */

    }, {
        key: "adjacencyMatrixToImage",
        value: function adjacencyMatrixToImage(adjMat, width, onload) {
            var _this15 = this;

            var toArrF = function toArrF(arr) {
                var arrO = new Uint8Array(arr.length * 4);
                for (var n = 0; n < arr.length; n++) {
                    var idO = n * 4;
                    arrO[idO] = arr[n] * 255;
                    arrO[idO + 1] = arr[n] * 255;
                    arrO[idO + 2] = arr[n] * 255;
                    arrO[idO + 3] = 255;
                }

                return arrO;
            };

            var toImage = function toImage(arrO, w, h) {
                var canvas = Utils.getCanvasFromUint8Array(arrO, w, h);
                _this15._utils.getImageFromCanvas(canvas, function (im) {
                    onload(im);
                });
            };

            var arrF = toArrF(adjMat);
            toImage(arrF, width, width);
        }
    }, {
        key: "enableForceLayout",
        value: function enableForceLayout() {
            this.comp_renderer_nodes.setArg("enableForceLayout", function () {
                return 1.0;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("enableForceLayout", function () {
                    return 1.0;
                });
            }this._enabledForceLayout = true;
        }
    }, {
        key: "disableForceLayout",
        value: function disableForceLayout() {
            this.comp_renderer_nodes.setArg("enableForceLayout", function () {
                return 0.0;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("enableForceLayout", function () {
                    return 0.0;
                });
            }this._enabledForceLayout = false;
        }
    }, {
        key: "enableShowOutputWeighted",
        value: function enableShowOutputWeighted() {
            this.comp_renderer_nodes.setArg("multiplyOutput", function () {
                return 1.0;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("multiplyOutput", function () {
                    return 1.0;
                });
            }for (var _na16 = 0; _na16 < this.arrowsObj.length; _na16++) {
                this.arrowsObj[_na16].componentRenderer.setArg("multiplyOutput", function () {
                    return 1.0;
                });
            }
        }
    }, {
        key: "disableShowOutputWeighted",
        value: function disableShowOutputWeighted() {
            this.comp_renderer_nodes.setArg("multiplyOutput", function () {
                return 0.0;
            });
            for (var na = 0; na < this.linksObj.length; na++) {
                this.linksObj[na].componentRenderer.setArg("multiplyOutput", function () {
                    return 0.0;
                });
            }for (var _na17 = 0; _na17 < this.arrowsObj.length; _na17++) {
                this.arrowsObj[_na17].componentRenderer.setArg("multiplyOutput", function () {
                    return 0.0;
                });
            }
        }
    }, {
        key: "enableShowWeightDynamics",
        value: function enableShowWeightDynamics() {
            this.comp_renderer_nodes.setArg("viewNeuronDynamics", function () {
                return 1.0;
            });
        }
    }, {
        key: "disableShowWeightDynamics",
        value: function disableShowWeightDynamics() {
            this.comp_renderer_nodes.setArg("viewNeuronDynamics", function () {
                return 0.0;
            });
        }
    }, {
        key: "enableShowValues",
        value: function enableShowValues() {
            this.comp_renderer_nodesText.setArg("showValues", function () {
                return 1.0;
            });
        }
    }, {
        key: "disableShowValues",
        value: function disableShowValues() {
            this.comp_renderer_nodesText.setArg("showValues", function () {
                return 0.0;
            });
        }
    }]);

    return Graph;
}();

global.Graph = Graph;
module.exports.Graph = Graph;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./KERNEL_ADJMATRIX_UPDATE.class":9,"./KERNEL_DIR.class":10,"./ProccessImg.class":12,"./VFP_NODE.class":13,"./VFP_NODEPICKDRAG.class":14,"./resources":18,"scejs":4}],9:[function(require,module,exports){
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
            return ["x", ["adjacencyMatrix", "adjacencyMatrixB", "adjacencyMatrixC", "adjacencyMatrixD"],
            // head
            "",

            // source
            "vec4 adjMat = adjacencyMatrix[x]; \n            vec4 adjMatB = adjacencyMatrixB[x];\n            vec4 adjMatC = adjacencyMatrixC[x];\n            vec4 adjMatD = adjacencyMatrixD[x];\n\n            float linkLayerNum = adjMat.x;\n            float weightQuadSum = adjMat.y;\n            float linkWeight = adjMat.z;\n            float linkTypeParent = adjMat.w;\n            \n            float multiplier = adjMatD.z;\n            float weightAbsSum = adjMatB.x;\n            float costA = adjMatB.y;\n            float costB = adjMatC.x;\n            float costC = adjMatC.y;\n            float costD = adjMatC.z;\n            float costE = adjMatC.w;\n            float costF = adjMatD.x;\n            float costG = adjMatD.y;\n            \n            if(multiplier != 1.0) {\n                linkWeight = multiplier;\n            } else if(currentTrainLayer == -10.0) { \n                costA = 0.0;\n                costB = 0.0;\n                costC = 0.0;\n                costD = 0.0;\n                costE = 0.0;\n                costF = 0.0;\n                costG = 0.0;\n            } else if(linkTypeParent == 0.5 && linkLayerNum >= 0.0) {\n                float id = adjMatB.z;\n                float idInv = adjMatB.w;\n            \n                vec2 xGeometryCurrentChild = get_global_id(id, bufferNodesWidth, " + geometryLength.toFixed(1) + ");\n                vec2 xGeometryParent = get_global_id(idInv, bufferNodesWidth, " + geometryLength.toFixed(1) + (");\n\n\n                float childBiasNode = dataB[xGeometryCurrentChild].x;\n                \n                float childGOutputA = dataB[xGeometryCurrentChild].z;\n                float childGOutputB = dataF[xGeometryCurrentChild].x;\n                float childGOutputC = dataF[xGeometryCurrentChild].z;\n                float childGOutputD = dataG[xGeometryCurrentChild].x;\n                float childGOutputE = dataG[xGeometryCurrentChild].z;\n                float childGOutputF = dataH[xGeometryCurrentChild].x;\n                float childGOutputG = dataH[xGeometryCurrentChild].z;\n                \n                \n                float parentGOutputA = dataB[xGeometryParent].z;\n                float parentGOutputB = dataF[xGeometryParent].x;\n                float parentGOutputC = dataF[xGeometryParent].z;\n                float parentGOutputD = dataG[xGeometryParent].x;\n                float parentGOutputE = dataG[xGeometryParent].z;\n                float parentGOutputF = dataH[xGeometryParent].x;\n                float parentGOutputG = dataH[xGeometryParent].z;\n                \n                float parentGDeltaA = dataB[xGeometryParent].y;\n                float parentGDeltaB = dataB[xGeometryParent].w;\n                float parentGDeltaC = dataF[xGeometryParent].y;\n                float parentGDeltaD = dataF[xGeometryParent].w;\n                float parentGDeltaE = dataG[xGeometryParent].y;\n                float parentGDeltaF = dataG[xGeometryParent].w;\n                float parentGDeltaG = dataH[xGeometryParent].y;\n                \n                \n                \n                float lr = learningRate;\n                float l1_decay = 0.0;\n                float l2_decay = 0.01;\n                float gpu_batch_size = 7.0;     \n                \n                if(currentTrainLayer == linkLayerNum) {\n                    if(weightQuadSum != 0.0) {                        \n                        float parentGOutputDerivA = 1.0;                    \n                        float parentGOutputDerivB = 1.0;\n                        float parentGOutputDerivC = 1.0;\n                        float parentGOutputDerivD = 1.0;\n                        float parentGOutputDerivE = 1.0;\n                        float parentGOutputDerivF = 1.0;\n                        float parentGOutputDerivG = 1.0;\n                        if(linkLayerNum < layerCount-2.0) {\n                            parentGOutputDerivA = (parentGOutputA <= 0.0) ? 0.01 : 1.0;                    \n                            parentGOutputDerivB = (parentGOutputB <= 0.0) ? 0.01 : 1.0;\n                            parentGOutputDerivC = (parentGOutputC <= 0.0) ? 0.01 : 1.0;\n                            parentGOutputDerivD = (parentGOutputD <= 0.0) ? 0.01 : 1.0;\n                            parentGOutputDerivE = (parentGOutputE <= 0.0) ? 0.01 : 1.0;\n                            parentGOutputDerivF = (parentGOutputF <= 0.0) ? 0.01 : 1.0;\n                            parentGOutputDerivG = (parentGOutputG <= 0.0) ? 0.01 : 1.0;\n                        }\n                        \n                        float dA = parentGDeltaA*parentGOutputDerivA;\n                        float dB = parentGDeltaB*parentGOutputDerivB;\n                        float dC = parentGDeltaC*parentGOutputDerivC;\n                        float dD = parentGDeltaD*parentGOutputDerivD;\n                        float dE = parentGDeltaE*parentGOutputDerivE;\n                        float dF = parentGDeltaF*parentGOutputDerivF;\n                        float dG = parentGDeltaG*parentGOutputDerivG;\n                        \n                        float wT = 0.0;\n                        wT += dA*childGOutputA;\n                        wT += dB*childGOutputB;\n                        wT += dC*childGOutputC;\n                        wT += dD*childGOutputD;\n                        wT += dE*childGOutputE;\n                        wT += dF*childGOutputF;\n                        wT += dG*childGOutputG;\n                        wT /= (gpu_batch_size*batch_repeats);\n                    \n                        costA = dA*linkWeight;\n                        costB = dB*linkWeight;\n                        costC = dC*linkWeight;\n                        costD = dD*linkWeight;\n                        costE = dE*linkWeight;\n                        costF = dF*linkWeight;\n                        costG = dG*linkWeight;\n                        \n                        linkWeight += -lr*(" + " wT);\n                        weightQuadSum = 0.0;\n                        weightAbsSum = 0.0;                        \n                    } else {\n                        weightQuadSum += linkWeight*linkWeight;\n                        weightAbsSum += abs(linkWeight);\n                    }\n                }\n            }\n            \n            return [vec4(linkLayerNum, weightQuadSum, linkWeight, linkTypeParent), vec4(weightAbsSum, costA, adjMatB.z, adjMatB.w), vec4(costB, costC, costD, costE), vec4(costF, costG, 0.0, 0.0)];\n            ")];
        }
    }]);

    return KERNEL_ADJMATRIX_UPDATE;
}();

global.KERNEL_ADJMATRIX_UPDATE = KERNEL_ADJMATRIX_UPDATE;
module.exports.KERNEL_ADJMATRIX_UPDATE = KERNEL_ADJMATRIX_UPDATE;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],10:[function(require,module,exports){
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
            "float nodeId = data[x].x;\n            vec2 xGeometry = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n            vec3 currentPos = posXYZW[xGeometry].xyz;\n            vec3 currentDir = dir[xGeometry].xyz;\n\n\n            vec4 currentDataB = dataB[xGeometry];\n            vec4 currentDataF = dataF[xGeometry];\n            vec4 currentDataG = dataG[xGeometry];\n            vec4 currentDataH = dataH[xGeometry];\n\n            currentDir = vec3(0.0, 0.0, 0.0);\n\n            \n            vec3 atraction = vec3(0.0, 0.0, 0.0);\n            float acumAtraction = 1.0;\n            vec3 repulsion = vec3(0.0, 0.0, 0.0);\n\n            vec3 force = vec3(0.0, 0.0, 0.0);\n\n\n            float netChildInputSumA = 0.0;\n            float foutputA = 0.0;\n            float netParentErrorWeightA = 0.0;\n            \n            float netChildInputSumB = 0.0;\n            float foutputB = 0.0;\n            float netParentErrorWeightB = 0.0;\n            \n            float netChildInputSumC = 0.0;\n            float foutputC = 0.0;\n            float netParentErrorWeightC = 0.0;\n            \n            float netChildInputSumD = 0.0;\n            float foutputD = 0.0;\n            float netParentErrorWeightD = 0.0;\n            \n            float netChildInputSumE = 0.0;\n            float foutputE = 0.0;\n            float netParentErrorWeightE = 0.0;\n            \n            float netChildInputSumF = 0.0;\n            float foutputF = 0.0;\n            float netParentErrorWeightF = 0.0;\n            \n            float netChildInputSumG = 0.0;\n            float foutputG = 0.0;\n            float netParentErrorWeightG = 0.0;\n            \n\n            if(nodeId < nodesCount && enableTrain == 0.0) {\n                float currentActivationFn = 0.0;\n                vec2 xGeomCurrent = get_global_id(nodeId, uBufferWidth, " + geometryLength.toFixed(1) + ");\n                for(int n=0; n < 4096; n++) {\n                    if(float(n) >= nodesCount) {break;}\n                    if(float(n) != nodeId) {\n                        vec2 xGeomOpposite = get_global_id(float(n), uBufferWidth, " + geometryLength.toFixed(1) + ");\n\n\n                        vec2 xAdjMatCurrent = get_global_id(vec2(float(n), nodeId), widthAdjMatrix);\n                        vec2 xAdjMatOpposite = get_global_id(vec2(nodeId, float(n)), widthAdjMatrix);\n\n                        vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n                        vec4 pixAdjMatAOpposite = adjacencyMatrix[xAdjMatOpposite];\n\n                        vec4 pixAdjMatBCurrent = adjacencyMatrixB[xAdjMatCurrent];\n                        vec4 pixAdjMatBOpposite = adjacencyMatrixB[xAdjMatOpposite];\n                        \n                        vec4 pixAdjMatCCurrent = adjacencyMatrixC[xAdjMatCurrent];\n                        vec4 pixAdjMatCOpposite = adjacencyMatrixC[xAdjMatOpposite];\n                        \n                        vec4 pixAdjMatDCurrent = adjacencyMatrixD[xAdjMatCurrent];\n                        vec4 pixAdjMatDOpposite = adjacencyMatrixD[xAdjMatOpposite];\n\n\n                                                                    \n                        " + "\n                        float currentLayerNum = pixAdjMatACurrent.x;\n                        float currentWeight = pixAdjMatACurrent.z;\n                        float currentIsParent = pixAdjMatACurrent.w;            \n                        " + "\n                        float oppositeLayerNum = pixAdjMatAOpposite.x;\n                        float oppositeWeight = pixAdjMatAOpposite.z;\n                        float oppositeIsParent = pixAdjMatAOpposite.w;\n            \n            \n                        " + "\n                        float currentCostA = pixAdjMatBCurrent.y;            \n                        " + "            \n            \n                        " + "\n                        float currentCostB = pixAdjMatCCurrent.x;\n                        float currentCostC = pixAdjMatCCurrent.y;\n                        float currentCostD = pixAdjMatCCurrent.z;\n                        float currentCostE = pixAdjMatCCurrent.w;            \n                        " + "                        \n                        \n                        " + "\n                        float currentCostF = pixAdjMatDCurrent.x;\n                        float currentCostG = pixAdjMatDCurrent.y;           \n                        " + "\n                        \n                        \n            \n                        " + "            \n                        " + "                        \n                        float oppositeNetOutputA = dataB[xGeomOpposite].z;                        \n                        float oppositeNetOutputB = dataF[xGeomOpposite].x;                    \n                        float oppositeNetOutputC = dataF[xGeomOpposite].z;                    \n                        float oppositeNetOutputD = dataG[xGeomOpposite].x;                    \n                        float oppositeNetOutputE = dataG[xGeomOpposite].z;                    \n                        float oppositeNetOutputF = dataH[xGeomOpposite].x;                    \n                        float oppositeNetOutputG = dataH[xGeomOpposite].z;\n            \n            \n                        " + "\n                        " + "\n                        " + "            \n                        " + "\n                        vec3 oppositePos = posXYZW[xGeomOpposite].xyz;\n                        vec3 oppositeDir = dir[xGeomOpposite].xyz;\n            \n                        " + "\n                        vec3 dirToOpposite = (oppositePos-currentPos);\n                        vec3 dirToOppositeN = normalize(dirToOpposite);\n            \n                        float dist = distance(oppositePos, currentPos); " + "\n                        float distN = max(0.0,dist)/100000.0;\n            \n                        float mm = 10000000.0;\n                        float m1 = 400000.0/mm;\n                        float m2 = 48.0/mm;\n                        if(currentIsParent == 1.0) {\n                            netChildInputSumA += oppositeNetOutputA*oppositeWeight;\n                            netChildInputSumB += oppositeNetOutputB*oppositeWeight;\n                            netChildInputSumC += oppositeNetOutputC*oppositeWeight;\n                            netChildInputSumD += oppositeNetOutputD*oppositeWeight;\n                            netChildInputSumE += oppositeNetOutputE*oppositeWeight;\n                            netChildInputSumF += oppositeNetOutputF*oppositeWeight;\n                            netChildInputSumG += oppositeNetOutputG*oppositeWeight;\n                            \n                            atraction += dirToOppositeN*max(1.0, distN*abs(oppositeWeight)*(m1/2.0));\n                            repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(oppositeWeight)*(m2/2.0));\n                            acumAtraction += 1.0;\n                        } else if(currentIsParent == 0.5) {\n                            netParentErrorWeightA += currentCostA;\n                            netParentErrorWeightB += currentCostB;\n                            netParentErrorWeightC += currentCostC;\n                            netParentErrorWeightD += currentCostD;\n                            netParentErrorWeightE += currentCostE;\n                            netParentErrorWeightF += currentCostF;\n                            netParentErrorWeightG += currentCostG;\n                            \n                            atraction += dirToOppositeN*max(1.0, distN*abs(currentWeight)*m1);\n                            repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*m2);\n                            acumAtraction += 1.0;\n                        }\n            \n                        repulsion += -dirToOppositeN*max(1.0, (1.0-distN)*abs(currentWeight)*(m2/8.0));\n                        acumAtraction += 1.0;\n                    }\n                }\n                \n                float vndm = (viewNeuronDynamics == 1.0) ? netChildInputSumA : 1.0;\n                force += (atraction/acumAtraction)*abs(vndm);\n                force += (repulsion/acumAtraction)*abs(vndm);\n                currentDir += force;\n                \n                \n                float currentBiasNode = dataB[xGeomCurrent].x;\n                \n                " + KERNEL_DIR.efferentNodesStr(afferentNodesCount, efferentStart, efferentNodesCount) + "\n                \n                currentDataB = vec4(currentDataB.x, netParentErrorWeightA, foutputA, netParentErrorWeightB);\n                currentDataF = vec4(foutputB, netParentErrorWeightC, foutputC, netParentErrorWeightD);\n                currentDataG = vec4(foutputD, netParentErrorWeightE, foutputE, netParentErrorWeightF);\n                currentDataH = vec4(foutputF, netParentErrorWeightG, foutputG, 0.0);\n            }\n\n            " + (customCode !== undefined ? customCode : '') + "\n\n            if(enableDrag == 1.0) {\n                if(nodeId == idToDrag) {\n                    currentPos = vec3(MouseDragTranslationX, MouseDragTranslationY, MouseDragTranslationZ);\n                }\n            }\n\n            currentPos += currentDir;\n            if(only2d == 1.0) {\n                currentPos.y = 0.0;\n            }\n\n            " + returnStr];
        }
    }, {
        key: "efferentNodesStr",
        value: function efferentNodesStr(afferentNodesCount, efferentStart, efferentNodesCount) {
            /////////////////////////////////////////////////
            // OUTPUT
            /////////////////////////////////////////////////
            var str = "\n            if(freezeOutput == 0.0) {\n                if(nodeId < afferentNodesCount) {\n                    for(float n=0.0; n < 1024.0; n+=1.0) {\n                        if(n >= afferentNodesCount) {\n                            break;\n                        }\n                        if(nodeId == n) {\n                            foutputA = afferentNodesA[int(n)];\n                            foutputB = afferentNodesB[int(n)];\n                            foutputC = afferentNodesC[int(n)];\n                            foutputD = afferentNodesD[int(n)];\n                            foutputE = afferentNodesE[int(n)];\n                            foutputF = afferentNodesF[int(n)];\n                            foutputG = afferentNodesG[int(n)];\n                            break;\n                        }\n                    }\n                } else {\n                    if(currentBiasNode == 0.0) {\n                        if(nodeId >= " + efferentStart.toFixed(1) + (") {\n                            foutputA = netChildInputSumA;\n                            foutputB = netChildInputSumB;\n                            foutputC = netChildInputSumC;\n                            foutputD = netChildInputSumD;\n                            foutputE = netChildInputSumE;\n                            foutputF = netChildInputSumF;\n                            foutputG = netChildInputSumG;\n                        } else {                                  \n                            foutputA = (netChildInputSumA <= 0.0) ? 0.01*netChildInputSumA : netChildInputSumA; " + "\n                            foutputB = (netChildInputSumB <= 0.0) ? 0.01*netChildInputSumB : netChildInputSumB;\n                            foutputC = (netChildInputSumC <= 0.0) ? 0.01*netChildInputSumC : netChildInputSumC;\n                            foutputD = (netChildInputSumD <= 0.0) ? 0.01*netChildInputSumD : netChildInputSumD;\n                            foutputE = (netChildInputSumE <= 0.0) ? 0.01*netChildInputSumE : netChildInputSumE;\n                            foutputF = (netChildInputSumF <= 0.0) ? 0.01*netChildInputSumF : netChildInputSumF;\n                            foutputG = (netChildInputSumG <= 0.0) ? 0.01*netChildInputSumG : netChildInputSumG;\n                        }\n                    } else {\n                        foutputA = 1.0;\n                        foutputB = 1.0;\n                        foutputC = 1.0;\n                        foutputD = 1.0;\n                        foutputE = 1.0;\n                        foutputF = 1.0;\n                        foutputG = 1.0;\n                    }\n                }\n            } else {\n                foutputA = currentDataB.z;\n                foutputB = currentDataF.x;\n                foutputC = currentDataF.z;\n                foutputD = currentDataG.x;\n                foutputE = currentDataG.z;\n                foutputF = currentDataH.x;\n                foutputG = currentDataH.z;\n            }");

            /////////////////////////////////////////////////
            // ERROR
            /////////////////////////////////////////////////
            for (var n = efferentStart; n < efferentStart + efferentNodesCount; n++) {
                var cond = n === efferentStart ? "if" : "else if";
                str += "\n            " + cond + "(nodeId == " + n.toFixed(1) + (") {\n                if(freezeOutput == 0.0) {\n                    foutputA = netChildInputSumA; " + " \n                    foutputB = netChildInputSumB;\n                    foutputC = netChildInputSumC;\n                    foutputD = netChildInputSumD;\n                    foutputE = netChildInputSumE;\n                    foutputF = netChildInputSumF;\n                    foutputG = netChildInputSumG;\n                } else {\n                    foutputA = currentDataB.z;\n                    foutputB = currentDataF.x;\n                    foutputC = currentDataF.z;\n                    foutputD = currentDataG.x;\n                    foutputE = currentDataG.z;\n                    foutputF = currentDataH.x;\n                    foutputG = currentDataH.z;\n                }               \n                netParentErrorWeightA = efferentNodesA[") + Math.round(n - efferentStart) + "];\n                netParentErrorWeightB = efferentNodesB[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightC = efferentNodesC[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightD = efferentNodesD[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightE = efferentNodesE[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightF = efferentNodesF[" + Math.round(n - efferentStart) + "];\n                netParentErrorWeightG = efferentNodesG[" + Math.round(n - efferentStart) + "];\n            }";
            }
            return str;
        }
    }]);

    return KERNEL_DIR;
}();

global.KERNEL_DIR = KERNEL_DIR;
module.exports.KERNEL_DIR = KERNEL_DIR;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],11:[function(require,module,exports){
(function (global){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

// VIS library from CONVNETJS
var Plot = exports.Plot = function () {
    function Plot(options) {
        _classCallCheck(this, Plot);

        options = options || {};
        this.step_horizon = options.step_horizon || 1000;

        this.currentMode = 0; // 0 total; 1 last 1000
        this.pts = [];
        this.ptsT = [];

        this.maxy = -9999;
        this.miny = 9999;
    }

    // canv is the canvas we wish to update with this new datapoint


    _createClass(Plot, [{
        key: 'add',
        value: function add(step, y) {
            var time = new Date().getTime(); // in ms
            if (y > this.maxy * 0.99) this.maxy = y * 1.05;
            if (y < this.miny * 1.01) this.miny = y * 0.95;

            this.pts.push({ step: step, time: time, y: y });
            if (this.pts.length > 1000) this.pts.shift();
            this.ptsT.push({ step: step, time: time, y: y });
            if (step > this.step_horizon) this.step_horizon *= 2;
        }
        // elt is a canvas we wish to draw into

    }, {
        key: 'drawSelf',
        value: function drawSelf(canv) {
            var pad = 25;
            var H = canv.height;
            var W = canv.width;
            var ctx = canv.getContext('2d');

            ctx.clearRect(0, 0, W, H);
            ctx.font = "10px Georgia";

            var f2t = function f2t(x) {
                var dd = 1.0 * Math.pow(10, 2);
                return '' + Math.floor(x * dd) / dd;
            };

            var horizon = this.currentMode === 0 ? this.step_horizon : 1000;
            // draw guidelines and values
            ctx.strokeStyle = "#999";
            ctx.beginPath();
            var ng = 10;
            for (var i = 0; i <= ng; i++) {
                var xpos = i / ng * (W - 2 * pad) + pad;
                ctx.moveTo(xpos, pad);
                ctx.lineTo(xpos, H - pad);
                ctx.fillText(f2t(i / ng * horizon / 1000) + 'k', xpos, H - pad + 14);
            }
            for (var _i = 0; _i <= ng; _i++) {
                var ypos = _i / ng * (H - 2 * pad) + pad;
                ctx.moveTo(pad, ypos);
                ctx.lineTo(W - pad, ypos);
                ctx.fillText(f2t((ng - _i) / ng * (this.maxy - this.miny) + this.miny), 0, ypos);
            }
            ctx.stroke();

            var selectedPts = this.currentMode === 0 ? this.ptsT : this.pts;
            var N = selectedPts.length;
            if (N < 2) return;

            // draw the actual curve
            var t = function t(x, y, s) {
                var horizon = s.currentMode === 0 ? s.step_horizon : 1000;

                var tx = x / horizon * (W - pad * 2) + pad;
                var ty = H - ((y - s.miny) / (s.maxy - s.miny) * (H - pad * 2) + pad);
                return { tx: tx, ty: ty };
            };

            ctx.strokeStyle = "red";
            ctx.beginPath();
            for (var _i2 = 0; _i2 < N; _i2++) {
                // draw line from i-1 to i
                var p = selectedPts[_i2];
                var pt = t(this.currentMode === 0 ? p.step : _i2, p.y, this);
                if (_i2 === 0) ctx.moveTo(pt.tx, pt.ty);else ctx.lineTo(pt.tx, pt.ty);
            }
            ctx.stroke();
        }
    }, {
        key: 'clear',
        value: function clear() {}
    }]);

    return Plot;
}();

global.Plot = Plot;
module.exports.Plot = Plot;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}],12:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.ProccessImg = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

require("scejs");

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var ProccessImg = exports.ProccessImg = function () {
    function ProccessImg(jsonIn) {
        _classCallCheck(this, ProccessImg);

        this.stackNodesImg = jsonIn.stackNodesImg;
        this.drawedImages = 0;
        this.output = {
            "nodesImg": null,
            "nodesImgCrosshair": null };

        var NODE_IMG_WIDTH = jsonIn.NODE_IMG_WIDTH;
        this.NODE_IMG_COLUMNS = jsonIn.NODE_IMG_COLUMNS;
        this.NODE_IMG_SPRITE_WIDTH = NODE_IMG_WIDTH / this.NODE_IMG_COLUMNS;

        var nodesImgMask = jsonIn.nodesImgMask;
        this.datMask = Utils.getUint8ArrayFromHTMLImageElement(nodesImgMask);

        var nodesImgCrosshair = jsonIn.nodesImgCrosshair;
        this.datCrosshair = Utils.getUint8ArrayFromHTMLImageElement(nodesImgCrosshair);

        this.canvasNodeImg = document.createElement('canvas');
        this.canvasNodeImg.width = NODE_IMG_WIDTH;
        this.canvasNodeImg.height = NODE_IMG_WIDTH;
        this.ctxNodeImg = this.canvasNodeImg.getContext('2d');

        this.canvasNodeImgCrosshair = document.createElement('canvas');
        this.canvasNodeImgCrosshair.width = NODE_IMG_WIDTH;
        this.canvasNodeImgCrosshair.height = NODE_IMG_WIDTH;
        this.ctxNodeImgCrosshair = this.canvasNodeImgCrosshair.getContext('2d');

        var canvasNodeImgTMP = document.createElement('canvas');
        canvasNodeImgTMP.width = this.NODE_IMG_SPRITE_WIDTH;
        canvasNodeImgTMP.height = this.NODE_IMG_SPRITE_WIDTH;
        var ctxNodeImgTMP = canvasNodeImgTMP.getContext('2d');

        for (var n = 0; n < this.stackNodesImg.length; n++) {
            var currStack = this.stackNodesImg[n];
            var image = new Image();
            image.onload = function (image, currStack) {
                ctxNodeImgTMP.clearRect(0, 0, this.NODE_IMG_SPRITE_WIDTH, this.NODE_IMG_SPRITE_WIDTH);
                var quarter = this.NODE_IMG_SPRITE_WIDTH / 4;
                ctxNodeImgTMP.drawImage(image, 0, 0, image.width, image.height, quarter, quarter, this.NODE_IMG_SPRITE_WIDTH / 2, this.NODE_IMG_SPRITE_WIDTH / 2);

                new Utils().getImageFromCanvas(canvasNodeImgTMP, function (currStack, img) {
                    currStack.image = Utils.getUint8ArrayFromHTMLImageElement(img);

                    var allImg = true;
                    for (var nb = 0; nb < this.stackNodesImg.length; nb++) {
                        if (this.stackNodesImg[nb].image == null) {
                            allImg = false;
                            break;
                        }
                    }

                    if (allImg === true) this.generateAll(jsonIn);
                }.bind(this, currStack));
            }.bind(this, image, currStack);
            image.src = currStack.url;
        }
    }

    _createClass(ProccessImg, [{
        key: "generateAll",
        value: function generateAll(jsonIn) {
            var drawOnAtlas = function (currStack, ctx, newImgData) {
                var _this = this;

                var get2Dfrom1D = function ( /*Int*/idx, /*Int*/columns) {
                    var n = idx / columns;
                    var row = parseFloat(parseInt(n));
                    var col = Utils.fract(n) * columns;

                    return {
                        "col": col,
                        "row": row };
                }.bind(this);

                var readAll = function (onend) {
                    var pasteImg = function (onend, imgname, img) {
                        this.output[imgname] = img;
                        if (this.output.nodesImg != null && this.output.nodesImgCrosshair != null) onend(this.output);
                    }.bind(this, onend);

                    new Utils().getImageFromCanvas(this.canvasNodeImg, function (imgAtlas) {
                        pasteImg("nodesImg", imgAtlas);
                    }.bind(this));
                    new Utils().getImageFromCanvas(this.canvasNodeImgCrosshair, function (imgAtlas) {
                        pasteImg("nodesImgCrosshair", imgAtlas);
                    }.bind(this));
                }.bind(this, jsonIn.onend);

                new Utils().getImageFromCanvas(Utils.getCanvasFromUint8Array(newImgData, this.NODE_IMG_SPRITE_WIDTH, this.NODE_IMG_SPRITE_WIDTH), function (imgB) {
                    // draw image on atlas
                    var loc = get2Dfrom1D(currStack.locationIdx, _this.NODE_IMG_COLUMNS);
                    ctx.drawImage(imgB, loc.col * _this.NODE_IMG_SPRITE_WIDTH, loc.row * _this.NODE_IMG_SPRITE_WIDTH, _this.NODE_IMG_SPRITE_WIDTH, _this.NODE_IMG_SPRITE_WIDTH);

                    _this.drawedImages++;
                    if (_this.drawedImages === _this.stackNodesImg.length * 2) readAll();
                });
            }.bind(this);

            for (var n = 0; n < this.stackNodesImg.length; n++) {
                var currStack = this.stackNodesImg[n];
                var newImgData = this.stackNodesImg[n].image;

                // masked image
                for (var nb = 0; nb < this.datMask.length / 4; nb++) {
                    var idx = nb * 4;
                    if (newImgData[idx + 3] > 0) newImgData[idx + 3] = this.datMask[idx + 3];
                }
                drawOnAtlas(currStack, this.ctxNodeImg, newImgData);

                // crosshair image
                for (var _nb = 0; _nb < this.datCrosshair.length / 4; _nb++) {
                    var _idx = _nb * 4;

                    newImgData[_idx] = (this.datCrosshair[_idx] * this.datCrosshair[_idx + 3] + newImgData[_idx] * (255 - this.datCrosshair[_idx + 3])) / 255;
                    newImgData[_idx + 1] = (this.datCrosshair[_idx + 1] * this.datCrosshair[_idx + 3] + newImgData[_idx + 1] * (255 - this.datCrosshair[_idx + 3])) / 255;
                    newImgData[_idx + 2] = (this.datCrosshair[_idx + 2] * this.datCrosshair[_idx + 3] + newImgData[_idx + 2] * (255 - this.datCrosshair[_idx + 3])) / 255;
                    newImgData[_idx + 3] = (this.datCrosshair[_idx + 3] * this.datCrosshair[_idx + 3] + newImgData[_idx + 3] * (255 - this.datCrosshair[_idx + 3])) / 255;
                }
                drawOnAtlas(currStack, this.ctxNodeImgCrosshair, newImgData);
            }
        }
    }]);

    return ProccessImg;
}();

global.ProccessImg = ProccessImg;
module.exports.ProccessImg = ProccessImg;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"scejs":4}],13:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.VFP_NODE = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

require("scejs");

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var VFP_NODE = exports.VFP_NODE = function () {
    function VFP_NODE() {
        _classCallCheck(this, VFP_NODE);
    }

    _createClass(VFP_NODE, null, [{
        key: "getSrc",
        value: function getSrc(customCode, geometryLength) {
            return [["RGB"],

            //██╗   ██╗███████╗██████╗ ████████╗███████╗██╗  ██╗    ██╗  ██╗███████╗ █████╗ ██████╗
            //██║   ██║██╔════╝██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██║  ██║██╔════╝██╔══██╗██╔══██╗
            //██║   ██║█████╗  ██████╔╝   ██║   █████╗   ╚███╔╝     ███████║█████╗  ███████║██║  ██║
            //╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██╔══╝   ██╔██╗     ██╔══██║██╔══╝  ██╔══██║██║  ██║
            // ╚████╔╝ ███████╗██║  ██║   ██║   ███████╗██╔╝ ██╗    ██║  ██║███████╗██║  ██║██████╔╝
            //  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝
            'varying vec4 vVertexColor;\n' + 'varying vec4 vVertexColorNetError;\n' + 'varying vec4 vUV;\n' + 'varying vec2 vVertexUV;\n' + 'varying float vUseTex;\n' + 'varying vec4 vWNMatrix;\n' + 'varying float vNodeId;\n' + 'varying float vNodeIdOpposite;\n' + 'varying float vDist;\n' + 'varying float vIsSelected;\n' + 'varying float vIsHover;\n' + 'varying float vUseCrosshair;\n' + 'varying float vIstarget;\n' + 'vec2 get2Dfrom1D(float idx, float columns) {' + 'float n = idx/columns;' + 'float row = float(int(n));' + 'float col = fract(n)*columns;' + 'float ts = 1.0/columns;' + 'return vec2(ts*col, ts*row);' + '}' + "float getDigit(float planeId, float value) {           \t\t\n       \t\t    if(value >= 0.0 && planeId == 1.0) return 39.0;\n       \t\t    \n           \t\tif(value < 0.0 && planeId == 0.0) return 38.0;\n       \t\t    if(value < 0.0 && planeId == 2.0) return 39.0;\n       \t\t     \n       \t\t    float val = value;                         \n                float selectedDigit = 0.0;\n                for(int n=0; n <= 12; n++) {\n                    float num = floor(abs(val));\n                    if(num == 0.0) {                                \n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                        }\n                    } else if(num == 1.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -10.0 : 10.0;\n                        }                    \n                    } else if(num == 2.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -20.0 : 20.0;\n                        }                   \n                    } else if(num == 3.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -30.0 : 30.0;\n                        }                    \n                    } else if(num == 4.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -40.0 : 40.0;\n                        }                     \n                    } else if(num == 5.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -50.0 : 50.0;\n                        }                    \n                    } else if(num == 6.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -60.0 : 60.0;\n                        }                   \n                    } else if(num == 7.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -70.0 : 70.0;\n                        }                  \n                    } else if(num == 8.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -80.0 : 80.0;\n                        }                 \n                    } else if(num == 9.0) {\n                        if((value >= 0.0) && ((n == 0 && n == int(planeId)) || (n > 0 && n == int(planeId)-1))) {\n                            selectedDigit = num; break;\n                        } else if((value < 0.0) && ((n == 0 && n == int(planeId)-1) || (n > 0 && n == int(planeId)-2))) {\n                            selectedDigit = num; break;\n                        } else {\n                            val *= 10.0;\n                            val += (val >= 0.0) ? -90.0 : 90.0;\n                        }               \n                    }\n                }\n                selectedDigit = abs(floor(selectedDigit));\n                if(selectedDigit == 0.0) {\n                    return 28.0;\n                } else if(selectedDigit == 1.0) {\n                    return 29.0;\n                } else if(selectedDigit == 2.0) {\n                    return 30.0;\n                } else if(selectedDigit == 3.0) {\n                    return 31.0;\n                } else if(selectedDigit == 4.0) {\n                    return 32.0;\n                } else if(selectedDigit == 5.0) {\n                    return 33.0;\n                } else if(selectedDigit == 6.0) {\n                    return 34.0;\n                } else if(selectedDigit == 7.0) {\n                    return 35.0;\n                } else if(selectedDigit == 8.0) {\n                    return 36.0;\n                } else if(selectedDigit == 9.0) {\n                    return 37.0;\n                }\n            }" + 'mat4 lookAt(vec3 eye, vec3 center, vec3 up) {' + 'vec3 zaxis = normalize(center - eye);' + 'vec3 xaxis = normalize(cross(up, zaxis));' + 'vec3 yaxis = cross(zaxis, xaxis);' + 'mat4 matrix;' +
            //Column Major
            'matrix[0][0] = xaxis.x;' + 'matrix[1][0] = yaxis.x;' + 'matrix[2][0] = zaxis.x;' + 'matrix[3][0] = 0.0;' + 'matrix[0][1] = xaxis.y;' + 'matrix[1][1] = yaxis.y;' + 'matrix[2][1] = zaxis.y;' + 'matrix[3][1] = 0.0;' + 'matrix[0][2] = xaxis.z;' + 'matrix[1][2] = yaxis.z;' + 'matrix[2][2] = zaxis.z;' + 'matrix[3][2] = 0.0;' + 'matrix[0][3] = -dot(xaxis, eye);' + 'matrix[1][3] = -dot(yaxis, eye);' + 'matrix[2][3] = -dot(zaxis, eye);' + 'matrix[3][3] = 1.0;' + 'return matrix;' + '}' + 'mat4 transpose(mat4 m) {' + 'return mat4(  m[0][0], m[1][0], m[2][0], m[3][0],' + 'm[0][1], m[1][1], m[2][1], m[3][1],' + 'm[0][2], m[1][2], m[2][2], m[3][2],' + 'm[0][3], m[1][3], m[2][3], m[3][3]);' + '}' + 'mat4 rotationMatrix(vec3 axis, float angle) {' + 'axis = normalize(axis);' + 'float s = sin(angle);' + 'float c = cos(angle);' + 'float oc = 1.0 - c;' + 'return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,' + 'oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,' + 'oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,' + '0.0,                                0.0,                                0.0,                                1.0);' + '}' + Utils.degToRadGLSLFunctionString() + Utils.radToDegGLSLFunctionString() + Utils.cartesianToSphericalGLSLFunctionString() + Utils.sphericalToCartesianGLSLFunctionString() + Utils.getVectorGLSLFunctionString(),

            //██╗   ██╗███████╗██████╗ ████████╗███████╗██╗  ██╗    ███████╗ ██████╗ ██╗   ██╗██████╗  ██████╗███████╗
            //██║   ██║██╔════╝██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝
            //██║   ██║█████╗  ██████╔╝   ██║   █████╗   ╚███╔╝     ███████╗██║   ██║██║   ██║██████╔╝██║     █████╗
            //╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║   ██╔══╝   ██╔██╗     ╚════██║██║   ██║██║   ██║██╔══██╗██║     ██╔══╝
            // ╚████╔╝ ███████╗██║  ██║   ██║   ███████╗██╔╝ ██╗    ███████║╚██████╔╝╚██████╔╝██║  ██║╚██████╗███████╗
            //  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚══════╝
            "\n            mat4 nodepos = nodeWMatrix;\n            \n            vec4 nodeVertexPosition = nodeVertexPos[];\n            vec4 nodeVertexTex = nodeVertexTexture[];\n            \n            float nodeId = data[].x;            \n            vec2 xGeometryNode = get_global_id(nodeId, bufferNodesWidth, " + geometryLength.toFixed(1) + ("); " + "\n\n            vec4 currentPosition = posXYZW[xGeometryNode];\n            vec3 oppositePosition = vec3(0.0, 0.0, 0.0);\n            \n\n            float currentLineVertex = data[].z; " + "\n            float vertexCount = 1.0;\n                \n            vec4 nodeVertexColor = vec4(1.0, 1.0, 1.0, 1.0);\n            vec4 nodeVertexColorNetError = vec4(1.0, 1.0, 1.0, 1.0);\n            vVertexUV = vec2(-1.0, -1.0);\n\n            if(isNode == 1.0) {\n                float foutput = dataB[xGeometryNode].z;\n                float error = dataB[xGeometryNode].y;\n            \n                currentPosition += vec4(0.0, 0.1, 0.0, 1.0);\n\n                mat4 mm = rotationMatrix(vec3(1.0,0.0,0.0), (3.1416/2.0)*3.0);\n                nodepos = nodepos*mm;\n                float nodeImgId = nodeImgId[];\n\n                if(nodeImgId != -1.0) {\n                    vUseTex = 1.0;\n                    vVertexUV = get2Dfrom1D(nodeImgId, nodeImgColumns)+vec2(nodeVertexTexture.x/nodeImgColumns,nodeVertexTexture.y/nodeImgColumns);\n                }\n\n                nodeVertexColor = vec4(0.0, 0.0, 0.0, 1.0);\n                if(foutput > 0.0) {\n                    nodeVertexColor = vec4(abs(foutput), abs(foutput), abs(foutput), 1.0);\n                } else if(foutput < 0.0) {\n                    nodeVertexColor = vec4(abs(foutput), 0.0, 0.0, 1.0);\n                }\n\n                if(error > 0.0) {\n                    nodeVertexColorNetError = vec4(0.0, abs(error), 0.0, 1.0);\n                } else if(error < 0.0) {\n                    nodeVertexColorNetError = vec4(abs(error), 0.0, 0.0, 1.0);\n                } else {\n                    nodeVertexColorNetError = vec4(0.0,0.0,0.0, 0.0);\n                }\n\n                vIsSelected = (idToDrag == data[].x) ? 1.0 : 0.0;\n                vIsHover = (idToHover == data[].x) ? 1.0 : 0.0;\n            }\n            if(isLink == 1.0) {        \n                float foutput = dataB[xGeometryNode].z;\n                float error = dataB[xGeometryNode].w;\n                \n                float nodeIdOpposite = data[].y;\n                \n                vec2 xGeometryNode_opposite = get_global_id(nodeIdOpposite, bufferNodesWidth, ") + geometryLength.toFixed(1) + (");\n                vec3 oppositePosition = posXYZW[xGeometryNode_opposite].xyz;\n                \n                " + VFP_NODE.linkStr() + "\n            \n                currentPosition += vec4(dirN*(lineIncrements*currentLineVertex), 1.0); " + "\n\n                vIsSelected = (idToDrag == data[].x || idToDrag == data[].y) ? 1.0 : 0.0;\n                vIsHover = (idToHover == data[].x || idToHover == data[].y) ? 1.0 : 0.0;\n                \n                nodeVertexColor = vec4(abs(foutput), abs(foutput), abs(foutput), 1.0);\n\n                if(error > 0.0) {\n                    nodeVertexColorNetError = vec4(0.0, abs(error), 0.0, 1.0);\n                } else if(error < 0.0) {\n                    nodeVertexColorNetError = vec4(abs(error), 0.0, 0.0, 1.0);\n                } else {\n                    nodeVertexColorNetError = vec4(0.0,0.0,0.0, 0.0);\n                }\n                        \n                vNodeIdOpposite = nodeIdOpposite;\n            }\n            if(isArrow == 1.0) {\n                float nodeIdOpposite = data[].y;\n                \n                vec2 xGeometryNode_opposite = get_global_id(nodeIdOpposite, bufferNodesWidth, ") + geometryLength.toFixed(1) + (");\n                vec3 oppositePosition = posXYZW[xGeometryNode_opposite].xyz;\n                \n                " + VFP_NODE.linkStr() + "                \n            \n                vec3 currentPositionTMP;\n                " + "\n                float currentLineVertexU = vertexCount-1.0;\n                currentPositionTMP = oppositePosition+((dirN*-1.0)*(lineIncrements*currentLineVertexU));\n\n                mat4 pp = lookAt(currentPositionTMP, vec3(currentPosition.x, currentPosition.y, currentPosition.z), vec3(0.0, 1.0, 0.0));\n                pp = transpose(pp);\n                nodepos[0][0] = pp[0][0];\n                nodepos[0][1] = pp[1][0];\n                nodepos[0][2] = pp[2][0];\n\n                nodepos[1][0] = pp[0][1];\n                nodepos[1][1] = pp[1][1];\n                nodepos[1][2] = pp[2][1];\n\n                nodepos[2][0] = pp[0][2];\n                nodepos[2][1] = pp[1][2];\n                nodepos[2][2] = pp[2][2];\n\n                " + "\n                dir = currentPositionTMP-vec3(currentPosition.x, currentPosition.y, currentPosition.z);\n                currentPosition += vec4(normalize(dir),1.0)*2.0;\n\n                vIsSelected = (idToDrag == data[].x || idToDrag == data[].y) ? 1.0 : 0.0;\n                vIsHover = (idToHover == data[].x || idToHover == data[].y) ? 1.0 : 0.0;\n                \n                vNodeIdOpposite = nodeIdOpposite;\n            }\n            if(isNodeText == 1.0) {\n                float foutput = dataB[xGeometryNode].z;\n                \n                float letId = (showValues == 1.0) ? getDigit(data[].y, foutput) : letterId[];\n                mat4 mm = rotationMatrix(vec3(1.0,0.0,0.0), (3.1416/2.0)*3.0);\n                nodepos = nodepos*mm;\n\n                vVertexUV = get2Dfrom1D(letId, fontImgColumns)+vec2(nodeVertexTexture.x/fontImgColumns,nodeVertexTexture.y/fontImgColumns);\n                nodeVertexPosition = vec4(nodeVertexPosition.x*0.1, nodeVertexPosition.y*0.1, nodeVertexPosition.z*0.1, 1.0);\n                currentPosition.x += 3.5;\n                currentPosition.y += 0.5;\n\n                vIsSelected = (idToDrag == data[].x) ? 1.0 : 0.0;\n                vIsHover = (idToHover == data[].x) ? 1.0 : 0.0;\n                \n                vIstarget = (currentLineVertex == 1.0) ? 1.0 : 0.0;\n            }\n            nodepos[3][0] = currentPosition.x;\n            nodepos[3][1] = currentPosition.y;\n            nodepos[3][2] = currentPosition.z;\n\n            mat4 nodeposG = nodeWMatrix;\n            vWNMatrix = nodeposG * nodeVertexNormal[];\n\n            vUseCrosshair = 0.0;\n            \n\n            " + customCode + "\n            \n            vDist = max(0.3, 1.0-(distance(currentPosition.xyz, oppositePosition)*0.01)); " + "\n            vVertexColor = nodeVertexColor;\n            vVertexColorNetError = nodeVertexColorNetError;\n            vUV = nodeVertexTexture;\n            vNodeId = nodeId;\n\n\n            gl_Position = PMatrix * cameraWMatrix * nodepos * nodeVertexPosition;"),

            //███████╗██████╗  █████╗  ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗    ██╗  ██╗███████╗ █████╗ ██████╗
            //██╔════╝██╔══██╗██╔══██╗██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝    ██║  ██║██╔════╝██╔══██╗██╔══██╗
            //█████╗  ██████╔╝███████║██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║       ███████║█████╗  ███████║██║  ██║
            //██╔══╝  ██╔══██╗██╔══██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║       ██╔══██║██╔══╝  ██╔══██║██║  ██║
            //██║     ██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║       ██║  ██║███████╗██║  ██║██████╔╝
            //╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝
            "varying vec4 vVertexColor;\n            varying vec4 vVertexColorNetError;\n            varying vec4 vUV;\n       \t\tvarying vec2 vVertexUV;\n       \t\tvarying float vUseTex;\n       \t\tvarying vec4 vWNMatrix;\n            varying float vNodeId;\n            varying float vNodeIdOpposite;\n       \t\tvarying float vDist;\n       \t\tvarying float vIsSelected;\n            varying float vIsHover;\n       \t\tvarying float vUseCrosshair;\n       \t\tvarying float vIstarget;",

            //███████╗██████╗  █████╗  ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗    ███████╗ ██████╗ ██╗   ██╗██████╗  ██████╗███████╗
            //██╔════╝██╔══██╗██╔══██╗██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝    ██╔════╝██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝
            //█████╗  ██████╔╝███████║██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║       ███████╗██║   ██║██║   ██║██████╔╝██║     █████╗
            //██╔══╝  ██╔══██╗██╔══██║██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║       ╚════██║██║   ██║██║   ██║██╔══██╗██║     ██╔══╝
            //██║     ██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║       ███████║╚██████╔╝╚██████╔╝██║  ██║╚██████╗███████╗
            //╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚══════╝
            'vec4 color = vVertexColor;\n' + 'vec4 colorB = vVertexColorNetError;\n' + 'vec4 colorOrange = vec4(255.0/255.0, 131.0/255.0, 0.0/255.0, 1.0);' + 'vec4 colorOrangeDark = vec4(255.0/255.0, 80.0/255.0, 0.0/255.0, 1.0);' + 'vec4 colorPurple = vec4(132.0/255.0, 0.0/255.0, 255.0/255.0, 1.0);' + 'vec4 fcolor = vec4(0.0, 0.0, 0.0, 1.0);' + 'if(isNode == 1.0) {' + 'if(vUseTex == 1.0) {' + 'vec4 tex;' + 'if(vUseCrosshair == 1.0) {' + 'tex = nodesImgCrosshair[vVertexUV.xy];' + '} else if(vIsSelected == 1.0) {' + 'color = colorOrangeDark;' + 'tex = nodesImgCrosshair[vVertexUV.xy];' + '} else if(vIsHover == 1.0) {' + 'color = colorPurple;' + 'tex = nodesImg[vVertexUV.xy];' + '} else {' + 'tex = nodesImg[vVertexUV.xy];' + '}' + 'color = ' + VFP_NODE.nodesDrawMode(geometryLength) + ';\n' + '}' + 'if(color.a < 0.1) discard;' + 'fcolor = color;\n' +
            // half up: ouput; half down: error
            'if(vUV.y > (0.7)) ' + 'fcolor = colorB;\n' + // 0.75-(colorB.w*1.0)   'fcolor = (vUV.y < (0.5)) ? color : ((colorB.w == 0.0)?color:colorB);\n'+
            '} else if(isLink == 1.0) {' + 'if(vIsSelected == 1.0) ' + 'color = colorOrange;' + 'else if(vIsHover == 1.0) ' + 'color = colorPurple;' +

            // weight color
            'vec2 xAdjMatCurrent = get_global_id(vec2(vNodeIdOpposite, vNodeId), widthAdjMatrix);' + 'vec4 pixAdjMatACurrent = adjacencyMatrix[xAdjMatCurrent];\n' +

            // x weight
            'if(pixAdjMatACurrent.z > 0.0) ' + 'fcolor = vec4(0.0, pixAdjMatACurrent.z, 0.0, 1.0);\n' + 'else ' + 'fcolor = vec4(abs(pixAdjMatACurrent.z), 0.0, 0.0, 1.0);\n' +

            // x output
            'if(multiplyOutput == 1.0) ' + 'fcolor *= vec4(color.xyz, 1.0);\n' + '} else if(isArrow == 1.0) {' + 'if(vIstarget == 1.0) {' + 'if(vIsSelected == 1.0) {' + 'color = vec4(colorOrange.rgb*vDist, 1.0);\n' + '} else if(vIsHover == 1.0) {' + 'color = vec4(colorPurple.rgb*vDist, 1.0);\n' + '}' + '} else {' + 'color = vec4(1.0, 0.0, 0.0, 0.0);' + '}' + 'fcolor = color;\n' + '} else if(isNodeText == 1.0) {' + 'fcolor = fontsImg[vVertexUV.xy];\n' + '}' + 'return [fcolor];'];
        }
    }, {
        key: "linkStr",
        value: function linkStr() {
            return "\n\t    vec3 dir = oppositePosition-vec3(currentPosition.x, currentPosition.y, currentPosition.z);\n        float dist = length(dir);\n        float lineIncrements = dist/vertexCount;\n        vec3 dirN = normalize(dir);\n\n        float repeatId = data[].w;\n        vec3 cr = cross(dirN, vec3(0.0, 1.0, 0.0));\n        float currentLineVertexSQRT = abs( currentLineVertex-(vertexCount/2.0) )/(vertexCount/2.0);\n        currentLineVertexSQRT = sqrt(1.0-currentLineVertexSQRT);";
        }
    }, {
        key: "nodesDrawMode",
        value: function nodesDrawMode(geometryLength) {
            if (geometryLength === 1) return "vec4(color.rgb, 1.0)";else return "vec4(tex.rgb*color.rgb, tex.a)";
        }
    }]);

    return VFP_NODE;
}();

global.VFP_NODE = VFP_NODE;
module.exports.VFP_NODE = VFP_NODE;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"scejs":4}],14:[function(require,module,exports){
(function (global){
'use strict';

Object.defineProperty(exports, "__esModule", {
            value: true
});
exports.VFP_NODEPICKDRAG = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

require('scejs');

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var VFP_NODEPICKDRAG = exports.VFP_NODEPICKDRAG = function () {
            function VFP_NODEPICKDRAG() {
                        _classCallCheck(this, VFP_NODEPICKDRAG);
            }

            _createClass(VFP_NODEPICKDRAG, null, [{
                        key: 'getSrc',
                        value: function getSrc(geometryLength) {
                                    return [[undefined],
                                    // vertex head
                                    'varying vec4 vColor;\n' +
                                    //'uniform sampler2D posXYZW;\n'+
                                    Utils.packGLSLFunctionString() + 'mat4 rotationMatrix(vec3 axis, float angle) {' + 'axis = normalize(axis);' + 'float s = sin(angle);' + 'float c = cos(angle);' + 'float oc = 1.0 - c;' + 'return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,' + 'oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,' + 'oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,' + '0.0,                                0.0,                                0.0,                                1.0);' + '}',

                                    // vertex source
                                    'vec2 xGeometry = get_global_id(data[].x, uBufferWidth, ' + geometryLength.toFixed(1) + ');' + 'vec4 nodePosition = posXYZW[xGeometry];\n' + 'mat4 nodepos = nodeWMatrix;' + 'mat4 mm = rotationMatrix(vec3(1.0,0.0,0.0), (3.1416/2.0)*3.0);' + 'nodepos = nodepos*mm;' + 'nodepos[3][0] = nodePosition.x;' + 'nodepos[3][1] = nodePosition.y;' + 'nodepos[3][2] = nodePosition.z;' + 'vColor = pack((data[].x+1.0)/1000000.0);' + 'if(vColor.x == 1.0 && vColor.y == 1.0 && vColor.z == 1.0 && vColor.w == 1.0) ' + 'nodepos[3][0] = 10000.0;' + 'gl_Position = PMatrix * cameraWMatrix * nodepos * nodeVertexPos[];\n' + 'gl_PointSize = 10.0;\n',

                                    // fragment head
                                    'varying vec4 vColor;\n',

                                    // fragment source
                                    //'vec2 x = get_global_id();'+ // no needed

                                    'return [vColor];\n'];
                        }
            }]);

            return VFP_NODEPICKDRAG;
}();

global.VFP_NODEPICKDRAG = VFP_NODEPICKDRAG;
module.exports.VFP_NODEPICKDRAG = VFP_NODEPICKDRAG;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"scejs":4}],15:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.GBrainRL = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _gbrain = require("./gbrain");

var _Plot = require("./Plot.class");

var _AvgWin = require("./AvgWin.class");

var _draggabilly = require("draggabilly");

var _draggabilly2 = _interopRequireDefault(_draggabilly);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

/**
 * ConvNetJS Reinforcement Learning Module (https://github.com/karpathy/convnetjs)
 */
/**
 * @class
 * @param {Object} jsonIn
 * @param {HTMLElement} jsonIn.target
 * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
 * @param {int} jsonIn.num_inputs
 * @param {int} jsonIn.num_actions
 * @param {int} jsonIn.temporal_window
 * @param {int} jsonIn.experience_size
 * @param {number} jsonIn.gamma
 * @param {number} jsonIn.epsilon_min
 * @param {number} jsonIn.epsilon_test_time
 * @param {int} jsonIn.start_learn_threshold
 * @param {int} jsonIn.batch_repeats
 * @param {int} jsonIn.learning_steps_total
 * @param {int} jsonIn.learning_steps_burnin
 * @param {Array<Object>} jsonIn.layer_defs
 */

var GBrainRL = exports.GBrainRL = function () {
    function GBrainRL(jsonIn) {
        var _this = this;

        _classCallCheck(this, GBrainRL);

        this.gbrain = null;

        this.num_inputs = jsonIn.num_inputs;
        this.num_actions = jsonIn.num_actions;
        this.temporal_window = jsonIn.temporal_window;
        this.experience_size = jsonIn.experience_size;
        this.gamma = jsonIn.gamma;
        this.epsilon_min = jsonIn.epsilon_min;
        this.epsilon_test_time = jsonIn.epsilon_test_time;
        this.start_learn_threshold = jsonIn.start_learn_threshold;
        this.learning_steps_total = jsonIn.learning_steps_total;
        this.learning_steps_burnin = jsonIn.learning_steps_burnin;

        this.net_inputs = this.num_inputs * this.temporal_window + this.num_actions * this.temporal_window + this.num_inputs;
        this.window_size = Math.max(this.temporal_window, 2);

        this.windows = [];
        for (var n = 0; n < 7; n++) {
            this.windows[n] = {};
            this.windows[n].state_window = new Array(this.window_size);
            this.windows[n].action_window = new Array(this.window_size);
            this.windows[n].reward_window = new Array(this.window_size);
            this.windows[n].net_window = new Array(this.window_size);
        }

        this.experience = [];

        this.onLearned = null;

        this.age = 0;
        this.ageEpoch = 0;
        this.epoch = 0;
        this.epsilon = 1.0;
        this.loss = 0.0;

        this.latest_reward = 0;
        this.last_input_array = null;
        this.forward_passes = 0;
        this.learning = true;

        this.sweep = 0;
        this.sweepMax = 200;
        this.sweepDir = 0;
        this.sweepEnable = false;
        this.plotEnable = true;

        this.arrInputs = [];
        this.arrTargets = [];

        this.lastTotalError = 0;

        this.showOutputWeighted = false;
        this.showWD = false;
        this.showValues = true;

        this.windowSavedPosLeft = 0;
        this.windowSavedPosTop = 0;
        this.windowEnabled = true;

        var target = document.createElement("div");
        document.getElementsByTagName("body")[0].appendChild(target);
        target.style.width = "950px";
        target.innerHTML = "\n        <div style=\"font-size:12px; box-shadow:rgba(0, 0, 0, 0.683594) 3px 3px 8px 1px,rgb(255, 255, 255) 0 0 5px 0 inset; border-radius:5px;\">\n            <div id=\"elGbrainWindowHandle\" style=\"border-top-left-radius:5px; border-top-right-radius:5px; width:100%; background:rgba(200,200,200,0.7); cursor:move;\tdisplay:table;\">\n                <div style=\"padding-left:5px;color:#000;font-weight:bold;display:table-cell;vertical-align:middle;\">GBrain</div>\n                <div style=\"width:22px;\tpadding:2px; display:table-cell; vertical-align:middle;\">\n                    <div id=\"elGbrainMinMax\" style=\"font-weight:bold;cursor:pointer;\">&#95;</div>\n                </div>\n            </div>\n            <div id=\"elGbrainContent\" style=\"border-bottom-left-radius:5px; border-bottom-right-radius:5px; min-width:220px;\tcursor:default;\tpadding:5px; color:#FFF; background:rgba(50,50,50,0.95); overflow-y:auto;\">\n                <div style=\"display:inline-block;width:400px;\">\n                    Loss\n                    <canvas id=\"elPlotLoss\" style=\"background:#FFF\"></canvas><br />\n                    Epsilon\n                    <canvas id=\"elPlotEpsilon\" style=\"background:#FFF\"></canvas><br />\n                    <button id=\"BTNID_PLOTMODE\" style=\"display:inline-block;\">Plot mode</button>\n                    <button id=\"BTNID_PLOTENABLE\" style=\"display:inline-block;\">Enable plot</button>\n                    <div id=\"el_info\"></div>\n                    <div>\n                        Show weight*neuron output<input title=\"weight*output\" type=\"checkbox\" id=\"elem_enableOutputWeighted\"/><br />\n                        Show weight dynamics<input title=\"weight dynamics\" type=\"checkbox\" id=\"elem_enableWeightDynamics\"/><br />\n                        Show output values<input title=\"input values\" type=\"checkbox\" checked=\"checked\" id=\"elem_enableShowValues\"/>\n                    </div>\n                    <button id=\"BTNID_SWEEPEPSILON\" style=\"display:inline-block;\">Sweep*reward epsilon</button>\n                    <button id=\"BTNID_STOP\" style=\"display:inline-block;\">Stop train</button>\n                    <button id=\"BTNID_RESUME\" style=\"display:inline-block;\">Resume train</button>\n                    <button id=\"BTNID_TOJSON\" style=\"display:inline-block;\">Output model in console</button>\n                    <button id=\"BTNID_TOLSJSON\" style=\"display:inline-block;\">Save model in LocalStorage</button>\n                    <button id=\"BTNID_FROMLSJSON\" style=\"display:inline-block;\">Load model from LocalStorage</button>\n                </div>\n                <div style=\"display:inline-block;\">\n                    <div id=\"el_gbrainDisplay\"></div>\n                </div>\n            </div>\n        </div>\n        ";
        this.el_info = target.querySelector("#el_info");

        target.querySelector("#BTNID_PLOTMODE").addEventListener("click", function () {
            _this.plotLoss.currentMode = _this.plotLoss.currentMode === 0 ? 1 : 0;
            _this.plotEpsilon.currentMode = _this.plotEpsilon.currentMode === 0 ? 1 : 0;
        });
        target.querySelector("#BTNID_PLOTENABLE").addEventListener("click", function () {
            _this.plotEnable = _this.plotEnable !== true;
        });
        target.querySelector("#elem_enableOutputWeighted").addEventListener("click", function () {
            _this.showOutputWeighted === false ? _this.gbrain.enableShowOutputWeighted() : _this.gbrain.disableShowOutputWeighted();
            _this.showOutputWeighted = !_this.showOutputWeighted;
        });
        target.querySelector("#elem_enableWeightDynamics").addEventListener("click", function () {
            _this.showWD === false ? _this.gbrain.enableShowWeightDynamics() : _this.gbrain.disableShowWeightDynamics();
            _this.showWD = !_this.showWD;
        });
        target.querySelector("#elem_enableShowValues").addEventListener("click", function () {
            _this.showValues === false ? _this.gbrain.enableShowValues() : _this.gbrain.disableShowValues();
            _this.showValues = !_this.showValues;
        });

        target.querySelector("#BTNID_SWEEPEPSILON").addEventListener("click", function () {
            _this.sweepEnable = _this.sweepEnable !== true;
        });

        target.querySelector("#BTNID_STOP").addEventListener("click", function () {
            _this.stopLearning();
        });

        target.querySelector("#BTNID_RESUME").addEventListener("click", function () {
            _this.resumeLearning();
        });

        target.querySelector("#BTNID_TOJSON").addEventListener("click", function () {
            _this.toJson();
        });

        target.querySelector("#BTNID_TOLSJSON").addEventListener("click", function () {
            localStorage.trainedModel = _this.toJson();
        });
        target.querySelector("#BTNID_FROMLSJSON").addEventListener("click", function () {
            _this.fromJson(JSON.parse(localStorage.trainedModel));
        });

        target.querySelector("#elGbrainMinMax").addEventListener("click", function () {
            if (_this.windowEnabled === true) {
                _this.windowSavedPosLeft = target.style.left;
                _this.windowSavedPosTop = target.style.top;
                target.style.position = "absolute";
                target.style.left = "50";
                target.style.top = "0";
                target.style.width = "150px";
                target.querySelector("#elGbrainContent").style.display = "none";
                target.querySelector("#elGbrainMinMax").innerHTML = "&square;";
                _this.windowEnabled = false;
            } else {
                target.style.position = "relative";
                target.style.left = _this.windowSavedPosLeft;
                target.style.top = _this.windowSavedPosTop;
                target.style.width = "950px";
                target.querySelector("#elGbrainContent").style.display = "block";
                target.querySelector("#elGbrainMinMax").innerHTML = "&#95;";
                _this.windowEnabled = true;
            }
        });

        var dragg = new _draggabilly2.default(target, {
            handle: '#elGbrainWindowHandle'
        });
        target.style.left = -target.getBoundingClientRect().left + 100 + "px";
        target.style.top = -target.getBoundingClientRect().top + 100 + "px";

        this.avgLossWin = new _AvgWin.AvgWin();

        this.plotLoss = new _Plot.Plot();
        this.plotLossCanvas = target.querySelector("#elPlotLoss");

        this.plotEpsilon = new _Plot.Plot();
        this.plotEpsilonCanvas = target.querySelector("#elPlotEpsilon");

        this.clock = 0;

        if (jsonIn.layer_defs !== undefined && jsonIn.layer_defs !== null) {
            this.gbrain = new _gbrain.GBrain({ "target": target.querySelector("#el_gbrainDisplay"),
                "dimensions": { "width": 500, "height": 500 },
                "batch_repeats": jsonIn.batch_repeats,
                "learning_rate": jsonIn.learning_rate });
            this.gbrain.makeLayers(jsonIn.layer_defs);
        }
    }

    _createClass(GBrainRL, [{
        key: "fromJson",
        value: function fromJson(jsonIn) {
            this.gbrain.fromJson(jsonIn);
        }
    }, {
        key: "toJson",


        /**
         * @returns {String}
         */
        value: function toJson() {
            return this.gbrain.toJson();
        }
    }, {
        key: "getNetInput",
        value: function getNetInput(iId, xt) {
            // return s = (x,a,x,a,x,a,xt) state vector.
            // It's a concatenation of last window_size (x,a) pairs and current state x
            var w = [];
            w = w.concat(xt); // start with current state
            // and now go backwards and append states and actions from history temporal_window times
            var n = this.window_size;
            for (var k = 0; k < this.temporal_window; k++) {
                // state
                w = w.concat(this.windows[iId].state_window[n - 1 - k]);
                // action, encoded as 1-of-k indicator vector. We scale it up a bit because
                // we dont want weight regularization to undervalue this information, as it only exists once
                var action1ofk = new Array(this.num_actions);
                for (var q = 0; q < this.num_actions; q++) {
                    action1ofk[q] = 0.0;
                }action1ofk[this.windows[iId].action_window[n - 1 - k]] = 1.0 /* *this.num_inputs*/;
                w = w.concat(action1ofk);
            }
            return w;
        }
    }, {
        key: "random_action",
        value: function random_action() {
            return Math.floor(Math.random() * this.num_actions);
        }
    }, {
        key: "policy",
        value: function policy(s, onP) {
            // compute the value of doing any action in this state
            // and return the argmax action and its value
            this.gbrain.forward(s, function (maxacts) {
                onP(maxacts);
            });
        }
    }, {
        key: "pushWindow",
        value: function pushWindow(iId, input_array, net_input, action) {
            this.windows[iId].state_window.shift();
            this.windows[iId].state_window.push(input_array);

            this.windows[iId].net_window.shift();
            this.windows[iId].net_window.push(net_input);

            this.windows[iId].action_window.shift();
            this.windows[iId].action_window.push(action);
        }
    }, {
        key: "stopLearning",
        value: function stopLearning() {
            this.learning = false;
            this.forward_passes = 0;

            for (var n = 0; n < 7; n++) {
                this.windows[n] = {};
                this.windows[n].state_window = new Array(this.window_size);
                this.windows[n].action_window = new Array(this.window_size);
                this.windows[n].reward_window = new Array(this.window_size);
                this.windows[n].net_window = new Array(this.window_size);
            }
        }
    }, {
        key: "resumeLearning",
        value: function resumeLearning() {
            this.learning = true;
            this.forward_passes = 0;

            for (var n = 0; n < 7; n++) {
                this.windows[n] = {};
                this.windows[n].state_window = new Array(this.window_size);
                this.windows[n].action_window = new Array(this.window_size);
                this.windows[n].reward_window = new Array(this.window_size);
                this.windows[n].net_window = new Array(this.window_size);
            }
        }
    }, {
        key: "forward",
        value: function forward(input_array, onAction) {
            var _this2 = this;

            if (this.learning === true) {
                this.epsilon = Math.min(1.0, Math.max(this.epsilon_min, 1.0 - (this.age - this.learning_steps_burnin) / (this.learning_steps_total - this.learning_steps_burnin)));
                if (this.sweepEnable === true) {
                    if (this.sweep >= this.sweepMax) this.sweepDir = -1;else if (this.sweep <= 0) this.sweepDir = 1;
                    this.sweep += this.sweepDir;
                    if (this.latest_reward > 0) {
                        var rewardMultiplier = 1.0 - Math.min(1, Math.max(0.0, this.latest_reward * 2));
                        var sweepMultiplier = Math.abs(this.sweep) / this.sweepMax;
                        this.epsilon = Math.max(this.epsilon_min, rewardMultiplier * sweepMultiplier * this.epsilon);
                    }
                }
            } else this.epsilon = this.epsilon_test_time;

            this.forward_passes++;
            this.last_input_array = input_array;

            var entNetInput = [];
            var windowsTemp = [];
            for (var n = 0; n < input_array.length; n++) {
                windowsTemp[n] = { input_array: input_array[n],
                    net_input: null,
                    action: null };
                windowsTemp[n].net_input = this.getNetInput(n, windowsTemp[n].input_array);

                for (var nb = 0; nb < windowsTemp[n].net_input.length; nb++) {
                    entNetInput.push(windowsTemp[n].net_input[nb]);
                }
            }
            if (this.forward_passes > this.temporal_window) {
                var rf = Math.random();
                if (rf < this.epsilon) {
                    var retActs = [];
                    for (var _n = 0; _n < input_array.length; _n++) {
                        windowsTemp[_n].action = this.random_action();
                        this.pushWindow(_n, windowsTemp[_n].input_array, windowsTemp[_n].net_input, windowsTemp[_n].action);
                        retActs.push(windowsTemp[_n].action);
                    }
                    onAction(retActs);
                } else {
                    this.policy(entNetInput, function (maxact) {
                        var retActs = [];
                        for (var _n2 = 0; _n2 < input_array.length; _n2++) {
                            windowsTemp[_n2].action = maxact[_n2].action;
                            _this2.pushWindow(_n2, windowsTemp[_n2].input_array, windowsTemp[_n2].net_input, windowsTemp[_n2].action);
                            retActs.push(windowsTemp[_n2].action);
                        }
                        onAction(retActs);
                    });
                }
            } else {
                var _retActs = [];
                for (var _n3 = 0; _n3 < input_array.length; _n3++) {
                    windowsTemp[_n3].action = this.random_action();
                    this.pushWindow(_n3, windowsTemp[_n3].input_array, windowsTemp[_n3].net_input, windowsTemp[_n3].action);
                    _retActs.push(windowsTemp[_n3].action);
                }
                onAction(_retActs);
            }
        }
    }, {
        key: "backward",
        value: function backward(reward, _onLearned) {
            var _this3 = this;

            this.onLearned = _onLearned;
            this.latest_reward = reward;

            this.clock++;
            this.el_info.innerHTML = "epsilon: " + this.epsilon + "<br />" + "reward: " + this.latest_reward + "<br />" + "age: " + this.age + "<br />" + "average Q-learning loss: " + this.loss + "<br />" + "current learning rate: " + this.gbrain.currentLearningRate;

            if (this.learning === false) {
                this.onLearned();
            } else {
                //this.average_reward_window.add(reward); TODO
                this.windows[0].reward_window.shift();
                this.windows[0].reward_window.push(reward);

                if (this.ageEpoch === this.experience_size) {
                    this.epoch++;
                    this.ageEpoch = 0;
                    var dec_rate = 1.0;
                    //this.gbrain.setLearningRate((1.0/(1.0+dec_rate*this.epoch))*this.gbrain.initialLearningRate);
                }
                this.age++;
                this.ageEpoch++;

                // it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
                // (given that an appropriate number of state measurements already exist, of course)
                if (this.forward_passes > this.temporal_window + 1) {
                    var n = this.window_size;
                    var e = { "state0": this.windows[0].net_window[n - 2],
                        "action0": this.windows[0].action_window[n - 2],
                        "reward0": this.windows[0].reward_window[n - 2],
                        "state1": this.windows[0].net_window[n - 1] };
                    if (this.experience.length < this.experience_size) this.experience.push(e);else {
                        var r = Math.floor(Math.random() * this.experience.length);
                        this.experience[r] = e;
                    }

                    if (this.experience.length > this.start_learn_threshold) {
                        var bEntries = [];
                        var state1_entries = [];

                        var _e = this.experience[this.experience.length - 1];
                        bEntries.push(_e);
                        for (var nb = 0; nb < _e.state1.length; nb++) {
                            state1_entries.push(_e.state1[nb]);
                        }for (var _n4 = 1; _n4 < this.gbrain.graph.batch_repeats * this.gbrain.graph.gpu_batch_size; _n4++) {
                            var _e2 = this.experience[Math.floor(Math.random() * this.experience.length)];
                            bEntries.push(_e2);
                            for (var _nb = 0; _nb < _e2.state1.length; _nb++) {
                                state1_entries.push(_e2.state1[_nb]);
                            }
                        }
                        this.policy(state1_entries, function (maxact) {
                            _this3.arrInputs = [];
                            _this3.arrTargets = [];

                            for (var _n5 = 0; _n5 < _this3.gbrain.graph.batch_repeats * _this3.gbrain.graph.gpu_batch_size; _n5++) {
                                var _r = bEntries[_n5].reward0 + _this3.gamma * maxact[_n5].value;
                                var ystruct = { dim: bEntries[_n5].action0, val: _r };

                                _this3.arrTargets.push(ystruct);

                                for (var _nb2 = 0; _nb2 < bEntries[_n5].state0.length; _nb2++) {
                                    _this3.arrInputs.push(bEntries[_n5].state0[_nb2]);
                                }
                            }

                            _this3.gbrain.forward(_this3.arrInputs, function (data) {
                                _this3.gbrain.train(_this3.arrTargets, function (loss) {

                                    _this3.loss = loss / (_this3.gbrain.graph.batch_repeats * _this3.gbrain.graph.gpu_batch_size);
                                    _this3.avgLossWin.add(Math.min(10.0, _this3.loss));

                                    _this3.plotLoss.add(_this3.clock, _this3.avgLossWin.get_average());
                                    if (_this3.plotEnable === true) _this3.plotLoss.drawSelf(_this3.plotLossCanvas);

                                    _this3.plotEpsilon.add(_this3.clock, _this3.epsilon);
                                    if (_this3.plotEnable === true) _this3.plotEpsilon.drawSelf(_this3.plotEpsilonCanvas);

                                    _this3.onLearned(_this3.loss);
                                });
                            }, false);
                        });
                    } else this.onLearned();
                } else this.onLearned();
            }
        }
    }]);

    return GBrainRL;
}();

global.GBrainRL = GBrainRL;
module.exports.GBrainRL = GBrainRL;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./AvgWin.class":7,"./Plot.class":11,"./gbrain":16,"draggabilly":1}],16:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.GBrain = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

require("scejs");

var _Graph = require("./Graph.class");

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

/**
 * @class
 * @param {Object} jsonIn
 * @param {HTMLElement} jsonIn.target
 * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
 * @param {int} jsonIn.batch_repeats
 */
var GBrain = exports.GBrain = function () {
    function GBrain(jsonIn) {
        _classCallCheck(this, GBrain);

        this.project = null;
        this.graph = null;

        this.ini(jsonIn);
    }

    /**
     * @param {Object} jsonIn
     * @param {HTMLElement} jsonIn.target
     * @param {Object} [jsonIn.dimensions={width: Int, height: Int}]
     * @param {int} jsonIn.batch_repeats
     * @param {number} jsonIn.learning_rate
     * @param {WebGLRenderingContext} [jsonIn.gl=undefined]
     */


    _createClass(GBrain, [{
        key: "ini",
        value: function ini(jsonIn) {
            this.sce = new SCE();
            this.sce.initialize(jsonIn);

            this.project = new Project();
            this.sce.loadProject(this.project);

            var stage = new Stage();
            this.project.addStage(stage);
            this.project.setActiveStage(stage);

            // CAMERA
            var simpleCamera = new SimpleCamera(this.sce);
            simpleCamera.setView(Constants.VIEW_TYPES.TOP);
            simpleCamera.setVelocity(1.0);
            this.sce.setDimensions(jsonIn.dimensions.width, jsonIn.dimensions.height);

            // GRID
            //let grid = new Grid(this.sce);
            //grid.generate(100.0, 1.0);

            this.graph = new _Graph.Graph(this.sce, { "enableFonts": true });
            this.graph.enableNeuronalNetwork();
            this.graph.layerCount = 0;
            this.outputCount = 0;
            this.layerNodes = [];
            this.graph.batch_repeats = jsonIn.batch_repeats;
            this.initialLearningRate = jsonIn.learning_rate;
            this.currentLearningRate = jsonIn.learning_rate;

            var mesh_point = new Mesh().loadPoint();
            //this.graph.setNodeMesh(mesh_point);
        }
    }, {
        key: "makeLayers",


        /**
         * @param {Array<Object>} layer_defs
         */
        value: function makeLayers(layer_defs) {
            var _this = this;

            var ml = function ml(w, h, isInput, type, posZ) {
                _this.graph.layer_defs[_this.graph.layerCount].hasBias = _this.graph.layer_defs[_this.graph.layerCount + 1].activation === "relu" || _this.graph.layer_defs[_this.graph.layerCount + 1].type === "regression" ? 1.0 : 0.0;

                if (type === "conv") {
                    w -= 2;
                    h -= 2;
                    _this.graph.layer_defs[_this.graph.layerCount].out_sx = w;
                    _this.graph.layer_defs[_this.graph.layerCount].out_sy = h;
                }

                return _this.graph.createNeuronLayer(w, h, [_this.offsetX, 0.0, posZ, 1.0], 5.0, _this.graph.layer_defs[_this.graph.layerCount].hasBias, isInput);
            };

            var mr = function mr(w, originLayer, targetLayer, weights, type, convMatrixId) {
                if (type === "fc") {
                    _this.graph.connectNeuronLayerWithNeuronLayer({ "neuronLayerOrigin": originLayer,
                        "neuronLayerTarget": targetLayer,
                        "weights": weights !== undefined && weights !== null ? weights : null,
                        "layer_neurons_count": _this.layerNodes[_this.layerNodes.length - 1].length,
                        "layerNum": _this.graph.layerCount - 1,
                        "hasBias": _this.graph.layer_defs[_this.graph.layerCount].hasBias }); // TODO l.activation
                } else if (type === "conv") {
                    _this.graph.connectConvXYNeuronsFromXYNeurons({ "w": w,
                        "neuronLayerOrigin": originLayer,
                        "neuronLayerTarget": targetLayer,
                        "weights": weights !== undefined && weights !== null ? weights : null,
                        "layer_neurons_count": _this.layerNodes[_this.layerNodes.length - 1].length,
                        "layerNum": _this.graph.layerCount - 1,
                        "hasBias": _this.graph.layer_defs[_this.graph.layerCount].hasBias,
                        "convMatrixId": convMatrixId });
                }
            };

            this.graph.layer_defs = layer_defs;
            this.offsetX = 0;

            var lType = { "input": function input(l) {
                    var newNeurons = l.out_sx !== undefined ? ml(l.out_sx, l.out_sy, 1, "input", 0, false) : ml(1, l.depth, 1, "input", 0, false);

                    _this.layerNodes.push(newNeurons);

                    _this.graph.layerCount++;
                    _this.offsetX += l.out_sx !== undefined ? 100 : 30;
                },
                "fc": function fc(l) {
                    var newNeurons = ml(1, l.num_neurons, 0, "fc", 0, false);
                    mr(_this.graph.layer_defs[_this.graph.layerCount].out_sx, _this.layerNodes[_this.layerNodes.length - 1], newNeurons, l.weights, "fc");

                    _this.layerNodes.push(newNeurons);

                    _this.graph.layerCount++;
                    _this.offsetX += 30;
                },
                "conv": function conv(l) {
                    var layerOrig = _this.layerNodes[_this.layerNodes.length - 1];

                    var newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", 180);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 0);
                    _this.layerNodes.push(newNeurons);

                    newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", 120);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 1);
                    _this.layerNodes[_this.layerNodes.length - 1] = _this.layerNodes[_this.layerNodes.length - 1].concat(newNeurons);

                    newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", 60);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 2);
                    _this.layerNodes[_this.layerNodes.length - 1] = _this.layerNodes[_this.layerNodes.length - 1].concat(newNeurons);

                    newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", -60);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 3);
                    _this.layerNodes[_this.layerNodes.length - 1] = _this.layerNodes[_this.layerNodes.length - 1].concat(newNeurons);

                    newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", -120);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 4);
                    _this.layerNodes[_this.layerNodes.length - 1] = _this.layerNodes[_this.layerNodes.length - 1].concat(newNeurons);

                    newNeurons = ml(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, _this.graph.layer_defs[_this.graph.layerCount - 1].out_sy, 0, "conv", -180);
                    mr(_this.graph.layer_defs[_this.graph.layerCount - 1].out_sx, layerOrig, newNeurons, l.weights, "conv", 5);
                    _this.layerNodes[_this.layerNodes.length - 1] = _this.layerNodes[_this.layerNodes.length - 1].concat(newNeurons);

                    _this.graph.layerCount++;
                    _this.offsetX += 100;
                },
                "regression": function regression(l) {
                    var offsetZ = -30.0 * (l.num_neurons / 2);
                    var we = l.weights;
                    var newWe = [];
                    if (l.weights !== undefined && l.weights !== null) {
                        for (var n = 0; n < l.num_neurons; n++) {
                            for (var nb = 0; nb < l.weights.length; nb = nb + l.num_neurons) {
                                newWe.push(l.weights[nb + n]);
                            }
                        }
                    }
                    for (var _n = 0; _n < l.num_neurons; _n++) {
                        _this.graph.addEfferentNeuron("O" + _this.outputCount, [_this.offsetX, 0.0, offsetZ, 1.0]); // efferent neuron (output)
                        _this.graph.connectNeuronLayerWithNeuron({ "neuronLayer": _this.layerNodes[_this.layerNodes.length - 1],
                            "neuron": "O" + _this.outputCount,
                            "weight": l.weights !== undefined && l.weights !== null ? newWe.slice(0, _this.layerNodes[_this.layerNodes.length - 1].length) : null,
                            "layer_neurons_count": _this.layerNodes[_this.layerNodes.length - 2].length * _this.layerNodes[_this.layerNodes.length - 1].length,
                            "layerNum": _this.graph.layerCount - 1 });
                        if (l.weights !== undefined && l.weights !== null) newWe = newWe.slice(_this.layerNodes[_this.layerNodes.length - 1].length);

                        _this.outputCount++;
                        offsetZ += 30.0;
                    }
                    _this.graph.layerCount++;
                } };
            for (var n = 0; n < layer_defs.length; n++) {
                var l = layer_defs[n];
                lType[l.type](l);
            }

            this.graph.createWebGLBuffers();
            this.graph.enableForceLayout();

            this.graph.setLearningRate(this.currentLearningRate);
        }
    }, {
        key: "fromJson",
        value: function fromJson(jsonIn) {
            var layer_defs = [];
            for (var n = 0; n < jsonIn.layers.length; n++) {
                if (jsonIn.layers[n].layer_type === "input") {} else if (jsonIn.layers[n].layer_type === "fc") {
                    jsonIn.layers[n].weights = [];
                    for (var key in jsonIn.layers[n].filters[0].w) {
                        for (var nb = 0; nb < jsonIn.layers[n].filters.length; nb++) {
                            jsonIn.layers[n].weights.push(jsonIn.layers[n].filters[nb].w[key]);
                        }
                    }
                    for (var _key in jsonIn.layers[n].biases.w) {
                        jsonIn.layers[n].weights.push(jsonIn.layers[n].biases.w[_key]);
                    }
                } else if (jsonIn.layers[n].layer_type === "regression") {
                    jsonIn.layers[n].weights = [];
                    for (var _key2 in jsonIn.layers[n].filters[0].w) {
                        for (var _nb = 0; _nb < jsonIn.layers[n].filters.length; _nb++) {
                            jsonIn.layers[n].weights.push(jsonIn.layers[n].filters[_nb].w[_key2]);
                        }
                    }
                    for (var _key3 in jsonIn.layers[n].biases.w) {
                        jsonIn.layers[n].weights.push(jsonIn.layers[n].biases.w[_key3]);
                    }
                }
            }
            for (var _n2 = 0; _n2 < jsonIn.layers.length; _n2++) {
                if (jsonIn.layers[_n2].layer_type === "input") layer_defs.push({ "type": "input", "depth": jsonIn.layers[_n2].out_depth });else if (jsonIn.layers[_n2].layer_type === "fc") layer_defs.push({ "type": "fc", "num_neurons": jsonIn.layers[_n2].out_depth, "activation": "relu", "weights": jsonIn.layers[_n2].weights });else if (jsonIn.layers[_n2].layer_type === "regression") layer_defs.push({ "type": "regression", "num_neurons": jsonIn.layers[_n2].out_depth, "weights": jsonIn.layers[_n2].weights });
            }

            this.sce.target.innerHTML = "";
            this.ini({ "target": this.sce.target,
                "dimensions": this.sce.dimensions,
                "batch_repeats": this.graph.batch_repeats,
                "learning_rate": this.currentLearningRate });
            this.makeLayers(layer_defs);
        }
    }, {
        key: "toJson",


        /**
         * @returns {String}
         */
        value: function toJson() {
            return this.graph.toJson();
        }
    }, {
        key: "forward",


        /**
         * @param {Array<number>} state
         * @param {Function} onAction
         * @param {boolean} [readOutput]
         */
        value: function forward(state, _onAction, readOutput) {
            this.graph.forward({ "state": state,
                "readOutput": readOutput,
                "onAction": function onAction(maxacts) {
                    _onAction(maxacts);
                } });
        }
    }, {
        key: "train",


        /**
         * @param {Array<Object>} reward [{dim: actionId for the reward , val: number}]
         * @param {Function} onTrain
         */
        value: function train(reward, onTrain) {
            this.graph.train({ "reward": reward,
                "onTrained": function onTrained(loss) {
                    onTrain(loss);
                } });
        }
    }, {
        key: "setLearningRate",
        value: function setLearningRate(v) {
            this.currentLearningRate = v;
            this.graph.setLearningRate(v);
        }
    }, {
        key: "enableShowOutputWeighted",
        value: function enableShowOutputWeighted() {
            this.graph.enableShowOutputWeighted();
        }
    }, {
        key: "disableShowOutputWeighted",
        value: function disableShowOutputWeighted() {
            this.graph.disableShowOutputWeighted();
        }
    }, {
        key: "enableShowWeightDynamics",
        value: function enableShowWeightDynamics() {
            this.graph.enableShowWeightDynamics();
        }
    }, {
        key: "disableShowWeightDynamics",
        value: function disableShowWeightDynamics() {
            this.graph.disableShowWeightDynamics();
        }
    }, {
        key: "enableShowValues",
        value: function enableShowValues() {
            this.graph.enableShowValues();
        }
    }, {
        key: "disableShowValues",
        value: function disableShowValues() {
            this.graph.disableShowValues();
        }
    }, {
        key: "tick",
        value: function tick() {
            this.project.getActiveStage().tick();
        }
    }]);

    return GBrain;
}();

global.GBrain = GBrain;
module.exports.GBrain = GBrain;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./Graph.class":8,"scejs":4}],17:[function(require,module,exports){
(function (global){
"use strict";

require("./gbrain");

require("./gbrain-rl");

require("./Graph.class");

require("./KERNEL_ADJMATRIX_UPDATE.class");

require("./KERNEL_DIR.class");

require("./ProccessImg.class");

require("./resources");

require("./VFP_NODE.class");

require("./VFP_NODEPICKDRAG.class");

require("./Plot.class");

require("./AvgWin.class");

module.exports.GBrain = global.GBrain = GBrain;

module.exports.GBrainRL = global.GBrainRL = GBrainRL;

module.exports.Graph = global.Graph = Graph;

module.exports.KERNEL_ADJMATRIX_UPDATE = global.KERNEL_ADJMATRIX_UPDATE = KERNEL_ADJMATRIX_UPDATE;

module.exports.KERNEL_DIR = global.KERNEL_DIR = KERNEL_DIR;

module.exports.ProccessImg = global.ProccessImg = ProccessImg;

module.exports.Resources = global.Resources = Resources;

module.exports.VFP_NODE = global.VFP_NODE = VFP_NODE;

module.exports.VFP_NODEPICKDRAG = global.VFP_NODEPICKDRAG = VFP_NODEPICKDRAG;

module.exports.Plot = global.Plot = Plot;

module.exports.AvgWin = global.AvgWin = AvgWin;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{"./AvgWin.class":7,"./Graph.class":8,"./KERNEL_ADJMATRIX_UPDATE.class":9,"./KERNEL_DIR.class":10,"./Plot.class":11,"./ProccessImg.class":12,"./VFP_NODE.class":13,"./VFP_NODEPICKDRAG.class":14,"./gbrain":16,"./gbrain-rl":15,"./resources":18}],18:[function(require,module,exports){
(function (global){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Resources = exports.Resources = function () {
    function Resources() {
        _classCallCheck(this, Resources);
    }

    _createClass(Resources, null, [{
        key: "imgWhite",
        value: function imgWhite() {
            return "data:image/jpeg;base64,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AAEAoZmaAAYAAAAAAAEAMgAAAAEAWgAAAAYAAAAAAAEANQAAAAEALQAAAAYAAAAAAAE4QklNA/gAAAAAAHAAAP////////////////////////////8D6AAAAAD/////////////////////////////A+gAAAAA/////////////////////////////wPoAAAAAP////////////////////////////8D6AAAOEJJTQQIAAAAAAAQAAAAAQAAAkAAAAJAAAAAADhCSU0EHgAAAAAABAAAAAA4QklNBBoAAAAAAz8AAAAGAAAAAAAAAAAAAAAQAAAAEAAAAAUAdwBoAGkAdABlAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAQAAAAEAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAABAAAAABAAAAAAAAbnVsbAAAAAIAAAAGYm91bmRzT2JqYwAAAAEAAAAAAABSY3QxAAAABAAAAABUb3AgbG9uZwAAAAAAAAAATGVmdGxvbmcAAAAAAAAAAEJ0b21sb25nAAAAEAAAAABSZ2h0bG9uZwAAABAAAAAGc2xpY2VzVmxMcwAAAAFPYmpjAAAAAQAAAAAABXNsaWNlAAAAEgAAAAdzbGljZUlEbG9uZwAAAAAAAAAHZ3JvdXBJRGxvbmcAAAAAAAAABm9yaWdpbmVudW0AAAAMRVNsaWNlT3JpZ2luAAAADWF1dG9HZW5lcmF0ZWQAAAAAVHlwZWVudW0AAAAKRVNsaWNlVHlwZQAAAABJbWcgAAAABmJvdW5kc09iamMAAAABAAAAAAAAUmN0MQAAAAQAAAAAVG9wIGxvbmcAAAAAAAAAAExlZnRsb25nAAAAAAAAAABCdG9tbG9uZwAAABAAAAAAUmdodGxvbmcAAAAQAAAAA3VybFRFWFQAAAABAAAAAAAAbnVsbFRFWFQAAAABAAAAAAAATXNnZVRFWFQAAAABAAAAAAAGYWx0VGFnVEVYVAAAAAEAAAAAAA5jZWxsVGV4dElzSFRNTGJvb2wBAAAACGNlbGxUZXh0VEVYVAAAAAEAAAAAAAlob3J6QWxpZ25lbnVtAAAAD0VTbGljZUhvcnpBbGlnbgAAAAdkZWZhdWx0AAAACXZlcnRBbGlnbmVudW0AAAAPRVNsaWNlVmVydEFsaWduAAAAB2RlZmF1bHQAAAALYmdDb2xvclR5cGVlbnVtAAAAEUVTbGljZUJHQ29sb3JUeXBlAAAAAE5vbmUAAAAJdG9wT3V0c2V0bG9uZwAAAAAAAAAKbGVmdE91dHNldGxvbmcAAAAAAAAADGJvdHRvbU91dHNldGxvbmcAAAAAAAAAC3JpZ2h0T3V0c2V0bG9uZwAAAAAAOEJJTQQoAAAAAAAMAAAAAj/wAAAAAAAAOEJJTQQRAAAAAAABAQA4QklNBBQAAAAAAAQAAAABOEJJTQQMAAAAAAIyAAAAAQAAABAAAAAQAAAAMAAAAwAAAAIWABgAAf/Y/+0ADEFkb2JlX0NNAAL/7gAOQWRvYmUAZIAAAAAB/9sAhAAMCAgICQgMCQkMEQsKCxEVDwwMDxUYExMVExMYEQwMDAwMDBEMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMAQ0LCw0ODRAODhAUDg4OFBQODg4OFBEMDAwMDBERDAwMDAwMEQwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAAQABADASIAAhEBAxEB/90ABAAB/8QBPwAAAQUBAQEBAQEAAAAAAAAAAwABAgQFBgcICQoLAQABBQEBAQEBAQAAAAAAAAABAAIDBAUGBwgJCgsQAAEEAQMCBAIFBwYIBQMMMwEAAhEDBCESMQVBUWETInGBMgYUkaGxQiMkFVLBYjM0coLRQwclklPw4fFjczUWorKDJkSTVGRFwqN0NhfSVeJl8rOEw9N14/NGJ5SkhbSVxNTk9KW1xdXl9VZmdoaWprbG1ub2N0dXZ3eHl6e3x9fn9xEAAgIBAgQEAwQFBgcHBgU1AQACEQMhMRIEQVFhcSITBTKBkRShsUIjwVLR8DMkYuFygpJDUxVjczTxJQYWorKDByY1wtJEk1SjF2RFVTZ0ZeLys4TD03Xj80aUpIW0lcTU5PSltcXV5fVWZnaGlqa2xtbm9ic3R1dnd4eXp7fH/9oADAMBAAIRAxEAPwD1VJJJJT//2ThCSU0EIQAAAAAAVQAAAAEBAAAADwBBAGQAbwBiAGUAIABQAGgAbwB0AG8AcwBoAG8AcAAAABMAQQBkAG8AYgBlACAAUABoAG8AdABvAHMAaABvAHAAIABDAFMANQAAAAEAOEJJTQQGAAAAAAAHAAUAAAABAQD/4RBcaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjAtYzA2MCA2MS4xMzQzNDIsIDIwMTAvMDEvMTAtMTg6MDY6NDMgICAgICAgICI+IDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcFJpZ2h0cz0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3JpZ2h0cy8iIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczpzdEV2dD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3NUeXBlL1Jlc291cmNlRXZlbnQjIiB4bWxuczpJcHRjNHhtcENvcmU9Imh0dHA6Ly9pcHRjLm9yZy9zdGQvSXB0YzR4bXBDb3JlLzEuMC94bWxucy8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6Y3JzPSJodHRwOi8vbnMuYWRvYmUuY29tL2NhbWVyYS1yYXctc2V0dGluZ3MvMS4wLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bXBSaWdodHM6TWFya2VkPSJGYWxzZSIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ1M1IiB4bXA6Q3JlYXRlRGF0ZT0iMjAxMS0wMS0xM1QxODozNjoyMyswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMTEtMDEtMTNUMTk6NDA6MjArMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMTEtMDEtMTNUMTk6NDA6MjArMDE6MDAiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6RkQ0MDEyOTM0NDFGRTAxMUI5QTZERTVCRDJEOEE4NUMiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6RkIwRTk2NkIwODZDMTFFMDlFQjNFQTUyRjBCQkI1RUQiIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDpGQjBFOTY2QjA4NkMxMUUwOUVCM0VBNTJGMEJCQjVFRCIgZGM6Zm9ybWF0PSJpbWFnZS9qcGVnIiBjcnM6QWxyZWFkeUFwcGxpZWQ9IlRydWUiIHBob3Rvc2hvcDpDb2xvck1vZGU9IjMiPiA8eG1wUmlnaHRzOlVzYWdlVGVybXM+IDxyZGY6QWx0PiA8cmRmOmxpIHhtbDpsYW5nPSJ4LWRlZmF1bHQiLz4gPC9yZGY6QWx0PiA8L3htcFJpZ2h0czpVc2FnZVRlcm1zPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDpGQjBFOTY2ODA4NkMxMUUwOUVCM0VBNTJGMEJCQjVFRCIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDpGQjBFOTY2OTA4NkMxMUUwOUVCM0VBNTJGMEJCQjVFRCIvPiA8eG1wTU06SGlzdG9yeT4gPHJkZjpTZXE+IDxyZGY6bGkgc3RFdnQ6YWN0aW9uPSJzYXZlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDpGRDQwMTI5MzQ0MUZFMDExQjlBNkRFNUJEMkQ4QTg1QyIgc3RFdnQ6d2hlbj0iMjAxMS0wMS0xM1QxOTo0MDoyMCswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIENTNSAoMTIuMHgyMDEwMDExNSBbMjAxMDAxMTUubS45OTggMjAxMC8wMS8xNTowMjowMDowMCBjdXRvZmY7IG0gYnJhbmNoXSkgIFdpbmRvd3MiIHN0RXZ0OmNoYW5nZWQ9Ii8iLz4gPC9yZGY6U2VxPiA8L3htcE1NOkhpc3Rvcnk+IDxJcHRjNHhtcENvcmU6Q3JlYXRvckNvbnRhY3RJbmZvIElwdGM0eG1wQ29yZTpDaUFkckV4dGFkcj0iIiBJcHRjNHhtcENvcmU6Q2lBZHJDaXR5PSIiIElwdGM0eG1wQ29yZTpDaUFkclJlZ2lvbj0iIiBJcHRjNHhtcENvcmU6Q2lBZHJQY29kZT0iIiBJcHRjNHhtcENvcmU6Q2lBZHJDdHJ5PSIiIElwdGM0eG1wQ29yZTpDaVRlbFdvcms9IiIgSXB0YzR4bXBDb3JlOkNpRW1haWxXb3JrPSIiIElwdGM0eG1wQ29yZTpDaVVybFdvcms9IiIvPiA8ZGM6dGl0bGU+IDxyZGY6QWx0Lz4gPC9kYzp0aXRsZT4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPD94cGFja2V0IGVuZD0idyI/Pv/uAA5BZG9iZQBkQAAAAAH/2wCEAAQDAwMDAwQDAwQGBAMEBgcFBAQFBwgGBgcGBggKCAkJCQkICgoMDAwMDAoMDAwMDAwMDAwMDAwMDAwMDAwMDAwBBAUFCAcIDwoKDxQODg4UFA4ODg4UEQwMDAwMEREMDAwMDAwRDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDP/AABEIABAAEAMBEQACEQEDEQH/3QAEAAL/xAGiAAAABwEBAQEBAAAAAAAAAAAEBQMCBgEABwgJCgsBAAICAwEBAQEBAAAAAAAAAAEAAgMEBQYHCAkKCxAAAgEDAwIEAgYHAwQCBgJzAQIDEQQABSESMUFRBhNhInGBFDKRoQcVsUIjwVLR4TMWYvAkcoLxJUM0U5KismNzwjVEJ5OjszYXVGR0w9LiCCaDCQoYGYSURUaktFbTVSga8uPzxNTk9GV1hZWltcXV5fVmdoaWprbG1ub2N0dXZ3eHl6e3x9fn9zhIWGh4iJiouMjY6PgpOUlZaXmJmam5ydnp+So6SlpqeoqaqrrK2ur6EQACAgECAwUFBAUGBAgDA20BAAIRAwQhEjFBBVETYSIGcYGRMqGx8BTB0eEjQhVSYnLxMyQ0Q4IWklMlomOywgdz0jXiRIMXVJMICQoYGSY2RRonZHRVN/Kjs8MoKdPj84SUpLTE1OT0ZXWFlaW1xdXl9UZWZnaGlqa2xtbm9kdXZ3eHl6e3x9fn9zhIWGh4iJiouMjY6Pg5SVlpeYmZqbnJ2en5KjpKWmp6ipqqusra6vr/2gAMAwEAAhEDEQA/APf2KuxV/9D39irsVf/Z";
        }
    }, {
        key: "nodesImgCrosshair",
        value: function nodesImgCrosshair() {
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAyBpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMC1jMDYwIDYxLjEzNDc3NywgMjAxMC8wMi8xMi0xNzozMjowMCAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENTNSBXaW5kb3dzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOkI5RDRFNjc3QzEzMzExRTVCNzJFQ0Y5RUM3NDVCNzMzIiB4bXBNTTpEb2N1bWVudElEPSJ4bXAuZGlkOkI5RDRFNjc4QzEzMzExRTVCNzJFQ0Y5RUM3NDVCNzMzIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6QjlENEU2NzVDMTMzMTFFNUI3MkVDRjlFQzc0NUI3MzMiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6QjlENEU2NzZDMTMzMTFFNUI3MkVDRjlFQzc0NUI3MzMiLz4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz6yqiYLAABZWklEQVR42uy9CXfjyI4s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";
        }
    }, {
        key: "nodesImgMask",
        value: function nodesImgMask() {
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAyBpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMC1jMDYwIDYxLjEzNDc3NywgMjAxMC8wMi8xMi0xNzozMjowMCAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENTNSBXaW5kb3dzIiB4bXBNTTpJbnN0YW5jZUlEPSJ4bXAuaWlkOjk2QjYzOENFQkY0RjExRTU4QjlERDczQjI3RDhFRDUxIiB4bXBNTTpEb2N1bWVudElEPSJ4bXAuZGlkOjk2QjYzOENGQkY0RjExRTU4QjlERDczQjI3RDhFRDUxIj4gPHhtcE1NOkRlcml2ZWRGcm9tIHN0UmVmOmluc3RhbmNlSUQ9InhtcC5paWQ6OTZCNjM4Q0NCRjRGMTFFNThCOURENzNCMjdEOEVENTEiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6OTZCNjM4Q0RCRjRGMTFFNThCOURENzNCMjdEOEVENTEiLz4gPC9yZGY6RGVzY3JpcHRpb24+IDwvcmRmOlJERj4gPC94OnhtcG1ldGE+IDw/eHBhY2tldCBlbmQ9InIiPz7NwWsNAAALMElEQVR42uzdW6ydYxoH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";
        }
    }, {
        key: "fonts",
        value: function fonts() {
            return "data:image/png;base64,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";
        }
    }]);

    return Resources;
}();

global.Resources = Resources;
module.exports.Resources = Resources;
}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}]},{},[17]);
