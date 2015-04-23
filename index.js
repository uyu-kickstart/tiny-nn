'use strict';

/*
 * # モジュールのエクスポート。
 */
exports.NeuralNetwork   = require('./src/neuralnetwork');
exports.activation_func = require('./src/activation_func');
 
var
util = require('./src/util');
Object.keys(util).forEach(function loop(key) {
  exports[key] = util[key];
});
