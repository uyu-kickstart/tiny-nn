'use strict';

/*
 * # 活性化関数
 */

/*
 * ## シグモイド関数
 */
exports.sigmoid = {
  // ```
  // $f(x) = \frac{1}{1 + {\rm exp}(-x)}$
  // ```
  func: function funcSigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  },

  // ```
  // $df(y) = y \times (1 - y)$
  // ```
  defunc: function defuncSigmoid(y) {
    return y * (1 - y);
  },
};

/*
 * ## 双曲線正接関数
 */
exports.tanh = {
  // ```
  // $f(x) = tanh(x)$
  // ```
  func: Math.tanh || function funcTanh(x) {
    var
    y;

    if (x === Infinity) {
      return 1;
    } else if (x === -Infinity) {
      return -1;
    } else {
      y = Math.exp(2 * x);
      return (y - 1) / (y + 1);
    }
  },

  // ```
  // $df(y) = 1 - y^2$
  // ```
  defunc: function defuncTanh(y) {
    return 1 - y * y;
  },
};

/*
 * ## 逆abs
 */
exports.inverse_abs = {
  // ```
  // $f(x) = \frac{x}{1 + \|x\|}$
  // ```
  func: function funcInverseAbs(x) {
    return x / (1 + (x < 0 ? -x : x));
  },

  // ```
  // $df(y) = (1 - \|y\|)^2$
  // ```
  defunc: function defuncInverseAbs(y) {
    if (y < 0) y = -y;
    return (1 - y) * (1 - y);
  },
};

/*
 * ## 線形関数
 */
exports.linear = {
  // ```
  // $f(x) = x$
  // ```
  func: function funcLinear(x) {
    return x;
  },

  // ```
  // $df(y) = 1$
  // ```
  defunc: function funcLinear(y) {
    return 1;
  },
};
