'use strict';

/*
 * # 便利関数達
 */

/*
 * ## 配列の中でもっとも大きい値の添字を返す。
 */
exports.maxIndex = function maxIndex(array) {
  var
  max = -Infinity, maxIndex = -1,
  i, len = array.length;

  for (i = 0; i < len; i++) {
    if (max < array[i]) {
      max = array[i];
      maxIndex = i;
    }
  }

  return maxIndex;
};

/*
 * ## `n`×`m`のランダムな行列を作る。
 */
exports.randomMatrix = function randomMatrix(n, m) {
  var
  mat = [],
  i, j;

  for (i = 0; i < n; i++) {
    mat[i] = [];
    for (j = 0; j < m; j++) {
      mat[i][j] = (Math.random() - 0.5) * 0.1;
    }
  }

  return mat;
};
