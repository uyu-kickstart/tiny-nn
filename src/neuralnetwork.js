'use strict';

/*
 *  # モジュールの読み込み
 */
var
// Node.jsのモジュール
assert          = require('assert'),
// 3rdパーティーのモジュール
msgpack         = require('msgpack5')(),
stream_buffers  = require('stream-buffers'),
// 自作のモジュール
activation_func = require('./activation_func');


/*
 * # NeuralNetworkクラス
 *
 * 基本的なアルゴリズムは<http://www-ailab.elcom.nitech.ac.jp/lecture/neuro/menu.html>のバックプロパゲーションを参照。
 */

/*
 * ## コンストラクター
 *
 * 引数は、
 *
 * - `fhName`は入力→隠れ層の活性化関数の名前。`'sigmoid'`か`'tanh'`か`'inverse_abs'`、`'linear'`。
 * - `foName`は入力→隠れ層の活性化関数の名前。`'sigmoid'`か`'tanh'`か`'inverse_abs'`、`'linear'`。
 * - `Wh`は入力→隠れ層の結合荷重（二重配列）
 * - `Wo`は隠れ層→出力の結合荷重（二重配列）
 */
function NeuralNetwork(fhName, foName, Wh, Wo) {
  // `new`無しでもインスタンスを作成できるようにする。
  if (!(this instanceof NeuralNetwork)) return new NeuralNetwork(Wh, Wo);

  // `Wh.length`は隠れ層のユニットの数に等しく、
  // `Wo[0].length`は隠れ層のユニットの数より一つ多い。
  // （隠れ層のユニットの値の最後に`1`を追加して閾値を表現するため）
  assert(Wh.length + 1 === Wo[0].length, 'array sizes don\'t match.');

  // 引数を保存。
  this.fhName = fhName;
  this.foName = foName;
  this.Wh = Wh;
  this.Wo = Wo;

  // 活性化関数`f(x)`と活性化関数を微分したもの`df(y)`（`y`も同時に渡しているのは速度面の都合）
  this.fh = activation_func[fhName].func;
  this.dfh = activation_func[fhName].defunc;
  this.fo = activation_func[foName].func;
  this.dfo = activation_func[foName].defunc;
}

/*
 * ## 各ユニットの出力を計算
 *
 * 引数は、
 * 
 * - `X`は入力値（配列）
 *
 * 返り値は、
 *
 * - `result.H`は隠れ層の各ユニットの値（配列）
 * - `result.O`は出力の各ユニットの値（配列）
 */
NeuralNetwork.prototype.fire = function fire(X) {
  // `Wh[0].length`は入力値の数より一つ多い。
  assert(X.length + 1 === this.Wh[0].length, 'array sizes don\'t match.');

  // 変数宣言。
  var
  // `this`無しで参照できるようにする。
  fh = this.fh, fo = this.fo,
  Wh = this.Wh, Wo = this.Wo,
  // - `H`は隠れ層の各ユニットの値の配列
  // - `O`は出力の各ユニットの値の配列
  H = [], O = [],
  // - `i`は`X`に対する添字
  // - `j`は`H`に対する添字
  // - `k`は`O`に対する添字
  i, j, k,
  // 合計値を一時的に格納する変数。
  sum;

  // 隠れ層の各ユニットの値の計算する。
  //
  // ```
  // ${\bf H}_j = f_h( \sum_{i = 0}^{{\rm len}({\bf X}) + 1}{{\bf X}_i {\bf Wh}_{ji}} )$
  // ```
  for (j = 0; j < Wh.length; j++) {
    sum = 0;
    for (i = 0; i < X.length; i++) {
      sum += X[i] * Wh[j][i];
    }
    sum += 1 * Wh[j][X.length];

    H[j] = fh(sum);
  }

  // 出力の各ユニットの値の計算する。
  //
  // ```
  // ${\bf O}_k = f_o( \sum_{j = 0}^{{\rm len}({\bf H}) + 1}{{\bf H}_j {\bf Wo}_{kj}} )$
  // ```
  for (k = 0; k < Wo.length; k++) {
    sum = 0;
    for (j = 0; j < H.length; j++) {
      sum += H[j] * Wo[k][j];
    }
    sum += 1 * Wo[k][H.length];

    O[k] = fo(sum);
  }

  // 返り値は`H`と`O`というプロパティを持ったオブジェクト。
  return {
    H: H,
    O: O,
  };
};

/*
 * ## 教師データを用いて学習
 *
 * 引数は、
 *
 * - `X`は入力値（配列）
 * - `T`は教師データとして期待される出力（配列）
 * - `eta`は学習係数（小数値、デフォルト値は`0.1`）
 *
 * 返り値は、`this.fire(X)`の結果。
 * （精度を確認する際に再計算を防ぐためで、あまり深い意味はない）
 */
NeuralNetwork.prototype.fit = function fit(X, T, eta) {
  // デフォルト値の処理
  if (typeof eta === 'undefined') eta = 0.1;

  // `Wo.length`は出力のユニットの数と等しい。
  assert(this.Wo.length === T.length, 'array sizes don\'t match.');

  // 変数宣言。
  var
  // `this`無しで参照できるようにする。
  dfh = this.dfh, dfo = this.dfo,
  Wh = this.Wh, Wo = this.Wo,
  // 隠れ層と出力の各ユニットの値を計算した結果。
  result = this.fire(X),
  // - `H`は隠れ層の各ユニットの値
  // - `O`は出力の各ユニットの値
  H = result.H,
  O = result.O,
  // - `dO`はδO。
  // - `dHj`はδHj。
  dO = [],
  dHj,
  // - `i`は`X`に対する添字
  // - `j`は`H`に対する添字
  // - `k`は`O`に対する添字
  i, j, k,
  // 合計値を一時的に格納する変数。
  sum;

  // δOの計算。
  //
  // ```
  // ${\bf \delta O}_k = ({\bf T}_k - {\bf O}_k) df_o({\bf O}_k)$
  // ```
  for (k = 0; k < O.length; k++) {
    dO[k] = (T[k] - O[k]) * dfo(O[k]);
  }

  // `Wh`の更新。
  //
  // ```
  // $\Delta {\bf Wh}_{ji} = \eta {\bf X}_i df_h({\bf H}_j) \sum_{k = 0}^{{\rm len}({\bf O})} {\bf Wo}_{kj} {\bf \delta O}_k$
  // ```
  for (j = 0; j < Wh.length; j++) {
    sum = 0;
    for (k = 0; k < O.length; k++) {
      sum += Wo[k][j] * dO[k];
    }
    dHj = dfh(H[j]) * sum;

    for (i = 0; i < X.length; i++) {
      Wh[j][i] += eta * dHj * X[i];
    }
    Wh[j][X.length] += eta * dHj * 1;
  }

  // `Wo`の更新。
  //
  // ```
  // $\Delta {\bf Wo}_{kj} = \eta {\bf H}_j {\bf \delta O}_k$
  // ```
  for (k = 0; k < Wo.length; k++) {
    for (j = 0; j < H.length; j++) {
      Wo[k][j] += eta * dO[k] * H[j];
    }
    Wo[k][H.length] += eta * dO[k] * 1;
  }

  // 返り値は`result`。
  return result;
};

/**
 * ## 学習の結果をJavaScriptのオブジェクトにする。
 *
 * 返り値は`[fhName, foName, Wh, Wo]`という形式。
 */
NeuralNetwork.prototype.toObject = function toObject() {
  return [this.fhName, this.foName, this.Wh, this.Wo];
};

/**
 * ## 学習の結果をJSONにする。
 */
NeuralNetwork.prototype.toJSON = function toJSON() {
  return JSON.stringify(this.toObject());
};

/*
 * ## 学習の結果をストリームに書き出す。
 *
 * 返り値は学習の結果の書き込まれたストリーム。
 */
NeuralNetwork.prototype.msgpackStream = function msgpackStream() {
  var
  // `toObject`の形式で、MessagePackでエンコードする。
  data = this.toObject(),
  buffer = msgpack.encode(data),

  // ストリームの作成。
  stream = new stream_buffers.ReadableStreamBuffer();

  // ストリームに書き込み。
  stream.put(buffer);
  setTimeout(function end() {
    stream.emit('end');
  }, 0);

  // 返り値はストリーム。
  return stream;
};

/*
 * ## 学習の結果をMsgpack形式のバッファから読み込む。
 *
 * 引数は、
 *
 * - `buffer`は学習の結果がMessagePackでエンコードされたバッファ。
 *
 * 返り値は読み込まれた`NeuralNetwork`クラスのインスタンス。
 */
NeuralNetwork.loadMsgpack = function loadMsgpack(buffer) {
  var
  data = msgpack.decode(buffer);

  assert(data instanceof Array, 'buffer is invalid format.');
  assert(data.length === 4, 'buffer is invalid format.');

  return new NeuralNetwork(data[0], data[1], data[2], data[3]);
};

/**
 * ## 学習の結果をJavaScriptのオブジェクトから読み込む。
 *
 * 引数は、
 *
 * - `data`は学習の結果のオブジェクト。
 *
 * 返り値は読み込まれた`NeuralNetwork`クラスのインスタンス。
 */
NeuralNetwork.load = function load(data) {
  assert(data instanceof Array, 'object is invalid format.');
  assert(data.length === 4, 'object is invalid format.');

  return new NeuralNetwork(data[0], data[1], data[2], data[3]);
};

/*
 * # クラスのエクスポート。
 */
module.exports = NeuralNetwork;
