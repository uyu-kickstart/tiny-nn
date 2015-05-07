'use strict';

/*
 * # モジュールの読み込み。
 */
var
// Node.jsのモジュール
fs = require('fs'),
// 自作のモジュール
nn = require('../..');

var
// `nn`モジュールのクラス等への参照。
NeuralNetwork = nn.NeuralNetwork;

/*
 * # `NeuralNetwork`の構築。
 */
var
// 入力値の数。
X_LEN = 2,
// 隠れ層のユニットの数。
H_LEN = 2,
// 出力のユニットの数。
O_LEN = 2,
// `network.fit`に渡す`eta`（学習係数）。
ETA = 0.1,
// 学習の繰り返し回数。
LOOP = 100;

// コマンドライン引数が足りないときは使い方を表示して終了。
if (process.argv.length < 3) {
  console.log('node xor.js [filename]');
  process.exit(1);
}

// 変数の宣言。
var
network,
// `node xor.js`のあとの最初の引数が学習データを読み書きするファイル名。
fileName = process.argv[2];

// ファイルを読み込むか、新しく`NeuralNetwork`クラスのインスタンスを作成する。
if (fs.existsSync(fileName)) {
  network = NeuralNetwork.loadMsgpack(fs.readFileSync(fileName));
} else {
  network = new NeuralNetwork('sigmoid', 'tanh', nn.randomMatrix(H_LEN, X_LEN + 1), nn.randomMatrix(O_LEN, H_LEN + 1));
}

// `network`に対して学習をする。
(function () {
  var
  i, j, p;

  console.log('training start');
  for (i = 1; i <= LOOP; i++) {
    console.time('loop ' + i);
    p = 0;
    for (j = 0; j < 50; j++) {
      if (train([0, 0], 0)) p++;
      if (train([1, 0], 1)) p++;
      if (train([0, 1], 1)) p++;
      if (train([1, 1], 0)) p++;
    }
    console.timeEnd('loop ' + i);
    console.log('loop ' + i + ': ' + (p / 200 * 100) + '%');
  }
  console.log('training end');

  // 入力`X`に対して、期待する出力は`tn`が最も大きくなると学習させる。
  //
  // 返り値は学習前に期待する出力が返ったかどうか（真偽値）。
  function train(X, tn) {
    var
    result,
    T = [],
    k;

    // `tn`以外`0`で、`tn`が`1`の配列を作る。
    for (k = 0; k < O_LEN; k++) T[k] = 0;
    T[tn] = 1;

    // 学習させる。
    result = network.fit(X, T, ETA);

    return nn.maxIndex(result.O) === tn;
  }
})();

// 実行して結果を確認する。
run([0, 0], 0);
run([0, 1], 1);
run([1, 0], 1);
run([1, 1], 0);

function run(X, tn) {
  var
  result = network.fire(X),
  on = nn.maxIndex(result.O);

  console.log('- - - - - - -');
  console.log('X =', X);
  console.log('O =', result.O);
  console.log('maxIndex(T) =', tn);
  console.log('maxIndex(O) =', on);
  console.log(tn === on ? 'succeed' : 'failure');
}

// 学習したデータをファイルに書き出す。
console.log('- - - - - - -');
console.log('save to ' + fileName);
network.msgpackStream()
  .pipe(fs.createWriteStream(fileName))
  .on('finish', function end() {
    console.log('save done');
    process.exit(0);
  });
