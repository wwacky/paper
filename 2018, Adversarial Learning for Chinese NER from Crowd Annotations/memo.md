[arxiv](https://arxiv.org/abs/1801.05147)|[PDF](https://arxiv.org/pdf/1801.05147.pdf)


| title        | Adversarial Learning for Chinese NER from Crowd Annotations                                              |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| author       | YaoSheng Yang(1), Meishan Zhang(4), Wenliang Chen(1), Wei Zhang(2), Haofen Wang(3), Min Zhang(1)         |
| organization | (1)Soochow University (2) Alibaba Group (3) Shenzhen Gowild Robotics Co. Ltd (4) Heilongjiang University |
| year         | 2018                                                                                                     |
| publish      | AAAI-2018                                                                                                |

# 要約

新規性を主張しているポイント
- クラウドソーシングなどで集めたノイズを含む学習データで訓練する場合にAdversarial Learningの仕組みを利用してノイズに強いモデルを構築する
  - ノイズを含む学習データで訓練すること自体は先行研究があるので、Adversarial Learningを用いたというのがポイント

提案手法
- AL-Crowd (Adversarial Learning Crowd)
  - 手法の特徴
    - Annotatorの識別をAdversarial Learningのdiscriminatorで学習させることで、generatorに相当する部分が入力データにannotatorの特徴を学習するようになる？(理解に自信なし)
  - 構成
  - input sequence(character level) → word embedding → (common BiLSTM + private BiLSTM) → CRF
    - private BiLSTMはinput sequenceから写像されてくるデータをそのまま学習したもの
    - common BiLSTMはAdversarial Learningを用いてAnnotator毎の特徴を反映させたもの
    - Worker-Adversarialを別途設けて、common BiLSTMのウェイトを学習を行う

手法の評価
- 評価対象のデータ(Table 1参照)
  - チャットボットの対話ログ:16,948件、平均9.21文字
  - ECの商品タイトル:2,337件、平均34.97文字
  - ECのユーザクエリ(検索クエリ?):2,300件、平均7.69文字
  - Alibabaのデータと明記されていないのが微妙なところ
- ラベルの種類
  - 対話ログ : 人名、歌の名前
  - ECの商品タイトル・クエリ : ブランド、Product(商品名？)、Model(モデル)、Material(素材)、Specification(スペック？)
    - 例が無いので上記以上の情報なし
- アノテーション
  - アノテーションガイドラインのドキュメントを与え、tipsを15分間教え、20個の例を提供した
  - 対話ログ
    - ランダムに20,000センテンスを選択し、43人の大学生がアノテーションを実施。1つのセンテンスを3人が行う。割り振り方法は不明
    - ラベル付の後にannotatorから報告があった不正なセンテンスを除外
    - 最終的に16,948センテンスにアノテーションが行われた
    - ラベル付け後に1,000センテンスをランダムサンプリングし、熟練者によるアノテーションを実施し、真のデータとした
  - ECの商品タイトル・クエリ
    - 5人の大学生がアノテーションを実施。1つのセンテンスを2人が行う。割り振り方法は不明
    - ラベル付の後にannotatorから報告があった不正なセンテンスを除外
    - ラベル付け後に400センテンスをランダムサンプリングし、熟練者によるアノテーションを実施し、真のデータとした
- 比較対象
  - Baseline：LSTM-CRF
  - 提案手法:AL-Crowd
  - その他の比較用の手法:CRF、CRF-VT(訓練データを多数決で生成して訓練)、CRF-MA(annotatorの情報を入れるっぽい)、LSTM-CRF-VT(Baselineでデータを多数決で生成して訓練)、LSTM-Crowd
  - 多数決をすると性能が悪くなる（多数決する人数が少ないか、重要な情報が過程で欠損したからと考察）
    - 純粋なCRFより悪くなってしまうのが面白い

所感
- クラウドソーシングの結果をそのまま使うのは面白いけど、性能(Table2 3)を見るとAdversarial learningではなく、DL使わない既存の手法でもよさそう。CRF-MAとか
- 多数決をすれば必ずしも良くなるわけではないというのも参考になる。ただ、この研究の比較では多数決をすると純粋にデータが減るので、多数決したデータ件数が多数決していないものと同じであれば品質勝負で性能が上回るかもしれないので、コストをかけて解決という力技もあるかも。Kappa value?を見て一致率が高ければ力技もあるかな。低いとダメそうだけど
- Adversarial Learningを使うとcommon BiLSTMの部分がどういう学習をするのか直感的によく分からない。例があるのでもうちょっとよく読んだ方がいいかも(今は理解できなかった)
- データを追加しようとした時にAnnotatorが変わってしまっても問題なく学習できるのか不明。Annotatorを追加したときはDiscriminatorの識別対象を増やしていけば良さそうだが、そうするとDiscriminatorの学習が難しくなってしまう。もしかしたらECのannotatorが5人と少ないのはDiscriminatorの学習難易度を下げるためかもしれない(書いてないけど)

# 各章のメモ

## Abstract

- クラウドソーシングなどでラベリングを行うと、品質が低いデータになってしまう
- 中国語のNERタスクでマルチラベリングされたノイズを含むシーケンスラベルを最大限に活用する方法を提案する
- Adversaialを参考にcommon Bi-LSTMとprivate Bi-LSTMを用いる。それぞれがannotator-generic informationとannotator-specific informationとなる
- annotator-generic informationはクラウドから簡単に作れる
- 2つのドメインから2つのデータセットを作成しNERタスクを行い検証した

## Introduction

- 最近の研究は熟練者によって大量のラベリングされた学習データを用いて行うものが典型的であった (Zhao and Kit 2008; Collobert et al.2011; Lample et al. 2016)
- すばやく新しい学習データを使うためにはクラウドソーシングを用いることができるが低品質
  - 多数決によって品質を上げることもできるが、無駄が多い
  - 主な研究は作業者の違いを学習してモデルを構築するものであった(l (Rodrigues, Pereira, and Ribeiro 2014; Nguyen et al. 2017))
- 中国語は英語に比べて大文字化(capitalization)や単語の区切りが不確かなので難易度が高い
  - character-level tagging( (Peng and Dredze 2015))で対応することもできるが、ドメインが変わる度にデータが必要になる
- 本研究ではAdversarial trainingでannotatorの独立した特徴を追加で取得する
  - Adversarial trainingはNLPでも利用されているが、クラウドアノテーションラーニングに利用するのは本研究が初めて
- another label Bi-LSTMはクラウドアノテーターが作成したデータから構築される
- common and label Bi-LSTMはdiscriminatorの入力になり、パラメータはadversarial learningで学習する
- 評価データセットは対話とECの2つのドメイン。
- ラベルは、人物、歌、ブランド、製品など
- 対話が入っているのはチャットボット想定

貢献
- adversarial learningを用いたクラウドアノテーションラーニングモデルを提案した
- 対話とECのデータでをクラウドアノテーションで作成し、提案手法が比較手法よりよい性能であることを確認した

## Related Work

## Baseline:LSTM-CRF
- BIOEスキーマを用いる
  - O:ラベルなし
  - B-XX:エンティティXXのB
  - E-XX:エンティティXXのE
  - I-XX:エンティティXXのI
  - XX:エンティティの種類
- baselineは主に3つのコンポーネントを持つ(Fig1の右側)
  - 単語のベクトル表現への変換
  - bi-directional LSTMを用いた特徴抽出(隠れ層にするだけっぽい)
  - CRF tagging

![Fig1](img/76961f9eb6bbeb40f7617755bee72b2a.png =800x)

### Vector Representation of Characters

- 大きなコーパスで学習済みのものをファインチューニングする

### Feature Extraction

- BiLSTMを使う
  - private BiLSTMと呼ぶ
  - Fig1でBaselineの方は $h^{private}_t$ になっていて、そこから $h^{ner}_t$が算出されている

### CRF Tagging

### Training

- クラウドソーシングで付与されたラベルを用いて損失関数を定義する

## Worker Adversarial

例がいっぱい書いてあるがイマイチわからん。ラベリングしているworkerを判別するdiscriminatorを作るっぽい

- 4つのコンポーネントで構成される
  - common BiLSTM over input Characters
  - additional BiLSTM to encode crowd-annotated NE label sequence
  - CNN to extract features for worker discriminator
  - output and prediction

### Common Bi-LSTM over Characters



### Additional Bi-LSTM over Annotated NER labels

- label BiLSTMと呼ぶ
- crowd annotatorを識別するためだけに用いる
- 入力はアノテーションで付与されたラベルのシーケンス(文字のシーケンスと対応するはず？)

### CNN

- worker discriminatorのインプットを作成する
  - common BiLSTMとlabel BiLSTMの出力を、CNNの入力とする
- window size = 5


### Output and Prediction

次の節でソフトマックスで確率に直す話が出てくる


### Adversarial Training

- baselineと違って外部のworker discriminatorの学習も必要
- 目的関数はbaselineのNERと同じnegative log-liklihoodと、worker discriminatorのnegative log-liklihoodの2つを持つことになる
- discriminatorのlog-liklihoodを計算するために、ソフトマックスで確率を算出する

## Experiments

- 対話とECの2つのドメインからセンテンスを集め、評価を行った
- 大学生にアノテーションを行わせた
  - 彼らには予め定義された種類のエンティティをセンテンスに対して付与するタスクを行わせた
  - ガイドラインのドキュメントを与え、tipsを15分間教え、20個の例を提供した

### Laveled Data: DL-PS

- dialog(対話)のドメイン
- チャットボットのアプリから生のセンテンスを集めた
  - ランダムに20,000センテンスを選択し、43人の大学生にアノテーションを行わせた
  - アノテーションは人物名と歌の名前のエンティティの2つをラベリングさせた
  - ラベル付は独立して行わせた。センテンスに対して3人が行うものもあったが無駄となるが、baselineのモデルでは多数決に利用できる
  - ラベル付の後にannotatorから報告があった不正なセンテンスを除外した
  - 最終的に16,948センテンスにアノテーションが行われた
  - 平均Kappa valueは0.6033
    - kappa valueはannotatorのラベルの一致を表す指標
  - 1,000センテンスをランダムサンプリングし、熟練者によるアノテーションを実施した
    - 300件を学習用、700件をテスト用とした。学生がアノテーションした残りのデータは学習データとして利用した

![Table1](img/65a6540549b07947f0e736eea9c741a0.png =600x)

### Labeled data: EC-MT and EC-UQ

- E-Commerceのドメインデータ
- MT:Title of Merchandise(商品タイトル)
- UQ:User Queries(ユーザクエリ)
- 5種類のエンティティを付与した
  - ブランド、Product(商品名？)、Model(モデル)、Material(素材)、Specification(スペック？)
  - 商品のKnowledge Graphを構築など、ECプラットフォームではこの5種類が重要
  - センテンス数が少なかったので5人の大学生が参加
  - DL-PSと同じ方法でアノテーションを実施。ただし、それぞれのセンテンスに2人のannotatorをアサインするようにだけ変更した
  - EC-MT:2,337センテンス
  - EC-UQ:2,300センテンス
  - 400件ランダムサンプリングして、熟練者によるアノテーションを行い、真の正解とした。100件を学習用、300件をテスト用にした。
    - 学生がアノテーションした残りのデータは学習用のデータとした

### Unlabeled data

- character level inputのvector representationを行うという話
- ラベル付されていないデータを用いる
- word2vecを用いて5Mセンテンスを学習した

### Settings

- 評価指標
  - Entity-level metrics
    - Precision (P)
    - Recall (R)
    - F1 value (F1)
- Baselineと提案手法はNER部分は一緒なので同じハイパーパラメータを使用
  - dimension size of character embedding : 100
  - dimension size of the NE label embedding : 50
  - dimension size of all the other hidden feature : 200
- mini batch size : 128
- max-epoch iteration : 200
- optimizer:RMSprop
  - learning rate : 10e-3
  - L2正則化のパラメータ : 10e-5
  - dropoutのドロップ率 : 0.2

### Comparison Systems

- AL-Crowd:提案手法 (Adversarial Learning Crowd)
- CRF:http://www.chokkan.org/software/crfsuite/　でクラウドソーシングでラベル付したデータを学習
- CRF-VT:CRFと同じセッティングだが、ラベル付されたデータを多数決で正解を付けた(vote version)
- CRF-MA: Rodrigues,Pereira, and Ribeiro (2014)のCRFを使い、クラウドソーシングのannotatorの事前確率が利用されている
- LSTM-CRF:Baselineのシステム。クラウドソーシングのデータを学習
- LSTM-CRF-VT:Baselineのシステム。クラウドソーシングのデータで多数決で正解を付ける(CRF-VTと同じ)
- LSTM-Crowd:先行研究のやつ（Nguyen et al. (2017)）


### Main Results

![Table2](img/099a4f13f368a91ec6d418756c5eff1a.png =600x)

![Table3](img/29f36faca192f89028cc68ba5226d8d9.png =600x)

- 多数決を行ったモデル(-VTの方が性能が悪い)
  - 3人の多数決じゃ少なかったかもしれない
  - 多数決で欠損した情報が重要だったのかも知れない

### Discussion
#### Impact of Character Embeddings

- characterを分散表現にするとF1が高くなっている(Fig2参照)

![Fig2](img/e44cbd97bca3493ebb2cf6c77ada76e3.png =600x)

#### Case Studies

- 3人にアノテーションさせたデータと、多数決(Majority-Voting)、LSTM-CRFの予測、ALCrowdの予測を比較する(Fig3)
  - ALCrowdは3人のアノテーション結果を上手く取り込んでいることがわかる

![Fig3](img/b7e48bc55c7f9e014b9f361d8d1c2e42.png =600x)


## Conclusion

# 略語
| term  | discription                                   |
| ----- | --------------------------------------------- |
| DL    | Dialog domain                                 |
| DL-PS | Dialog domain Person-name Song-name           |
| EC    | E-commerce domain                             |
| EC-MT | E-Commerce domain Title of Merchandise entity |
| EC-UQ | E-Commerce domain User Queries                |
| F1    | F1 value                                      |
| P     | Precision                                     |
| R     | Recall                                        |
|       |                                               |

# 変数
- $c_t$ : chinese character sequence
- $E^W$ : vector representation model parameter
- $h^{ner}_n$ : high-level feature
- $n$ : character length
- $o^{ner}_t$ : CRFで算出されるラベルのスコア(次元数はラベルの数になる)
- $\boldsymbol{T}$ : CRF model parameter
- $t$ : character token $t \in [1,n]$
- $x_t$ : vector representation
- $\boldsymbol{y}$ : output label sequence
- $\boldsymbol{\bar{y}}$ : crowd-annotated label sequences
- $\boldsymbol{Y_X}$ : candidate label sequence
- $\bar{z}$ : annotation worker
