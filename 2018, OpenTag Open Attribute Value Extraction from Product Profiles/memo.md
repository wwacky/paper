https://arxiv.org/pdf/1806.01264.pdf

# title
|              |                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| title        | OpenTag: Open Attribute Value Extraction from Product Profiles                                                                    |
| author       | Guineng Zheng (University of Utah), Subhabrata Mukherjee (Amazon.com), Xin Luna Dong (Amazon.com), Feifei Li (University of Utah) |
| year         | 2018                                                                                                                              |
| organization | KDD                                                                                                                               |
|              |                                                                                                                                   |

このへんメモってない


# EXPERIMENTS

## 5.1：OpenTag:Training
## 5.2：Data sets
- Evaluation measure
  - precision
  - recall
  - f-score


flavorが"ranch raised lamb"があった時に3語とも抜けないと正解にしない

## 5.3：Performance: Attribute Value Extraction
比較対象
- Baseline：BiLSTM[10]
- state-of-the-art：BiLSTM and CRF[11,13,15,17] ただし、辞書やhand-crafted featuresを利用しない

**Tagging strategy**
{B,I,O,E}を採用。他の方法も試したがこれが良かった
Attribute value extraction result

**Attribute value extraction results**

全てRandom train-test splitで評価。

descriptionといった文脈がわかりやすいものだと最大でstate-of-artのBiLSTM-CRFよりOpenTagの方が5.3%良かった。ただし、全体的なパフォーマンスは良くなかった？
この辺よくわからない


![](img/ed7f22ab0f93ed9178cabd58fd23af1e.png =600x)

**Discovering new attribute values with open world assumption(OWA)**


新しい特徴値を発見できているかの評価。いかなる属性値も学習データとテストデータで共有されないようにデータを分割した。Random splitは比較対象。baselineとの比較はないのか？


![](img/016b2e773fc00a6168415e07a2ba8434.png =600x)

**Joint extraction of multi-attribute values**

複数の属性値を抽出できるかのテスト。brand、favor、capacityをドッグフードのデータのtitles(disjoint split学習とテストで属性が被らないsplit)でテスト。{B,I,O,E}を使う。属性aごとに{B,I,E}をつける。

Table 4(前述のTable)のMulti AttributeのBiLSTM+CRFとOpenTagのf値を比較すると2%精度が上がっていた。

Multi Attributeのほうが意味的な分散を活用できるので、Single Attributeよりも精度を上げることができる。ブランドや容量の精度は上がったが、flavorの性能は下がってしまった


![](img/fe2987dba02de0a48e106a59f4df4ee9.png =600x)

## 5.4 OpenTag: Interpretability via Attention

**Interpretable explanation using attention**

OpenTagで学習を行った場合のAttention Matrix Aのヒートマップ。隣接する単語との重要度が高いものを表している。これによってタグの決定にどう影響しているか分かる。Fig3の例を見てみると、真ん中に白い四角が4つあり、列に(with,and)行に(beef,liver)がある。flavorのattributeが隣接して2つある場合も抽出できる？

このへんはAttentionを理解してないとキツそう

![](img/7e558bc07fd238d8fcc62b1f056767aa.png =600x)


**OpenTag achieves better concept clustering**

ドッグフードのタイトルでflavorを抽出下際の結果。t-SNEを使ってBiLSTMの隠れ層(サイズ200)を2次元に圧縮した。色は{B,I,O,E}。
(b)重要そうな単語をピックアップ。接続詞(with,and,&など)が右下に集まっており、上には量を表す語(pound,ounce,lb)がまとまっている。
attention machanismによって重要な単語が協会に現れている？といっているみたい。細かいロジックはよくわかってないが、attentionによって{B,I,O,E}が上手くまとまっているということらしい。Fig4の(d)


![Fig4](img/915ce24088e4f4798161c5338b4b7253.png =600x)

## 5.5 OpenTag with Active Learning: Results

### 5.5.1 Active Learning with Held-Out Test Set

Table3はUとHの比率を2 : 1。最初に50個の少ないラベリングされたサンプリングから始める。
Fig5はactive learningを20round行った結果。(a)洗剤のタイトルからscent-attributeを抽出(b)ドッグフードのタイトルからマルチ属性(ブランド、容量、flavor)を抽出を行った結果。

Active Learningを行うとstate-of-artとの比較を行っているTable4の結果より良くなる。Tag Flipの方がLeast Confidenceより良い。LC=least confidence、TF=tag flip

![Fig5](img/1521eddbca515b89a5d6f5518cdc110a.png =600x)

### 5.5.2 Active Learning without Held-out Data

active learningが人のアノテーションの負荷を下げることができるのか調べた。50個のラベルが付けられたセットから始める。何ラウンドで要求された性能(ここでは500個のラベル付データでOpenTagで学習した性能)に到達するか。Tag Flipが前のセクションで性能が良かったので使っている。Active learningだと50個から始めて150個まで増やせば、500個で最初にまとめて学習した結果と同程度の性能が出せている

![Fig6](img/1cd680d8f5d64a4dae7261938e6651a5.png)

# 6 RELATED WORK
- Rule-based extraction techniques [21]
- rule-based and linguistic approaches leveraging syn-tactic structure of sentences to extract dependency relations[3, 18]
- NER system was built to annotate brands in product listings of apparel products[25]
  - They used seed dictionaries containing over **6,000** known brands for bootstrapping
- Neaural Networks for sequence taggingは少ない
  - A multi-label multi-class Perceptron classifer for NER is used[16]
  - LSTM-CRF model is used[13]
  - They used **37,000** manually labeled search queries to train

この辺を見ると既存の研究の学習に必要なタグがかなり多い(というかOpenTagが極端に少ない？)ように見える

- Early attempts include [9, 23], which apply feed-forward neural networks (**FFNN**) and LSTM to NER tasks
- [5] combine deep FFNN and word embedding [19] to explore many NLP tasks including POS tagging, chunking and NER
- **Character-level CNNs** were integrated [26] to augment feature representation, and their model was later enhanced by LSTM [4]
- Huang et al. [11] adopts CRF with BiLSTM for jointly modeling sequence tagging
- Lample et al. [15] use BiLSTM to encode both **character-level and word-level feature**
- Ma et al. [17] replace the **character-level** model with CNNs

Currently, BiLSTM-CRF models as above is state-of-the-art for NER.

- Bahdanau et al. [1] successfully applied attention for alignment in NMT systems
- Early active learning for sequence labeling research [7, 27] em- ploy least confidence (LC) sampling strategies

# 7 CONCLUSIONS
- BiLSTM, CRF, Attentionを活用したend-to-endのタグ付与システムOpenTagを示した。これはタイトルや説明文、箇条書きから未知の属性も取得できる
- OpenTagは辞書や手作業で作成した特徴を学習時に必要としない
- OpenTagの特徴
  - Open World Assumption : 学習時にない新しい属性値を見つけることができる。マルチワードの属性値やマルチ属性の抽出でも同様
  - Irregular structure and sparce context : 文法構造が欠落していたり文脈がスパースでも利用できる
  - Limited annotated data : 学習データが少なくて済む。active learningによって人の負荷も少ない
  - Interpretability : attentionメカニズムを用いてデバッグを行うことができる
- 現実のデータを利用してたった150個(3.3倍の効果。多分500個に対してなので5.5.2の結果に基づいていると思われる)のアノテーションサンプルで新しい属性を発見でき、83%という高いF値を出せる

# 略語
| term | discription       |
| ---- | ----------------- |
| DS   | disjoint training |
| LC   | least confidence  |
| TF   | tag flip          |

# 変数
a : Attribute
A : Attention Matrixg
U : unlabeled pool
H : blind held-out test set
L : initial labeled set
