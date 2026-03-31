# 採点基準書

## 1. 概要

病理レポートバリデーション実験の結果を採点するための基準書。各結果ファイル（`out/results/{condition}/{model}/{case}.md`）に対して、Gold Standard（GS）との照合を行い、`<note>` と `<score>` ブロックを追記する。

### 入力データの構造

各結果ファイルには以下が含まれる：

- YAML frontmatter: model, condition, finish_reason, tokens, duration 等
- `<findings>`: 報告書の所見（モデルがエコーバックしたもの）
- `<diagnosis>`: 診断文（同上）
- `<invalidities>`: モデルが検出した問題点のリスト
- `<gold_standard>`: 正解ラベル（1症例1件）

### 出力形式

結果ファイルの末尾に `<note>` と `<score>` の2ブロックを追記する。

#### `<note>`: 判定の対応付け

`<invalidities>` 内の各指摘行に1対1で対応させ、それぞれがTP/FP-Relevant/FP-Spuriousのいずれかを記す。`<invalidities>` の内容でない行（前置き、空行、途中思考）は無視する。

```
<note>
[Inconsistency] pStageがIIIb... → TP(content-only) GSはRuleViolation
[Deficiency] cM0... → FP-relevant
[Deficiency] 腸壁区分... → FP-relevant
[Deficiency] INF根拠... → FP-relevant
[Deficiency] 肛門縁距離... → FP-relevant
</note>
```

書式: `元の行（省略可） → 判定ラベル 補足（任意）`

判定ラベル:
- `TP(exact)`: GSの内容に合致し、タグも一致
- `TP(content-only)`: GSの内容に合致するが、タグが不一致
- `FP-relevant`: 医学的に妥当だがGSスコープ外
- `FP-spurious`: 事実誤認・でたらめ
- `duplicate`: 他の行と同一の指摘の重複（カウント対象外）
- `not-a-finding`: 指摘ではない行（前置き・思考過程等、カウント対象外）

#### `<score>`: 集計用の構造化データ

```yaml
<score>
status: valid
detection: tp-content-only
fp_relevant: 4
fp_spurious: 0
</score>
```

| フィールド | 型 | 値 | 説明 |
|---|---|---|---|
| status | string | `valid` / `error` | 出力の有効性 |
| detection | string | `tp-exact` / `tp-content-only` / `fn` / `fn-clean` | GS検出結果 |
| fp_relevant | int | 0以上 | FP-Relevant件数（duplicate除く） |
| fp_spurious | int | 0以上 | FP-Spurious件数（duplicate除く） |

---

## 2. Step 0: 出力の有効性判定（status）

`<invalidities>` の内容を確認し、以下の基準で判定する。

### `valid` とする条件

以下のいずれかを満たせば有効：

- `[RuleViolation]` `[Deficiency]` `[Inconsistency]` `[Typo]` のいずれかのタグで始まる行が1行以上ある
- 「問題なし」「問題ありません」等、問題がない旨を明示する出力

タグ行の前後に前置き（「以下の問題点を指摘します。」等）や空行がある場合も、タグ行自体が存在すれば `valid` とする。

### `error` とする条件

以下のいずれかに該当すれば無効：

- **タグ形式の不追従**: `[Tag] description` 形式の行が1行もなく、自然文で指摘が書かれている
- **構造破壊**: `</findings>` や `</diagnosis>` 等のHTMLタグが `<invalidities>` 内に混入している
- **instructリーク**: プロンプトの指示文がそのまま出力に含まれている
- **出力の空白/欠損**: `<invalidities>` が空で、「問題なし」の明示もない

`status: error` の場合、`detection` は `fn`、`fp_relevant` と `fp_spurious` は `0`。`<note>` にはerrorの理由を記す。

### 判定例

**valid（前置きあり）:**
```
<invalidities>
報告書を検証した結果、以下の問題点を指摘します。

[RuleViolation] pStageがIIIbと記載されているが...
[Deficiency] 遠隔転移の記載が...
</invalidities>
```

**valid（問題なし）:**
```
<invalidities>
問題なし
</invalidities>
```

**error（構造破壊 - sip-jmed-13b典型例）:**
```
<invalidities>
[Typo] "pTis" の表記は...
</findings>
[RuleViolation] ...
</invalidities>
```
→ `</findings>` が invalidities 内に混入。構造が破壊されているため error。

**error（タグ不追従）:**
```
<invalidities>
この報告書にはいくつかの問題があります。まず、Stageの記載が...
次に、組織型の...
</invalidities>
```
→ `[Tag]` 形式が1行もないため error。

---

## 3. Step 1: GS検出の判定（detection）

`status: valid` の場合、`<invalidities>` 内のタグ行が `<gold_standard>` の問題を実質的に指摘しているかを判定する。

### 内容マッチングの基準

GSの問題の**核心**を捉えているかで判定する。表現や詳細度の違いは問わない。

例: GS `[RuleViolation] pT2 pN1a M0はpStage IIIaだがIIIbと記載されている`

- **マッチする**: 「pStageがIIIbだがpT2 N1aならIIIaが正しい」「Stage分類が誤っている」
- **マッチしない**: 「cM0の記載がない」（Stageの問題ではなくM記載の欠落を指摘）

複数行ある場合、いずれか1行でもマッチすればTPとする。

### タグ判定

内容がマッチした行のタグがGSのタグと一致するかで、Exact/ContentOnly を区別する。

- **TP-Exact**: 内容マッチ＋タグ一致
- **TP-ContentOnly**: 内容マッチ＋タグ不一致

複数行がGSにマッチする場合、いずれか1行でもタグが一致していれば TP-Exact とする。

### FN の判定

- **fn**: 有効な出力だがGSの問題を指摘する行がない
- **fn-clean**: fnのうち、モデルが「問題なし」と明示的に宣言した場合

---

## 4. Step 2: FP の分類

GSに対応しない余剰のタグ行を、FP-Relevant と FP-Spurious に分類する。

### FP-Relevant

**医学的に妥当な指摘だが、今回のGold Standardのスコープ（大腸癌取扱い規約第9版に基づく評価）の範囲外。**

具体例：
- cM0の記載がない（規約上必須だが今回のGSでは対象外の症例）
- 腸壁区分の記載がない
- WHO分類の組織型記載がない
- RM距離の具体的数値がない
- INFの根拠が所見本文にない（サマリーのみに記載）
- pR0やpCurAの接頭辞pについての指摘
- 直腸癌での肛門縁距離の記載がない

### FP-Spurious

**事実と異なる、医学的に誤っている、または意味不明な指摘。**

具体例：
- 正しいStageを誤りと指摘（pT1b N1a M0 → IIIaは正しいのにIIIbが正しいと主張）
- 正しい用語を誤字と指摘（「簇出」は正しい病理用語だが「芽出」の誤りと指摘）
- 存在しないルールを根拠にした指摘（「pT2ではpPMは定義されない」等）
- 切除マージンが腫瘍サイズを上回ることを矛盾と指摘（正常な状況）
- 許容される表記を不正と指摘（V0(E-Ma)のE-Ma表記を非標準と指摘→自施設で標準的に使用）

### カウント方法

- 同一の問題について複数行で言及している場合は1件とカウントし、2行目以降は `<note>` で `duplicate` とする
- 「取り下げます」「再整理します」等の途中思考は `not-a-finding` とする
- 明らかに指摘でない行（前置き、要約、挨拶）も `not-a-finding`

---

## 5. Invalidity 4分類のタグ判定基準

GSのタグとモデル出力のタグの一致判定に使用する。

### 各タグの定義（再掲）

| タグ | 定義 | 検出に必要なもの |
|---|---|---|
| RuleViolation | 記載が存在し、ルールセットに照らして値・形式が不正 | ルールセット参照 |
| Deficiency | 必須項目の記載がレポート内に存在しない | ルールセット参照 |
| Inconsistency | 個々はルール上合法だが、複数箇所を突き合わせると矛盾 | レポート内部の関係推論 |
| Typo | スペルミス・転記ミスなど、記載自体から判定可能 | 記載自体 |

### タグ一致の判定原則

**厳密一致のみを TP-Exact とする。** 以下のいずれのケースもタグ不一致（TP-ContentOnly）となる。

- **RuleViolation → Inconsistency**: Stage計算ミスをInconsistencyと判定。「pTとpNからStageを一意に決定する」のはルール照合であり、正しくはRuleViolation
- **Inconsistency → RuleViolation**: 所見-サマリー間の矛盾をRuleViolationと判定。どちらの記載が正しいかは外部情報に依存するため、正しくはInconsistency
- **Deficiency → RuleViolation**: 必須項目の欠落をRuleViolationと判定。「ないこと」を違反と呼ぶのは自然だが、Deficiency（欠落）はRuleViolation（不正記載）とは方向が逆であり、Deficiencyと正確に言えるべき
- **Typo → RuleViolation**: スペルミスを規約形式違反と判定。Typoは記載自体から判定可能であり、ルール参照は不要
- **Typo → 他の任意のタグ**: Typoの定義は「記載自体から判定可能」であり、他のタグとは検出メカニズムが本質的に異なる

---

## 6. 具体的な判定例

### 例1: 0001 / claude-opus / ruleset — TP-ContentOnly + FP-Relevant多数

GS: `[RuleViolation] pT2 pN1a M0はpStage IIIaだがIIIbと記載されている`

```
<note>
[Inconsistency] pStageがIIIb... → TP(content-only) GSはRuleViolation
[Inconsistency] V0にE-Ma付記が不適切... → FP-relevant
[RuleViolation] pR0の接頭辞p... → FP-relevant
[RuleViolation] pCurAの接頭辞p... → FP-relevant
[Inconsistency] pN1a 252番リンパ節... → not-a-finding 自ら問題なしと結論
[Deficiency] RM距離... → FP-relevant
[Deficiency] cM0... → FP-relevant
[Deficiency] 腸壁区分... → FP-relevant
[Deficiency] INF根拠... → FP-relevant
[Deficiency] 肛門縁距離... → FP-relevant
</note>

<score>
status: valid
detection: tp-content-only
fp_relevant: 8
fp_spurious: 0
</score>
```

### 例2: 0001 / deepseek-v3.2 / zeroshot — FN + FP-Spurious多数

GS: `[RuleViolation] pT2 pN1a M0はpStage IIIaだがIIIbと記載されている`

```
<note>
[RuleViolation] 全角スペース... → FP-spurious
[RuleViolation] 簇出→簇発の誤記... → FP-spurious 簇出は正しい病理用語
[RuleViolation] pT2ではpPMは定義されない... → FP-spurious 存在しないルール
[Inconsistency] Pn1とPn1a... → FP-spurious
[Inconsistency] Ly1aの程度... → FP-spurious
</note>

<score>
status: valid
detection: fn
fp_relevant: 0
fp_spurious: 5
</score>
```

### 例3: 0031 / gpt-oss-120b / zeroshot — TP-Exact

GS: `[Inconsistency] MLH1(-), MSH2(-), MSH6(-), PMS2(-) はすべてのMMRタンパク質の発現消失（dMMR）を示すが、「MSI陰性」と判定されており矛盾する`

```
<note>
[Inconsistency] MLH1,MSH2,MSH6,PMS2すべて陰性だがMSI陰性は矛盾 → TP(exact)
[Typo] 簇出は芽出の誤記 → FP-spurious 簇出は正しい病理用語
</note>

<score>
status: valid
detection: tp-exact
fp_relevant: 0
fp_spurious: 1
</score>
```

### 例4: 0046 / deepseek-v3.2 / zeroshot — TP-Exact + 混合FP

GS: `[Typo] diagnosisに英語スペルミスがある: "Adenocaricnoma"（正: Adenocarcinoma）`

```
<note>
[Typo] Adenocaricnoma → Adenocarcinoma → TP(exact)
[Typo] Metastatic adenocaricnoma → TP(exact) duplicate 同一Typoの別出現
[RuleViolation] pStage IIIaはIIIbが正しい → FP-spurious IIIaが正しい
[Inconsistency] 虫垂の確認ができない... → FP-relevant
[Inconsistency] pDM0(25mm)と断端露出が矛盾... → FP-spurious 追加切除で陰性は妥当
</note>

<score>
status: valid
detection: tp-exact
fp_relevant: 1
fp_spurious: 2
</score>
```

### 例5: sip-jmed-13b / 構造破壊 — Error

```
<note>
</findings>が<invalidities>内に混入。構造破壊のためerror
</note>

<score>
status: error
detection: fn
fp_relevant: 0
fp_spurious: 0
</score>
```

---

## 7. E-Ma表記に関する特記事項

自施設ではElastica-Masson染色を「E-Ma」と略記するのが標準的である。規約の記載例にはEVG等しか含まれないため、多くのモデルが「E-Ma」表記をFPとして指摘するが、これは自施設の慣習として正当な表記である。

- 「E-Ma」を非標準と指摘 → **FP-Spurious**（自施設では標準）
- 「V0に染色名を付記するのは不適切」→ **FP-Relevant**（陰性確認の記載として許容されうるが、議論の余地あり）
