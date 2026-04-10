# Marp Tips

## bg画像スライドでタイトルを白背景・上付きにする

### やりたいこと

`![bg fit]` で全面背景画像を使うスライドで、タイトル（h2）が画像の中央に黒字で表示されて読めない。白背景付きで上端に配置したい。

### 方法

グローバルstyleにクラスを定義:

```css
section.bg-title h2 {
  background: rgba(255,255,255,0.9);
  padding: 8px 16px;
  margin: 0;
  position: absolute;
  top: 0; left: 0; right: 0;
  z-index: 1;
}
```

スライド側で `_class: bg-title` を指定:

```markdown
## タイトル

![bg fit](image.jpg)

<!--
_class: bg-title
_paginate: false
-->
```

### 注意

- Marpは `![bg]` 使用時に内部で `data-marpit-advanced-background` による多層構造を生成するため、`justify-content` や `display: flex` でのレイアウト制御は効かない
- `position: absolute` で直接配置するのが確実
