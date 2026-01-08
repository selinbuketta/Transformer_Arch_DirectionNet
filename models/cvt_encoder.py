# coding=utf-8
"""v4
Cross-View Transformer encoder for DirectionNet on 256x256 inputs.

- Conv stem: 256x256 -> 64x64 feature maps.
- Patch projection with patch_size=4 on 64x64 -> 16x16 = 256 tokens per image.
- CLS token per view -> 257 tokens per image.
- Per-view Transformer blocks (self-attention + MLP).
- One bidirectional token-level cross-view attention block (cheap but powerful).
- CLS tokens also cross-attend to other-view tokens.
- Output: [B, 1, 1, 1024], with self.inplanes = 1024 (unchanged for decoder).
"""

from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Conv2D, Layer, Dense, Dropout, LayerNormalization


def gelu(x):
  """GELU activation (approximation) compatible with TF1 graph mode."""
  c0 = tf.constant(0.5, dtype=x.dtype)
  c1 = tf.constant(1.0, dtype=x.dtype)
  c2 = tf.constant(0.044715, dtype=x.dtype)
  sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)
  return c0 * x * (c1 + tf.tanh(sqrt_2_over_pi * (x + c2 * tf.pow(x, 3))))


# ---------- Lightweight downsampling ConvStem ----------

class ConvStem(Layer):
  """Downsampling stem: 256x256 -> 64x64 feature map."""
  def __init__(self, embed_dim, regularization=0.001, **kwargs):
    super(ConvStem, self).__init__(**kwargs)

    # 256x256 -> 128x128
    self.conv1 = Conv2D(
        filters=embed_dim // 4,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    # 128x128 -> 64x64
    self.conv2 = Conv2D(
        filters=embed_dim // 2,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    # 64x64 -> 64x64 refine
    self.conv3 = Conv2D(
        filters=embed_dim,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.norm3 = LayerNormalization(epsilon=1e-6)

  def call(self, x, training=False):
    x = gelu(self.norm1(self.conv1(x)))
    x = gelu(self.norm2(self.conv2(x)))
    x = gelu(self.norm3(self.conv3(x)))
    return x


# ---------- Attention blocks (with optional relative position bias) ----------

class MultiHeadSelfAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0,
               use_rpb=True, **kwargs):
    super(MultiHeadSelfAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.dropout_rate = dropout_rate
    self.use_rpb = use_rpb

    self.qkv = Dense(3 * embed_dim, use_bias=False)
    self.proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

    if self.use_rpb:
      self.rel_mlp = Dense(num_heads, use_bias=False)

  def _relative_position_bias(self, T):
    positions = tf.cast(tf.range(T), tf.float32)      # [T]
    rel = positions[tf.newaxis, :] - positions[:, tf.newaxis]  # [T,T]
    rel = rel[..., tf.newaxis]                        # [T,T,1]
    bias = self.rel_mlp(rel)                          # [T,T,H]
    bias = tf.transpose(bias, [2, 0, 1])              # [H,T,T]
    bias = bias[tf.newaxis, ...]                      # [1,H,T,T]
    return bias

  def call(self, x, training=False):
    # x: [B,T,C]
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]

    qkv = self.qkv(x)  # [B,T,3C]
    q, k, v = tf.split(qkv, 3, axis=-1)

    def reshape_heads(y):
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])  # [B,H,T,D]

    q = reshape_heads(q)
    k = reshape_heads(k)
    v = reshape_heads(v)

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,T,T]
    if self.use_rpb:
      rpb = self._relative_position_bias(T)
      attn_logits = attn_logits + rpb

    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v)                             # [B,H,T,D]
    out = tf.transpose(out, [0, 2, 1, 3])                # [B,T,H,D]
    out = tf.reshape(out, [B, T, self.embed_dim])
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MultiHeadCrossAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0,
               use_rpb=True, **kwargs):
    super(MultiHeadCrossAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.dropout_rate = dropout_rate
    self.use_rpb = use_rpb

    self.q = Dense(embed_dim, use_bias=False)
    self.kv = Dense(2 * embed_dim, use_bias=False)
    self.proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

    if self.use_rpb:
      self.rel_mlp = Dense(num_heads, use_bias=False)

  def _relative_position_bias(self, Tq, Tk):
    positions_q = tf.cast(tf.range(Tq), tf.float32)   # [Tq]
    positions_k = tf.cast(tf.range(Tk), tf.float32)   # [Tk]
    rel = positions_q[:, tf.newaxis] - positions_k[tf.newaxis, :]  # [Tq,Tk]
    rel = rel[..., tf.newaxis]                        # [Tq,Tk,1]
    bias = self.rel_mlp(rel)                          # [Tq,Tk,H]
    bias = tf.transpose(bias, [2, 0, 1])              # [H,Tq,Tk]
    bias = bias[tf.newaxis, ...]                      # [1,H,Tq,Tk]
    return bias

  def call(self, x_q, x_kv, training=False):
    # x_q: [B,Tq,C], x_kv: [B,Tk,C]
    B = tf.shape(x_q)[0]
    Tq = tf.shape(x_q)[1]
    Tk = tf.shape(x_kv)[1]

    q = self.q(x_q)
    kv = self.kv(x_kv)
    k, v = tf.split(kv, 2, axis=-1)

    def reshape_heads(y, T):
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])

    q = reshape_heads(q, Tq)
    k = reshape_heads(k, Tk)
    v = reshape_heads(v, Tk)

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,Tq,Tk]
    if self.use_rpb:
      rpb = self._relative_position_bias(Tq, Tk)
      attn_logits = attn_logits + rpb

    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v)                             # [B,H,Tq,D]
    out = tf.transpose(out, [0, 2, 1, 3])                # [B,Tq,H,D]
    out = tf.reshape(out, [B, Tq, self.embed_dim])
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MLP(Layer):
  def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(MLP, self).__init__(**kwargs)
    hidden = int(embed_dim * mlp_ratio)
    self.fc1 = Dense(hidden, activation=None)
    self.fc2 = Dense(embed_dim)
    self.drop = Dropout(dropout_rate)

  def call(self, x, training=False):
    x = self.fc1(x)
    x = gelu(x)
    x = self.drop(x, training=training)
    x = self.fc2(x)
    x = self.drop(x, training=training)
    return x


class TransformerBlock(Layer):
  """Per-view Transformer block (self-attention + MLP)."""

  def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
               dropout_rate=0.0, **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.mlp = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, x, training=False):
    x = x + self.attn(self.norm1(x), training=training)
    x = x + self.mlp(self.norm2(x), training=training)
    return x


class CrossViewTokenBlock(Layer):
  """Bidirectional token-level cross-view attention + MLP."""

  def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
               dropout_rate=0.0, **kwargs):
    super(CrossViewTokenBlock, self).__init__(**kwargs)
    # cross-attention for tokens
    self.norm_q1 = LayerNormalization(epsilon=1e-6)
    self.norm_kv1 = LayerNormalization(epsilon=1e-6)
    self.norm_q2 = LayerNormalization(epsilon=1e-6)
    self.norm_kv2 = LayerNormalization(epsilon=1e-6)
    self.cross_12 = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)
    self.cross_21 = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)

    # feed-forward
    self.ffn_norm1 = LayerNormalization(epsilon=1e-6)
    self.ffn_norm2 = LayerNormalization(epsilon=1e-6)
    self.ffn1 = MLP(embed_dim, mlp_ratio, dropout_rate)
    self.ffn2 = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, tokens1, tokens2, training=False):
    # Cache old states for symmetric cross-attention.
    t1_old = tokens1
    t2_old = tokens2

    q1 = self.norm_q1(t1_old)
    kv2 = self.norm_kv1(t2_old)
    q2 = self.norm_q2(t2_old)
    kv1 = self.norm_kv2(t1_old)

    tokens1 = tokens1 + self.cross_12(q1, kv2, training=training)
    tokens2 = tokens2 + self.cross_21(q2, kv1, training=training)

    f1 = self.ffn_norm1(tokens1)
    f2 = self.ffn_norm2(tokens2)
    tokens1 = tokens1 + self.ffn1(f1, training=training)
    tokens2 = tokens2 + self.ffn2(f2, training=training)
    return tokens1, tokens2


# ---------- CrossViewTransformerEncoder ----------

class CrossViewTransformerEncoder(keras.Model):
  """Vision-style Cross-View Transformer encoder for DirectionNet."""

  def __init__(self,
               embed_dim=384,
               num_layers=4,
               num_heads=6,
               mlp_ratio=4.0,
               patch_size=4,
               dropout_rate=0.1,
               regularization=0.001):
    super(CrossViewTransformerEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.inplanes = 1024
    self.patch_size = patch_size

    # 256x256 -> 64x64
    self.stem = ConvStem(embed_dim, regularization=regularization)

    # 64x64 -> 16x16 patches (patch_size=4) -> 256 tokens
    self.patch_proj = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    # CLS token shared across views
    self.cls_token = self.add_weight(
        name='cls_token',
        shape=[1, 1, embed_dim],
        initializer=tf.random_normal_initializer(stddev=0.02))

    # Per-view transformer blocks
    self.blocks = [
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]

    # One token-level cross-view block
    self.cross_tokens = CrossViewTokenBlock(
        embed_dim, num_heads, mlp_ratio, dropout_rate)

    # CLS-only cross-attention (cheap, extra fusion)
    self.cross_attn_cls = MultiHeadCrossAttention(
        embed_dim, num_heads, dropout_rate)

    self.norm = LayerNormalization(epsilon=1e-6)
    self.proj_to_decoder = Dense(1024)

  def _flatten_patches(self, x):
    # x: [B, H, W, C] -> tokens: [B, H*W, C], also return H,W
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]
    tokens = tf.reshape(x, [B, H * W, C])
    return tokens, H, W

  def _sinusoidal_positional(self, H, W):
    """1D sinusoidal positional encoding over H*W tokens."""
    N = H * W
    positions = tf.cast(tf.range(N), tf.float32)[:, tf.newaxis]  # [N,1]
    dim = self.embed_dim
    i = tf.cast(tf.range(dim), tf.float32)[tf.newaxis, :]        # [1,dim]
    angle_rates = 1.0 / tf.pow(10000.0,
                               2 * (tf.floor(i / 2.0)) / tf.cast(dim, tf.float32))
    angles = positions * angle_rates                              # [N,dim]
    sin = tf.sin(angles[:, 0::2])
    cos = tf.cos(angles[:, 1::2])
    pos = tf.concat([sin, cos], axis=-1)                          # [N,dim]
    pos = pos[:, :dim]
    return pos  # [N,dim]

  def _add_positional(self, tokens, H, W):
    pos = self._sinusoidal_positional(H, W)      # [N, C]
    pos = tf.expand_dims(tf.cast(pos, tokens.dtype), axis=0)  # [1,N,C]
    return tokens + pos

  def call(self, img1, img2, training=False):
    # Conv stem: 256x256 -> 64x64
    f1 = self.stem(img1, training=training)
    f2 = self.stem(img2, training=training)

    # Patch projection: 64x64 -> 16x16 -> 256 tokens
    p1 = self.patch_proj(f1)
    p2 = self.patch_proj(f2)

    t1, H1, W1 = self._flatten_patches(p1)  # [B, 256, C]
    t2, H2, W2 = self._flatten_patches(p2)

    t1 = self._add_positional(t1, H1, W1)
    t2 = self._add_positional(t2, H2, W2)

    B = tf.shape(t1)[0]
    cls = tf.tile(self.cls_token, [B, 1, 1])      # [B,1,C]

    x1 = tf.concat([cls, t1], axis=1)             # [B, 257, C]
    x2 = tf.concat([cls, t2], axis=1)             # [B, 257, C]

    # Per-view transformer (self-attention only)
    for blk in self.blocks:
      x1 = blk(x1, training=training)
      x2 = blk(x2, training=training)

    # Separate CLS and tokens
    cls1, tok1 = x1[:, :1, :], x1[:, 1:, :]
    cls2, tok2 = x2[:, :1, :], x2[:, 1:, :]

    # Token-level cross-view attention
    tok1, tok2 = self.cross_tokens(tok1, tok2, training=training)

    # Re-attach CLS
    x1 = tf.concat([cls1, tok1], axis=1)
    x2 = tf.concat([cls2, tok2], axis=1)

    # CLS cross-attends to other-view tokens (extra fusion)
    cls1 = x1[:, :1, :]
    cls2 = x2[:, :1, :]
    tok1 = x1[:, 1:, :]
    tok2 = x2[:, 1:, :]

    cls1 = cls1 + self.cross_attn_cls(cls1, tok2, training=training)
    cls2 = cls2 + self.cross_attn_cls(cls2, tok1, training=training)

    # Average CLS from both views and project to decoder interface
    cls_avg = 0.5 * (cls1[:, 0, :] + cls2[:, 0, :])  # [B,C]
    cls_avg = self.norm(cls_avg)

    y = self.proj_to_decoder(cls_avg)                # [B,1024]
    y = y[:, tf.newaxis, tf.newaxis, :]              # [B,1,1,1024]
    return tf.identity(y)
