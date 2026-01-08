# coding=utf-8
"""

v3
Memory-friendly Cross-View Transformer encoder for DirectionNet.

- Designed for ~4 GB GPUs. not valid
- Output: [B, 1, 1, 1024], self.inplanes = 1024 (unchanged for decoder).
patch_size = 16 
input images = 256x256
number of tokens per image = (256/16)^2 = 256 tokens
Each image 256 tokens + CLS = 257 tokens
Transformer has num_layers = 4 and embed_dim = 512 self-attention complexity per layer = O(T^2)
257 x 257 x 512 =33M operations per block * 4 layers = 132M operations
"""

from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Conv2D, Layer, Dense, Dropout, LayerNormalization


def gelu(x):
  c0 = tf.constant(0.5, dtype=x.dtype)
  c1 = tf.constant(1.0, dtype=x.dtype)
  c2 = tf.constant(0.044715, dtype=x.dtype)
  sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)
  return c0 * x * (c1 + tf.tanh(sqrt_2_over_pi * (x + c2 * tf.pow(x, 3))))


# ---------- Lightweight downsampling ConvStem ----------

class ConvStem(Layer):
  """Downsampling stem to keep memory small on 4GB GPUs."""
  def __init__(self, embed_dim, regularization=0.01, **kwargs):
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


# ---------- Attention blocks (with RPB) ----------

class MultiHeadSelfAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, use_rpb=True, **kwargs):
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

  def _ensure_layers(self):
    if not hasattr(self, 'qkv'):
      self.qkv = Dense(3 * self.embed_dim, use_bias=False)
    if not hasattr(self, 'proj'):
      self.proj = Dense(self.embed_dim, use_bias=False)
    if not hasattr(self, 'drop'):
      self.drop = Dropout(self.dropout_rate)
    if self.use_rpb and not hasattr(self, 'rel_mlp'):
      self.rel_mlp = Dense(self.num_heads, use_bias=False)

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
    self._ensure_layers()
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
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, use_rpb=True, **kwargs):
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
  def __init__(self, embed_dim, mlp_ratio=3.0, dropout_rate=0.0, **kwargs):
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
  def __init__(self, embed_dim, num_heads, mlp_ratio=3.0, dropout_rate=0.0, **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.mlp = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, x, training=False):
    x = x + self.attn(self.norm1(x), training=training)
    x = x + self.mlp(self.norm2(x), training=training)
    return x


# ---------- CrossViewTransformerEncoder (tiny) ----------

class CrossViewTransformerEncoder(keras.Model):
  def __init__(self,
               embed_dim=192,
               num_layers=2,
               num_heads=3,
               mlp_ratio=3.0,
               patch_size=32,
               dropout_rate=0.0,
               regularization=0.01):
    super(CrossViewTransformerEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.inplanes = 1024
    self.patch_size = patch_size

    self.stem = ConvStem(embed_dim, regularization=regularization)
    self.patch_proj = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    self.cls_token = self.add_weight(
        name='cls_token',
        shape=[1, 1, embed_dim],
        initializer=tf.random_normal_initializer(stddev=0.02))

    self.blocks = [
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]

    # CLS-only cross-attention (cheap)
    self.cross_attn_cls = MultiHeadCrossAttention(
        embed_dim, num_heads, dropout_rate)

    self.norm = LayerNormalization(epsilon=1e-6)
    self.proj_to_decoder = Dense(1024)

  def _flatten_patches(self, x):
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = x.shape.as_list()[-1]
    tokens = tf.reshape(x, [B, H * W, C])
    return tokens, H, W

  def _sinusoidal_positional(self, H, W):
    y = tf.cast(tf.range(H), tf.float32)
    x = tf.cast(tf.range(W), tf.float32)
    yy, xx = tf.meshgrid(y, x, indexing='ij')
    coords = tf.reshape(tf.stack([yy, xx], axis=-1), [H * W, 2])

    half = self.embed_dim // 2
    freq_range = tf.range(0, half, 2, dtype=tf.float32)
    inv_freq = tf.exp(-tf.math.log(10000.0) *
                      freq_range / tf.cast(tf.maximum(1, half), tf.float32))

    def _encode(coord):
      sinusoid_inp = coord * inv_freq
      return tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1)

    pos_y = _encode(coords[:, 0:1])
    pos_x = _encode(coords[:, 1:2])
    pos = tf.concat([pos_y, pos_x], axis=-1)
    pad_dim = tf.maximum(0, self.embed_dim - tf.shape(pos)[-1])
    pos = tf.pad(pos, [[0, 0], [0, pad_dim]])
    return pos[:, :self.embed_dim]

  def _add_positional(self, tokens, H, W):
    pos = self._sinusoidal_positional(H, W)
    pos = tf.expand_dims(tf.cast(pos, tokens.dtype), axis=0)
    return tokens + pos

  def call(self, img1, img2, training=False):
    # stem: 256x256 -> 64x64 feature map
    p1 = self.stem(img1, training=training)
    p2 = self.stem(img2, training=training)

    p1 = self.patch_proj(p1)  # [B, H', W', C]
    p2 = self.patch_proj(p2)

    t1, H1, W1 = self._flatten_patches(p1)
    t2, H2, W2 = self._flatten_patches(p2)

    t1 = self._add_positional(t1, H1, W1)
    t2 = self._add_positional(t2, H2, W2)

    B = tf.shape(t1)[0]
    cls = tf.tile(self.cls_token, [B, 1, 1])

    x1 = tf.concat([cls, t1], axis=1)
    x2 = tf.concat([cls, t2], axis=1)

    # per-view transformer
    for blk in self.blocks:
      x1 = blk(x1, training=training)
      x2 = blk(x2, training=training)

    cls1 = x1[:, :1, :]
    cls2 = x2[:, :1, :]
    tok1 = x1[:, 1:, :]
    tok2 = x2[:, 1:, :]

    # CLS cross-attends to other-view tokens
    cls1 = cls1 + self.cross_attn_cls(cls1, tok2, training=training)
    cls2 = cls2 + self.cross_attn_cls(cls2, tok1, training=training)

    cls_avg = 0.5 * (cls1[:, 0, :] + cls2[:, 0, :])  # [B,C]
    cls_avg = self.norm(cls_avg)

    y = self.proj_to_decoder(cls_avg)                # [B,1024]
    y = y[:, tf.newaxis, tf.newaxis, :]              # [B,1,1,1024]
    return tf.identity(y)
