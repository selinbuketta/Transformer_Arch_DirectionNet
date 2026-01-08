# coding=utf-8
"""
Cross-View Transformer (CVT) encoder for DirectionNet.

Design goals:
- Drop-in replacement for SiameseEncoder in model.py
- Keep decoder interface the same: output [B, 1, 1, 1024] with attribute `inplanes = 1024`
- Flow: Patch embedding -> per-view self-attn -> bi-directional cross-attn (I0<->I1)
        -> average CLS tokens -> 512-D -> project to 1024 and reshape to 1x1

Note: Implemented with tf.compat.v1.keras to match the codebase.
"""
from tensorflow.compat.v1 import keras
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import Conv2D, Layer, Dense, Dropout, LayerNormalization


def gelu(x):
  """GELU activation (approximation) compatible with TF1 graph mode.

  Uses tanh approximation from Hendrycks & Gimpel (2016):
    0.5 * x * (1 + tanh(\sqrt(2/\pi) * (x + 0.044715 x^3)))
  """
  c0 = tf.constant(0.5, dtype=x.dtype)
  c1 = tf.constant(1.0, dtype=x.dtype)
  c2 = tf.constant(0.044715, dtype=x.dtype)
  sqrt_2_over_pi = tf.constant(0.7978845608028654, dtype=x.dtype)  # sqrt(2/pi)
  return c0 * x * (c1 + tf.tanh(sqrt_2_over_pi * (x + c2 * tf.pow(x, 3))))


class MultiHeadSelfAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
    super(MultiHeadSelfAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.qkv = Dense(3 * embed_dim, use_bias=False)
    self.proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

  def call(self, x, training=False):
    # x: [B, T, C]
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    C = x.shape.as_list()[-1]
    qkv = self.qkv(x)  # [B, T, 3C]
    q, k, v = tf.split(qkv, 3, axis=-1)

    def reshape_heads(y):
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])  # [B, H, T, D]

    q = reshape_heads(q)
    k = reshape_heads(k)
    v = reshape_heads(v)

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,T,T]
    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v)  # [B,H,T,D]
    out = tf.transpose(out, [0, 2, 1, 3])  # [B,T,H,D]
    out = tf.reshape(out, [B, T, self.embed_dim])
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MultiHeadCrossAttention(Layer):
  def __init__(self, embed_dim, num_heads, dropout_rate=0.0, **kwargs):
    super(MultiHeadCrossAttention, self).__init__(**kwargs)
    assert embed_dim % num_heads == 0
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.q = Dense(embed_dim, use_bias=False)
    self.kv = Dense(2 * embed_dim, use_bias=False)
    self.proj = Dense(embed_dim, use_bias=False)
    self.drop = Dropout(dropout_rate)

  def call(self, x_q, x_kv, training=False):
    # x_q: [B, Tq, C], x_kv: [B, Tk, C]
    B = tf.shape(x_q)[0]
    Tq = tf.shape(x_q)[1]
    Tk = tf.shape(x_kv)[1]
    C = x_q.shape.as_list()[-1]

    q = self.q(x_q)
    kv = self.kv(x_kv)
    k, v = tf.split(kv, 2, axis=-1)

    def reshape_heads(y, T):
      y = tf.reshape(y, [B, T, self.num_heads, self.head_dim])
      return tf.transpose(y, [0, 2, 1, 3])  # [B,H,T,D]

    q = reshape_heads(q, Tq)
    k = reshape_heads(k, Tk)
    v = reshape_heads(v, Tk)

    attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale  # [B,H,Tq,Tk]
    attn = tf.nn.softmax(attn_logits, axis=-1)
    attn = self.drop(attn, training=training)
    out = tf.matmul(attn, v)  # [B,H,Tq,Tk]x[B,H,Tk,D]->[B,H,Tq,D]
    out = tf.transpose(out, [0, 2, 1, 3])  # [B,Tq,H,D]
    out = tf.reshape(out, [B, Tq, self.embed_dim])
    out = self.proj(out)
    out = self.drop(out, training=training)
    return out


class MLP(Layer):
  def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(MLP, self).__init__(**kwargs)
    hidden = int(embed_dim * mlp_ratio)
    # Avoid tf.nn.gelu under tf.compat.v1; apply GELU manually in call().
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
  def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout_rate=0.0, **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.mlp = MLP(embed_dim, mlp_ratio, dropout_rate)

  def call(self, x, training=False):
    x = x + self.attn(self.norm1(x), training=training)
    x = x + self.mlp(self.norm2(x), training=training)
    return x


class CrossViewTransformerEncoder(keras.Model):
  def __init__(self,
               embed_dim=512,
               num_layers=4,
               num_heads=8,
               mlp_ratio=4.0,
               patch_size=16,
               dropout_rate=0.0,
               regularization=0.01):
    super(CrossViewTransformerEncoder, self).__init__()
    self.embed_dim = embed_dim
    self.inplanes = 1024  # to match decoder expectation

    # Patch embedding via conv
    self.patch_embed = Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))

    # Positional embedding (added to tokens, incl. CLS)
    self.pos_embed = None  # lazily initialized based on input HxW
    self.cls_token = self.add_weight(
        name='cls_token', shape=[1, 1, embed_dim],
        initializer=tf.random_normal_initializer(stddev=0.02))

    self.blocks = [
        TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
        for _ in range(num_layers)
    ]
    # Cross attention for CLS tokens only (lightweight)
    self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)

    self.norm = LayerNormalization(epsilon=1e-6)
    self.proj_to_decoder = Dense(1024)

  def _flatten_patches(self, x):
    # x: [B,H,W,C] -> tokens [B, T, C]
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = x.shape.as_list()[-1]
    tokens = tf.reshape(x, [B, H * W, C])
    return tokens, H, W

  def _add_positional(self, tokens, H, W, training):
    # For TF1 simplicity and to avoid dynamic add_weight shapes, skip positional embedding.
    # You can implement sinusoidal or learned 2D embeddings later if desired.
    return tokens

  def call(self, img1, img2, training=False):
    # Patch embedding
    p1 = self.patch_embed(img1)  # [B, Hp, Wp, C]
    p2 = self.patch_embed(img2)
    t1, H1, W1 = self._flatten_patches(p1)
    t2, H2, W2 = self._flatten_patches(p2)

    # Add positional embeddings
    t1 = self._add_positional(t1, H1, W1, training)
    t2 = self._add_positional(t2, H2, W2, training)

    # Prepend CLS token to each
    B = tf.shape(t1)[0]
    cls = tf.tile(self.cls_token, [B, 1, 1])  # [B,1,C]
    x1 = tf.concat([cls, t1], axis=1)
    x2 = tf.concat([cls, t2], axis=1)

    # Per-view self-attention blocks
    for blk in self.blocks:
      x1 = blk(x1, training=training)
      x2 = blk(x2, training=training)

    # Cross-attention on CLS tokens (bi-directional)
    cls1 = x1[:, :1, :]  # [B,1,C]
    cls2 = x2[:, :1, :]
    tok1 = x1[:, 1:, :]
    tok2 = x2[:, 1:, :]

    cls1 = cls1 + self.cross_attn(cls1, tok2, training=training)
    cls2 = cls2 + self.cross_attn(cls2, tok1, training=training)

    # Average two CLS tokens to a single 512-D vector
    cls_avg = 0.5 * (cls1[:, 0, :] + cls2[:, 0, :])  # [B,C]
    cls_avg = self.norm(cls_avg)

    # Project to 1024 and reshape to 1x1 for the decoder
    y = self.proj_to_decoder(cls_avg)  # [B,1024]
    y = y[:, tf.newaxis, tf.newaxis, :]  # [B,1,1,1024]
    return y
